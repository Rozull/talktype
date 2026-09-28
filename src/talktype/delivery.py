"""`DeliveryThread`: the FIFO worker that turns transcripts into inserted text.

For each dictation, in order: `postprocess()`, the optional local rewrite (with the drift
guard), `history.add_pending`, `Injector.inject()`, `history.mark`, then `on_outcome`.

- Each dictation uses its own settings snapshot (`Dictation.settings`): post-processing, the
  rewrite switch and endpoint, the injection mode and the injection timings.
- A skipped rewrite keeps the post-processed text and reports the reason.
- An empty post-processed text is ``NO_SPEECH``: no history entry, no injection.
- `cancel(id)` is honoured until the injection starts; afterwards it is a no-op.
- History is written before every insert, so a crash mid-insert leaves the text `pending`.
- `inject_text()` (re-paste, tray history "Inserir") injects on the same thread, in FIFO
  order after the queued dictations, with no post-processing, rewrite or history.
- The thread pumps Win32 messages while idle and while injecting, which clipboard delayed
  rendering needs.
- Callbacks run on the delivery thread; consumers marshal them to Qt. The one exception is
  `on_inserting`, fired by a timer thread 500 ms into a slow injection and always before
  that dictation's `on_outcome`.
- Nothing crosses the thread boundary as an exception: failures are logged as
  ``worker_error thread=delivery`` and reported as ``FAILED_INSERT``, with the text left on
  the clipboard and in the history when possible.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from talktype import win32
from talktype.config import InjectionSettings, LlmSettings, Settings
from talktype.dictation import AsrOutcome, Dictation
from talktype.history import HistoryStatus, HistoryStore
from talktype.injector import InjectMode, InjectResult, InjectStatus, message_for
from talktype.llm import LocalRewriter, RewriteSkipped
from talktype.logging_setup import get_logger, log_event
from talktype.postprocess import postprocess
from talktype.strings import Msg

logger = get_logger("delivery")

INSERTING_AFTER_S = 0.500
IDLE_WAIT_MS = 100
WM_APP_WAKE = win32.WM_APP + 0x40


class DeliveryStatus(StrEnum):
    INSERTED = "inserted"
    FAILED_INSERT = "failed_insert"
    NO_SPEECH = "no_speech"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    dictation_id: int
    status: DeliveryStatus
    text: str = ""  # the final text that was (or would have been) inserted
    inject: InjectResult | None = None
    message: Msg | None = None  # from `message_for`
    rewrite_skipped: str | None = None  # a `RewriteSkipped` reason
    post_ms: int = 0
    llm_ms: int = 0
    inject_ms: int = 0
    history_id: str | None = None


class Rewriter(Protocol):
    def rewrite(self, text: str) -> str: ...


class InjectorLike(Protocol):
    def inject(
        self, text: str, mode: InjectMode | str, settings: InjectionSettings | None = None
    ) -> InjectResult: ...


RewriterFactory = Callable[[LlmSettings], Rewriter]


@dataclass(slots=True)
class _DictationJob:
    dictation: Dictation
    outcome: AsrOutcome


@dataclass(slots=True)
class _TextJob:
    text: str
    settings: Settings
    on_result: Callable[[InjectResult], None] | None


_Job = _DictationJob | _TextJob


@dataclass(slots=True)
class _Progress:
    """What one dictation has reached so far (used to salvage it after an error)."""

    text: str
    history_id: str | None = None
    rewrite_skipped: str | None = None
    post_ms: int = 0
    llm_ms: int = 0
    inject_ms: int = 0


class DeliveryThread:
    def __init__(
        self,
        *,
        injector: InjectorLike,
        history: HistoryStore,
        on_outcome: Callable[[DeliveryOutcome], None],
        rewriter_factory: RewriterFactory | None = None,
        on_inject_started: Callable[[int], None] | None = None,
        on_inserting: Callable[[int], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        inserting_after_s: float = INSERTING_AFTER_S,
    ) -> None:
        self._injector = injector
        self._history = history
        self._on_outcome = on_outcome
        self._factory: RewriterFactory = rewriter_factory or self._local_rewriter
        self._on_inject_started = on_inject_started
        self._on_inserting = on_inserting
        self._clock = clock
        self._inserting_after_s = inserting_after_s

        self._cond = threading.Condition()
        self._jobs: deque[_Job] = deque()
        self._waiting: set[int] = set()  # dictations that can still be cancelled
        self._cancelled: set[int] = set()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()

        self._slow_lock = threading.Lock()
        self._slow_id: int | None = None  # the injection that may still report "inserindo"

        self._rewriters_lock = threading.Lock()
        self._rewriters: dict[str, LocalRewriter] = {}  # keyed by the settings as JSON

    # --- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="talktype-delivery", daemon=True
            )
            self._thread.start()
            self._ready.wait(5.0)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Finish the current job and end the thread. Idempotent.

        Dictations still queued are not injected; they are saved as `failed_insert`.
        """
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._wake()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout_s)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- submission (any thread) ------------------------------------------------------

    def submit(self, d: Dictation, outcome: AsrOutcome) -> None:
        """Queue a finished dictation for delivery (FIFO)."""
        with self._cond:
            self._waiting.add(d.id)
            self._jobs.append(_DictationJob(d, outcome))
            self._cond.notify_all()
        self._wake()

    def cancel(self, dictation_id: int) -> None:
        """Drop a dictation that has not started injecting; a no-op afterwards."""
        with self._cond:
            if dictation_id in self._waiting:
                self._cancelled.add(dictation_id)

    def inject_text(
        self,
        text: str,
        settings: Settings,
        on_result: Callable[[InjectResult], None] | None = None,
    ) -> None:
        """Inject `text` as is (re-paste, history insert), after the queued dictations."""
        with self._cond:
            self._jobs.append(_TextJob(text, settings, on_result))
            self._cond.notify_all()
        self._wake()

    def prewarm(self, s: LlmSettings) -> None:
        """Pre-warm the rewriter used for `s` (background, never raises)."""
        try:
            prewarm = getattr(self._factory(s), "prewarm", None)
            if prewarm is not None:
                prewarm()
        except Exception as exc:
            log_event(
                logger,
                "rewrite_prewarm_failed",
                level=logging.WARNING,
                reason="error",
                exc=type(exc).__name__,
            )

    def _wake(self) -> None:
        if self._thread_id:
            with contextlib.suppress(OSError):
                win32.PostThreadMessageW(self._thread_id, WM_APP_WAKE)

    # --- delivery thread --------------------------------------------------------------

    def _loop(self) -> None:
        self._thread_id = win32.GetCurrentThreadId()
        win32.PeekMessageW(win32.MSG(), remove=win32.PM_NOREMOVE)  # create the message queue
        self._ready.set()
        try:
            while (job := self._next()) is not None:
                self._run(job)
            self._save_unsent()
        finally:
            self._close()

    def _next(self) -> _Job | None:
        while True:
            with self._cond:
                if self._stopping:
                    return None
                if self._jobs:
                    return self._jobs.popleft()
            win32.MsgWaitForMultipleObjects([], False, IDLE_WAIT_MS)
            win32.pump_pending_messages()

    def _run(self, job: _Job) -> None:
        if isinstance(job, _DictationJob):
            self._emit(self._deliver(job))
        else:
            self._inject_direct(job)

    def _ms(self, started: float) -> int:
        return round((self._clock() - started) * 1000)

    def _deliver(self, job: _DictationJob) -> DeliveryOutcome:
        did = job.dictation.id
        progress = _Progress(job.outcome.text)
        try:
            return self._pipeline(job, progress)
        except Exception as exc:
            log_event(
                logger,
                "worker_error",
                level=logging.ERROR,
                thread="delivery",
                exc=type(exc).__name__,
                exc_info=True,
            )
            return self._salvage(did, progress)
        finally:
            with self._cond:
                self._waiting.discard(did)
                self._cancelled.discard(did)

    def _cancelled_outcome(self, did: int, p: _Progress) -> DeliveryOutcome:
        log_event(logger, "delivery_cancelled", id=did)
        return DeliveryOutcome(
            did,
            DeliveryStatus.CANCELLED,
            p.text,
            rewrite_skipped=p.rewrite_skipped,
            post_ms=p.post_ms,
            llm_ms=p.llm_ms,
        )

    def _pipeline(self, job: _DictationJob, p: _Progress) -> DeliveryOutcome:
        did = job.dictation.id
        s = job.dictation.settings
        with self._cond:
            cancelled = did in self._cancelled
        if cancelled:
            return self._cancelled_outcome(did, p)
        if job.outcome.kind != "text":
            return DeliveryOutcome(did, DeliveryStatus.NO_SPEECH)

        started = self._clock()
        p.text = postprocess(job.outcome.text, s)
        p.post_ms = self._ms(started)
        if not p.text.strip(" \t"):
            return DeliveryOutcome(did, DeliveryStatus.NO_SPEECH, p.text, post_ms=p.post_ms)

        if s.llm.enabled:
            started = self._clock()
            p.text, p.rewrite_skipped = self._rewrite(did, s.llm, p.text)
            p.llm_ms = self._ms(started)

        with self._cond:  # the last point where a cancel is honoured
            cancelled = did in self._cancelled
            self._waiting.discard(did)
        if cancelled:
            return self._cancelled_outcome(did, p)

        p.history_id = self._history.add_pending(p.text)
        self._call(self._on_inject_started, did)
        started = self._clock()
        timer = self._arm_inserting(did)
        try:
            result = self._injector.inject(p.text, s.injection.mode, s.injection)
        finally:
            self._disarm_inserting(timer)
        p.inject_ms = self._ms(started)

        inserted = result.status == InjectStatus.INSERTED
        self._history.mark(
            p.history_id, HistoryStatus.INSERTED if inserted else HistoryStatus.FAILED_INSERT
        )
        self._log_result(did, s.injection.mode, result)
        return DeliveryOutcome(
            did,
            DeliveryStatus.INSERTED if inserted else DeliveryStatus.FAILED_INSERT,
            p.text,
            inject=result,
            message=message_for(result),
            rewrite_skipped=p.rewrite_skipped,
            post_ms=p.post_ms,
            llm_ms=p.llm_ms,
            inject_ms=p.inject_ms,
            history_id=p.history_id,
        )

    def _rewrite(self, did: int, s: LlmSettings, text: str) -> tuple[str, str | None]:
        """The rewritten text, or `text` and the reason the rewrite was skipped."""
        try:
            output = self._factory(s).rewrite(text)
            if output.strip():
                return output, None
            reason = "empty"
        except RewriteSkipped as exc:
            reason = exc.reason
        except Exception as exc:
            log_event(
                logger,
                "worker_error",
                level=logging.ERROR,
                thread="delivery",
                exc=type(exc).__name__,
                exc_info=True,
            )
            reason = "http_error"
        log_event(logger, "rewrite_skipped", id=did, reason=reason)
        return text, reason

    def _local_rewriter(self, s: LlmSettings) -> Rewriter:
        """The default factory: one cached `LocalRewriter` for the current settings."""
        key = s.model_dump_json()
        with self._rewriters_lock:
            rewriter = self._rewriters.get(key)
            if rewriter is None:
                for old in self._rewriters.values():
                    old.close()
                rewriter = LocalRewriter(s)
                self._rewriters = {key: rewriter}
            return rewriter

    def _salvage(self, did: int, p: _Progress) -> DeliveryOutcome:
        """After an unexpected error: keep the text on the clipboard and in history."""
        on_clipboard = self._put_on_clipboard(p.text)
        try:
            if p.history_id is None:
                p.history_id = self._history.add_pending(p.text)
            self._history.mark(p.history_id, HistoryStatus.FAILED_INSERT)
        except Exception as exc:
            log_event(logger, "history_error", level=logging.ERROR, exc=type(exc).__name__)
        return DeliveryOutcome(
            did,
            DeliveryStatus.FAILED_INSERT,
            p.text,
            message=Msg.INSERT_FAILED_CLIPBOARD if on_clipboard else Msg.INSERT_FAILED_OPEN_HISTORY,
            rewrite_skipped=p.rewrite_skipped,
            post_ms=p.post_ms,
            llm_ms=p.llm_ms,
            inject_ms=p.inject_ms,
            history_id=p.history_id,
        )

    def _inject_direct(self, job: _TextJob) -> None:
        s = job.settings
        try:
            result = self._injector.inject(job.text, s.injection.mode, s.injection)
        except Exception as exc:
            log_event(
                logger,
                "worker_error",
                level=logging.ERROR,
                thread="delivery",
                exc=type(exc).__name__,
                exc_info=True,
            )
            on_clipboard = self._put_on_clipboard(job.text)
            result = InjectResult(InjectStatus.FAILED_UNCONFIRMED, text_on_clipboard=on_clipboard)
        self._log_result("direct", s.injection.mode, result)
        if job.on_result is not None:
            self._call(job.on_result, result)

    def _save_unsent(self) -> None:
        """On stop: dictations that never reached injection go to history, not away."""
        with self._cond:
            jobs, self._jobs = list(self._jobs), deque()
        for job in jobs:
            if not isinstance(job, _DictationJob) or job.outcome.kind != "text":
                continue
            did = job.dictation.id
            with self._cond:
                if did in self._cancelled:
                    continue
            try:
                text = postprocess(job.outcome.text, job.dictation.settings)
                if not text.strip(" \t"):
                    continue
                history_id = self._history.add_pending(text)
                self._history.mark(history_id, HistoryStatus.FAILED_INSERT)
            except Exception as exc:
                log_event(logger, "history_error", level=logging.ERROR, exc=type(exc).__name__)
                continue
            log_event(logger, "delivery_not_sent", id=did)
            self._emit(
                DeliveryOutcome(
                    did,
                    DeliveryStatus.FAILED_INSERT,
                    text,
                    message=Msg.INSERT_FAILED_OPEN_HISTORY,
                    history_id=history_id,
                )
            )

    def _close(self) -> None:
        close = getattr(self._injector, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()
        with self._rewriters_lock:
            for rewriter in self._rewriters.values():
                with contextlib.suppress(Exception):
                    rewriter.close()
            self._rewriters = {}

    # --- helpers ----------------------------------------------------------------------

    def _put_on_clipboard(self, text: str) -> bool:
        put = getattr(self._injector, "put_on_clipboard", None)
        if put is None or not text:
            return False
        try:
            return bool(put(text))
        except Exception:
            return False

    def _log_result(self, did: int | str, mode: str, result: InjectResult) -> None:
        log_event(
            logger,
            "inject_result",
            id=did,
            mode=mode,
            status=result.status,
            restored=result.clipboard_restored,
            partial=result.restore_partial,
        )

    def _emit(self, outcome: DeliveryOutcome) -> None:
        self._call(self._on_outcome, outcome)

    def _call(self, callback: Callable[[Any], None] | None, value: Any) -> None:
        if callback is None:
            return
        try:
            callback(value)
        except Exception as exc:
            log_event(
                logger,
                "worker_error",
                level=logging.ERROR,
                thread="delivery",
                exc=type(exc).__name__,
                exc_info=True,
            )

    def _arm_inserting(self, did: int) -> threading.Timer | None:
        if self._on_inserting is None:
            return None
        with self._slow_lock:
            self._slow_id = did
        timer = threading.Timer(self._inserting_after_s, self._report_inserting, args=(did,))
        timer.daemon = True
        timer.start()
        return timer

    def _report_inserting(self, did: int) -> None:
        with self._slow_lock:  # held while reporting, so `on_outcome` always comes after
            if self._slow_id != did:
                return
            self._slow_id = None
            self._call(self._on_inserting, did)

    def _disarm_inserting(self, timer: threading.Timer | None) -> None:
        if timer is None:
            return
        timer.cancel()
        with self._slow_lock:
            self._slow_id = None
