"""`App`: the controller that wires talktype together.

`App` owns two state machines:
- the app state: ``DOWNLOADING → LOADING → READY ⇄ PAUSED``, plus ``ERROR(reason)``;
- the lifecycle of every `Dictation`, enforced by `App._transition()`.

It turns trigger effects and worker results into recorder, ASR, delivery, overlay, tray and
sound calls, and runs the preview and limit schedule. Worker threads reach it only through
the queued signals of `Bridge`; everything in `App` runs on the Qt GUI thread.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from talktype import autostart, seams
from talktype.asr import Audio, SpeechEngine
from talktype.asr_device import Device
from talktype.asr_worker import (
    AsrWorker,
    DownloadProgress,
    EngineFailed,
    EngineLoading,
    EngineReady,
    GpuReset,
    PreviewResult,
    ReloadFailed,
    VocabTruncated,
)
from talktype.audio import MIN_DBFS, Level, MicError, OpenInfo, Recorder, create_recorder
from talktype.config import ConfigManager, Settings, ValidationReport, atomic_write, resolve_preview
from talktype.delivery import DeliveryOutcome, DeliveryStatus, DeliveryThread
from talktype.dictation import AsrOutcome, Dictation, DictationMode, DictationState
from talktype.history import HistoryStore
from talktype.injector import Injector, InjectResult, InjectStatus, message_for
from talktype.llm import LocalRewriter, is_loopback_endpoint
from talktype.logging_setup import get_logger, log_event, set_log_text
from talktype.overlay import Overlay
from talktype.paths import Paths
from talktype.sounds import Sounds
from talktype.strings import Msg
from talktype.tray import KEY_MICROPHONE, KEY_REWRITE, Tray, TrayState
from talktype.trigger import Effect, trigger_vk
from talktype.winhook import (
    HookEvent,
    HookReinstalled,
    HookThread,
    HotkeyConflict,
    RepasteRequested,
    SessionInterrupted,
)

logger = get_logger("app")

TICK_MS = 20  # resolution of the preview and limit schedule while recording
COUNTDOWN_FROM_S = 30
MUTED_PEAK_DBFS = -60.0  # a no-speech recording quieter than this hints at a muted mic
SLOW_TOTAL_MS = 1500  # dictation_finished above this on CUDA is logged at WARNING

_DOWNLOAD_ERRORS = frozenset(
    {
        Msg.DOWNLOAD_OFFLINE,
        Msg.DOWNLOAD_BLOCKED,
        Msg.DOWNLOAD_NO_SPACE,
        Msg.MODEL_CACHE_NOT_WRITABLE,
    }
)
_MIC_MESSAGES = {"blocked": Msg.MIC_BLOCKED, "no_device": Msg.MIC_NO_DEVICE, "busy": Msg.MIC_BUSY}
_REWRITE_MESSAGES = {
    "too_long": Msg.REWRITE_SKIPPED_TOO_LONG,
    "auth": Msg.REWRITE_SKIPPED_AUTH,
    "unreachable": Msg.REWRITE_SKIPPED_UNREACHABLE,
}


class AppState(StrEnum):
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    PAUSED = "paused"
    ERROR = "error"


_S = DictationState
# The app lifecycle states; INTERRUPTED covers lock and suspend.
TRANSITIONS: dict[DictationState, frozenset[DictationState]] = {
    _S.RECORDING: frozenset({_S.TRANSCRIBING, _S.CANCELLED, _S.INTERRUPTED}),
    _S.TRANSCRIBING: frozenset({_S.POSTPROCESSING, _S.DISCARDED, _S.CANCELLED}),
    _S.POSTPROCESSING: frozenset({_S.INSERTING, _S.FAILED_INSERT, _S.DISCARDED, _S.CANCELLED}),
    _S.INSERTING: frozenset({_S.INSERTED, _S.FAILED_INSERT}),
    _S.INSERTED: frozenset(),
    _S.FAILED_INSERT: frozenset(),
    _S.CANCELLED: frozenset(),
    _S.DISCARDED: frozenset(),
    _S.INTERRUPTED: frozenset(),
}
_TERMINAL = frozenset(s for s, targets in TRANSITIONS.items() if not targets)


# --------------------------------------------------------------------------------------
# Collaborators (the real classes, or fakes in the unit tests)
# --------------------------------------------------------------------------------------


class MsClock(Protocol):
    def now_ms(self) -> int: ...


class MonotonicClock:
    def now_ms(self) -> int:
        return int(time.monotonic() * 1000)


class RecorderLike(Protocol):
    microphone: str

    def open(self) -> OpenInfo: ...
    def stop(self) -> Audio: ...
    def discard(self) -> None: ...
    def snapshot(self, seconds: float | None = None) -> Audio: ...


class AsrLike(Protocol):
    def start(self, model_id: str | None = None, device: str = "auto") -> None: ...
    def stop(self, timeout: float = 5.0) -> None: ...
    def reload(self, model_id: str, device: str = "auto", *, force: bool = False) -> bool: ...
    def submit_final(self, dictation: Dictation, audio: Audio) -> None: ...
    def submit_preview(self, dictation: Dictation, audio: Audio) -> bool: ...
    def drop_previews(self, dictation_id: int) -> None: ...


class DeliveryLike(Protocol):
    def start(self) -> None: ...
    def stop(self, timeout_s: float = 5.0) -> None: ...
    def submit(self, d: Dictation, outcome: AsrOutcome) -> None: ...
    def cancel(self, dictation_id: int) -> None: ...
    def inject_text(
        self,
        text: str,
        settings: Settings,
        on_result: Callable[[InjectResult], None] | None = None,
    ) -> None: ...


class HookLike(Protocol):
    def start(self, timeout_s: float = 5.0) -> None: ...
    def stop(self) -> None: ...
    def abort(self) -> None: ...
    def reset(self) -> None: ...
    def set_enabled(self, enabled: bool) -> None: ...
    def update_settings(self, trigger: Any, repaste_hotkey: str) -> None: ...
    def release_stuck_ctrl(self) -> bool: ...


class OverlayLike(Protocol):
    @property
    def hwnd(self) -> int: ...

    def show_recording(self, dictation_id: int, *, handsfree: bool = False) -> None: ...
    def set_level(self, level: float) -> None: ...
    def set_remaining(self, remaining_s: float | None) -> None: ...
    def show_preview(self, dictation_id: int, seq: int, text: str) -> bool: ...
    def show_transcribing(self) -> None: ...
    def show_inserting(self) -> None: ...
    def show_outcome(self, status: str | Msg) -> None: ...
    def show_message(self, text: str) -> None: ...
    def hide_now(self) -> None: ...


class SoundsLike(Protocol):
    def play_start(self) -> None: ...
    def play_end(self) -> None: ...
    def configure(self, settings: Any) -> None: ...


class Prewarmer(Protocol):
    def prewarm(self) -> None: ...


# --------------------------------------------------------------------------------------
# Thread bridge
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InjectStarted:
    dictation_id: int


@dataclass(frozen=True, slots=True)
class Inserting:
    dictation_id: int


@dataclass(frozen=True, slots=True)
class DirectInjectDone:
    """A re-paste or history insert finished on the delivery thread."""

    source: str
    result: InjectResult


class Bridge(QObject):
    """Queued signals that carry worker-thread callbacks onto the Qt GUI thread.

    Emitting from another thread queues the call; emitting on the GUI thread (unit tests)
    delivers it directly.
    """

    effects = Signal(object)  # tuple[Effect, ...] from the hook thread
    hook_event = Signal(object)  # HookEvent
    worker_event = Signal(object)  # AsrWorker events
    delivery_event = Signal(object)  # DeliveryOutcome | InjectStarted | Inserting | ...
    level = Signal(object)  # audio.Level from the PortAudio thread
    device_lost = Signal()

    def post_effects(self, effects: tuple[Effect, ...]) -> None:
        self.effects.emit(effects)

    def post_inject_started(self, dictation_id: int) -> None:
        self.delivery_event.emit(InjectStarted(dictation_id))

    def post_inserting(self, dictation_id: int) -> None:
        self.delivery_event.emit(Inserting(dictation_id))


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Run:
    """Controller-side bookkeeping for one dictation."""

    dictation: Dictation
    preview: bool
    deadline_ms: int
    next_preview_ms: int
    finished_ms: int | None = None
    record_ms: int = 0
    speech_s: float = 0.0
    asr_ms: int = 0
    peak_dbfs: float = MIN_DBFS
    mic_lost: bool = False
    last_seq: int = 0


@dataclass(slots=True)
class _State:
    onboarding_shown: bool = False


@dataclass(slots=True)
class _Deferred:
    """Config changes waiting for the active dictations to finish."""

    ops: list[tuple[str, Path | None]] = field(default_factory=list)


class App(QObject):
    def __init__(
        self,
        *,
        config: ConfigManager,
        history: HistoryStore,
        recorder: RecorderLike,
        asr: AsrLike,
        delivery: DeliveryLike,
        hook: HookLike,
        overlay: OverlayLike,
        tray: Tray,
        sounds: SoundsLike,
        bridge: Bridge | None = None,
        clock: MsClock | None = None,
        rewriter_factory: Callable[[Any], Prewarmer] | None = None,
        open_path: Callable[[Path], None] | None = None,
        quit_app: Callable[[], None] | None = None,
        autostart_venv: Path | None = None,
        injector: Any = None,
    ) -> None:
        super().__init__(None)
        self.config = config
        self.history = history
        self.recorder = recorder
        self.asr = asr
        self.delivery = delivery
        self.hook = hook
        self.overlay = overlay
        self.tray = tray
        self.sounds = sounds
        self.bridge = bridge or Bridge()
        self.paths = config.paths
        self._clock: MsClock = clock or MonotonicClock()
        self._rewriter_factory = rewriter_factory or LocalRewriter
        self._open_path = open_path or _startfile
        self._quit = quit_app or _quit_qt
        self._autostart_venv = autostart_venv
        self._injector = injector  # its trigger key follows reloads

        self.state = AppState.LOADING
        self.error_reason = ""
        self._pause_requested = False  # pause asked for before READY (UT-240)
        self._fatal = False  # the hook could not be installed; ERROR is final
        self._engine_ready = False
        self._supports_vocabulary = True
        self._device: Device | None = None
        self._download_percent = -1
        self._ids = itertools.count(1)
        self._active: dict[int, _Run] = {}
        self._recording: _Run | None = None
        self._mic_error: MicError | None = None
        self._mic_open = False
        self._mic_info: OpenInfo | None = None
        self._fallback_shown = False
        self._deferred = _Deferred()
        self._shut_down = False
        self._state_file = self.paths.state

        self._ticker = QTimer(self)
        self._ticker.setInterval(TICK_MS)
        self._ticker.timeout.connect(self.tick)

        self.bridge.effects.connect(self.handle_effects)
        self.bridge.hook_event.connect(self.on_hook_event)
        self.bridge.worker_event.connect(self.on_worker_event)
        self.bridge.delivery_event.connect(self.on_delivery_event)
        self.bridge.level.connect(self.on_level)
        self.bridge.device_lost.connect(self.on_device_lost)

        tray.setting_changed.connect(self.on_setting_changed)
        tray.pause_requested.connect(self._on_pause_requested)
        tray.autostart_requested.connect(self.set_autostart)
        tray.history_insert_requested.connect(self.insert_text)
        tray.history_clear_requested.connect(self.clear_history)
        tray.open_config_requested.connect(lambda: self._open(self.config.path))
        tray.open_config_folder_requested.connect(lambda: self._open(self.paths.home))
        tray.reload_config_requested.connect(self.reload_config)
        tray.export_requested.connect(self.export_config)
        tray.import_requested.connect(self.import_config)
        tray.retry_download_requested.connect(self.retry_download)
        tray.quit_requested.connect(self._quit)

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the workers and the first model load. The app starts LOADING, never PAUSED."""
        settings = self.config.settings
        log_event(logger, "app_start", model=settings.asr.model, device=settings.asr.device)
        self._set_state(AppState.LOADING, reason="startup")
        report = self.config.last_report
        if not report.ok:
            self._notice(Msg.CONFIG_RELOADED_WITH_ERRORS, report.notice(), error=True)
        if self.history.load_warning is not None:
            self._notice(self.history.load_warning, error=True)
        try:
            self.hook.start()
        except OSError as exc:
            log_event(logger, "hook_failed", level=logging.ERROR, exc=str(exc)[:200])
            self._fatal = True
            self._set_state(AppState.ERROR, reason=Msg.HOOK_BLOCKED.text)
            self._notice(Msg.HOOK_BLOCKED, error=True)
        self.delivery.start()
        self.asr.start(settings.asr.model, settings.asr.device)
        if settings.llm.enabled:
            self._prewarm(settings)

    def shutdown(self) -> None:
        """Stop everything; releases a replayed Ctrl first."""
        if self._shut_down:
            return
        self._shut_down = True
        self._ticker.stop()
        with contextlib.suppress(Exception):
            self.hook.release_stuck_ctrl()
        for step in (self.hook.stop, self.recorder.discard, self.delivery.stop, self.asr.stop):
            try:
                step()
            except Exception as exc:
                log_event(logger, "shutdown_error", level=logging.WARNING, exc=repr(exc)[:200])
        with contextlib.suppress(Exception):
            self.tray.hide()
        log_event(logger, "app_stop")

    # -- app state -----------------------------------------------------------------------

    def _set_state(self, state: AppState, *, reason: str = "") -> None:
        previous = self.state
        self.state = state
        self.error_reason = reason if state is AppState.ERROR else ""
        if previous is not state or state is AppState.ERROR:
            transition: dict[str, Any] = {"from": previous.name, "to": state.name}
            log_event(logger, "app_state", **transition, reason=reason)
        percent = max(self._download_percent, 0)
        self.tray.set_state(TrayState(state.value), reason=reason, percent=percent)

    def _ready_state(self) -> AppState:
        return AppState.PAUSED if self._pause_requested else AppState.READY

    # -- trigger effects -----------------------------------------------------------------

    def handle_effects(self, effects: Sequence[Effect]) -> None:
        for effect in effects:
            self.handle_effect(effect)

    def handle_effect(self, effect: Effect) -> None:
        if effect is Effect.OPEN_MIC:
            self._open_mic()
        elif effect is Effect.DISCARD_MIC:
            if self._recording is None:
                self._close_mic()
        elif effect is Effect.COMMIT_HOLD:
            self._commit(DictationMode.HOLD)
        elif effect is Effect.START_HANDSFREE:
            self._commit(DictationMode.HANDSFREE)
        elif effect is Effect.FINISH:
            if self._recording is not None:
                self._finish(self._recording, reason="release")
        elif effect is Effect.CANCEL:
            self._cancel_latest()
        elif effect is Effect.ABORT_SILENT:
            if self._recording is not None:
                self._stop_recording(self._recording, DictationState.CANCELLED, notice=None)
            else:
                self._close_mic()

    def _open_mic(self) -> None:
        self._mic_error = None
        if self.state is not AppState.READY or self._recording is not None or self._mic_open:
            return
        try:
            self._mic_info = self.recorder.open()
            self._mic_open = True
        except MicError as exc:
            self._mic_error = exc

    def _close_mic(self) -> None:
        self._mic_error = None
        self._mic_open = False
        self.recorder.discard()

    def _refuse(self, msg: Msg, text: str | None = None) -> None:
        """The trigger fired but no dictation may start: tell the user, keep the key sane."""
        self.hook.abort()
        self._close_mic()
        self._notice(msg, text, overlay=True)

    def _commit(self, mode: DictationMode) -> None:
        if self._recording is not None:
            return
        if self.state is not AppState.READY:
            if self.state is AppState.DOWNLOADING:
                self._refuse(Msg.MODEL_DOWNLOADING)
            elif self.state is AppState.ERROR:
                self._refuse(Msg.APP_ERROR, Msg.APP_ERROR.format(reason=self.error_reason))
            elif self.state is AppState.LOADING:
                self._refuse(Msg.MODEL_LOADING)
            else:  # PAUSED: the trigger is disabled, so this is a race with pause()
                self.hook.abort()
                self._close_mic()
            return
        if not self._mic_open and self._mic_error is None:
            self._open_mic()
        if self._mic_error is not None:
            kind = self._mic_error.kind
            log_event(logger, "dictation_refused", reason="mic_error", kind=kind)
            self._refuse(_MIC_MESSAGES[kind])
            return
        settings = self.config.settings  # the immutable per-dictation snapshot
        now = self._clock.now_ms()
        d = Dictation(next(self._ids), mode, settings, now)
        device = self._device.kind if self._device is not None else "cpu"
        run = _Run(
            dictation=d,
            preview=resolve_preview(settings.overlay.preview, device),
            deadline_ms=now + settings.recording.max_seconds * 1000,
            next_preview_ms=now + settings.overlay.preview_interval_ms,
        )
        self._active[d.id] = run
        self._recording = run
        self._mic_open = False
        self.sounds.play_start()
        self.overlay.show_recording(d.id, handsfree=mode is DictationMode.HANDSFREE)
        log_event(logger, "dictation_committed", id=d.id, mode=mode.value, preview=run.preview)
        if self._mic_info is not None and self._mic_info.pinned_missing:
            self._notice(
                Msg.MIC_PINNED_MISSING,
                Msg.MIC_PINNED_MISSING.format(device=settings.recording.microphone),
            )
        self._mic_info = None
        self._ticker.start()

    def _transition(self, d: Dictation, new: DictationState) -> None:
        if new not in TRANSITIONS[d.state]:
            raise AssertionError(f"illegal dictation transition {d.state} -> {new}")
        d.state = new

    def _finish(self, run: _Run, *, reason: str) -> None:
        """Stop recording and submit the captured audio (release, tap, limit or mic loss)."""
        d = run.dictation
        self._transition(d, DictationState.TRANSCRIBING)
        self._recording = None
        self._ticker.stop()
        audio = self.recorder.stop()
        now = self._clock.now_ms()
        run.finished_ms = now
        run.record_ms = now - d.committed_at_ms
        log_event(
            logger,
            "dictation_stopped",
            id=d.id,
            reason=reason,
            record_ms=run.record_ms,
            samples=audio.size,
        )
        self.sounds.play_end()
        self.overlay.show_transcribing()
        self.asr.submit_final(d, audio)

    def _stop_recording(self, run: _Run, state: DictationState, *, notice: Msg | None) -> None:
        """End a recording without transcribing it (cancel, pause, chord, lock)."""
        d = run.dictation
        self._transition(d, state)
        self._recording = None
        self._ticker.stop()
        self.recorder.discard()
        self.asr.drop_previews(d.id)
        if notice is not None:
            self.sounds.play_end()
            self.overlay.show_outcome(notice)
        else:
            self.overlay.hide_now()
        self._settle(run, status=state.value)

    def _cancel_latest(self) -> None:
        """Esc: cancel the recording, or else the newest dictation not yet inserting."""
        if self._recording is not None:
            self._stop_recording(self._recording, DictationState.CANCELLED, notice=Msg.CANCELLED)
            return
        pending = [
            r
            for r in self._active.values()
            if r.dictation.state in (DictationState.TRANSCRIBING, DictationState.POSTPROCESSING)
        ]
        if not pending:
            return
        run = max(pending, key=lambda r: r.dictation.id)
        d = run.dictation
        if d.state is DictationState.POSTPROCESSING:
            self.delivery.cancel(d.id)
        self._transition(d, DictationState.CANCELLED)
        self.sounds.play_end()
        self.overlay.show_outcome(Msg.CANCELLED)
        self._settle(run, status="cancelled")

    # -- schedule: preview and limit -----------------------------------------------------

    def tick(self) -> None:
        """Runs every `TICK_MS` while recording: previews, countdown and the limit."""
        run = self._recording
        if run is None:
            self._ticker.stop()
            return
        now = self._clock.now_ms()
        d = run.dictation
        remaining_s = (run.deadline_ms - now) / 1000
        if remaining_s <= 0:
            log_event(
                logger, "recording_limit", id=d.id, max_seconds=d.settings.recording.max_seconds
            )
            self.hook.abort()
            self._finish(run, reason="limit")
            return
        if remaining_s <= COUNTDOWN_FROM_S:
            self.overlay.set_remaining(math.floor(remaining_s))
        if run.preview and now >= run.next_preview_ms:
            interval = d.settings.overlay.preview_interval_ms
            run.next_preview_ms += interval
            if run.next_preview_ms <= now:  # a stalled event loop: do not burst
                run.next_preview_ms = now + interval
            audio = self.recorder.snapshot(float(d.settings.overlay.preview_window_seconds))
            self.asr.submit_preview(d, audio)

    # -- audio thread events -------------------------------------------------------------

    def on_level(self, level: Level) -> None:
        run = self._recording
        if run is None:
            return
        run.peak_dbfs = max(run.peak_dbfs, level.peak_dbfs)
        db = 20 * math.log10(max(level.rms, 1e-6))
        self.overlay.set_level(min(max((db + 60) / 50, 0.0), 1.0))

    def on_device_lost(self) -> None:
        run = self._recording
        if run is None:
            return
        run.mic_lost = True
        log_event(logger, "mic_lost", id=run.dictation.id)
        self.hook.abort()
        self._finish(run, reason="mic_lost")

    # -- hook events ---------------------------------------------------------------------

    def on_hook_event(self, event: HookEvent) -> None:
        if isinstance(event, RepasteRequested):
            self.repaste()
        elif isinstance(event, SessionInterrupted):
            self._interrupt(event.reason)
        elif isinstance(event, HotkeyConflict):
            self._notice(
                Msg.HOTKEY_CONFLICT, Msg.HOTKEY_CONFLICT.format(hotkey=event.hotkey), error=True
            )
        elif isinstance(event, HookReinstalled):
            log_event(logger, "hook_recovered", gap_ms=event.gap_ms)

    def _interrupt(self, reason: str) -> None:
        """Session lock or suspend: the recording is dropped, marked INTERRUPTED."""
        run = self._recording
        if run is not None:
            log_event(logger, "dictation_interrupted", id=run.dictation.id, reason=reason)
            self._stop_recording(run, DictationState.INTERRUPTED, notice=Msg.INTERRUPTED)
        else:
            self._close_mic()
        self.hook.reset()

    # -- ASR worker events ---------------------------------------------------------------

    def on_worker_event(self, event: object) -> None:
        if isinstance(event, AsrOutcome):
            self._on_asr_outcome(event)
        elif isinstance(event, PreviewResult):
            self._on_preview(event)
        elif isinstance(event, EngineLoading):
            if not self._engine_ready and not self._fatal:
                self._download_percent = -1
                self._set_state(AppState.LOADING, reason=event.model_id)
        elif isinstance(event, DownloadProgress):
            self._on_download(event)
        elif isinstance(event, EngineReady):
            self._on_engine_ready(event)
        elif isinstance(event, EngineFailed):
            self._on_engine_failed(event)
        elif isinstance(event, ReloadFailed):
            self._notice(event.msg, event.text, error=True)
        elif isinstance(event, GpuReset):
            self._notice(Msg.GPU_RESET)
        elif isinstance(event, VocabTruncated):
            self._notice(Msg.VOCAB_TRUNCATED, Msg.VOCAB_TRUNCATED.format(dropped=event.dropped))

    def _on_download(self, event: DownloadProgress) -> None:
        if self._engine_ready or self._fatal:
            return  # a model switch downloads in the background while the old model serves
        percent = int(event.downloaded * 100 / event.total) if event.total else 0
        if percent == self._download_percent:
            return
        self._download_percent = percent  # ModelStore logs `model_download` itself
        if event.total and event.downloaded >= event.total:
            self._set_state(AppState.LOADING, reason=event.model_id)
        else:
            self._set_state(AppState.DOWNLOADING, reason=event.model_id)

    def _on_engine_ready(self, event: EngineReady) -> None:
        first = not self._engine_ready
        self._engine_ready = True
        self._supports_vocabulary = event.supports_vocabulary
        self._device = event.device
        self._download_percent = -1
        self.tray.set_retry_visible(False)
        self.tray.set_status(event.model_id, event.device)
        settings = self.config.settings
        reason = event.device.reason
        if reason == "insufficient_vram":
            self._notice(
                Msg.DEVICE_FALLBACK_CPU, Msg.DEVICE_FALLBACK_CPU.format(model=event.model_id)
            )
        elif reason == "forced_unavailable" or (
            reason == "parakeet_cpu_only" and settings.asr.device == "cuda"
        ):
            self._notice(
                Msg.DEVICE_FORCED_FALLBACK, Msg.DEVICE_FORCED_FALLBACK.format(model=event.model_id)
            )
        if not event.supports_vocabulary and settings.asr.vocabulary:
            self._notice(Msg.VOCAB_IGNORED, Msg.VOCAB_IGNORED.format(model=event.model_id))
        if self._fatal:
            return
        if first or self.state in (AppState.LOADING, AppState.DOWNLOADING, AppState.ERROR):
            self._set_state(self._ready_state(), reason=event.model_id)
            self._maybe_onboard()

    def _on_engine_failed(self, event: EngineFailed) -> None:
        if self._engine_ready:
            self._notice(event.msg, event.text, error=True)
            return
        if event.msg is Msg.APP_ERROR:
            reason = event.text.removeprefix(Msg.APP_ERROR.text.split("{", 1)[0])
        else:
            reason = event.text
        self._download_percent = -1
        self._set_state(AppState.ERROR, reason=reason)
        self.tray.set_retry_visible(event.msg in _DOWNLOAD_ERRORS)
        self._notice(event.msg, event.text, error=True)

    def retry_download(self) -> None:
        """Tray "Tentar baixar novamente": run the model download and load once more."""
        settings = self.config.settings
        log_event(logger, "download_retry", model=settings.asr.model)
        self.tray.set_retry_visible(False)
        if not self._engine_ready and not self._fatal:
            self._set_state(AppState.LOADING, reason="retry")
        self.asr.reload(settings.asr.model, settings.asr.device, force=True)

    def _on_preview(self, event: PreviewResult) -> None:
        run = self._recording
        if run is None or run.dictation.id != event.dictation_id or event.seq <= run.last_seq:
            return  # stale: another dictation, finished, or out of order
        run.last_seq = event.seq
        self.overlay.show_preview(event.dictation_id, event.seq, event.text)

    def _on_asr_outcome(self, outcome: AsrOutcome) -> None:
        run = self._active.get(outcome.dictation_id)
        if run is None or run.dictation.state is not DictationState.TRANSCRIBING:
            return  # cancelled or interrupted meanwhile (UT-212)
        d = run.dictation
        run.speech_s = outcome.speech_s
        run.asr_ms = outcome.asr_ms
        if outcome.kind == "text":
            if outcome.used_cpu_fallback and not self._fallback_shown:
                self._fallback_shown = True
                self._notice(Msg.GPU_FALLBACK_CPU)
            self._transition(d, DictationState.POSTPROCESSING)
            self.delivery.submit(d, outcome)
            return
        self._transition(d, DictationState.DISCARDED)
        if outcome.kind == "no_speech":
            muted = run.peak_dbfs < MUTED_PEAK_DBFS
            msg = Msg.NOTHING_DETECTED_MUTED_HINT if muted else Msg.NOTHING_DETECTED
            status = "no_speech"
        else:
            msg, status = Msg.ASR_ERROR, "error"
        self._show_outcome(run, msg)
        self._settle(run, status=status)

    # -- delivery events -----------------------------------------------------------------

    def on_delivery_event(self, event: object) -> None:
        if isinstance(event, DeliveryOutcome):
            self._on_delivered(event)
        elif isinstance(event, InjectStarted):
            run = self._active.get(event.dictation_id)
            if run is not None and run.dictation.state is DictationState.POSTPROCESSING:
                self._transition(run.dictation, DictationState.INSERTING)
        elif isinstance(event, Inserting):
            run = self._active.get(event.dictation_id)
            if run is not None and self._owns_overlay(run):
                self.overlay.show_inserting()
        elif isinstance(event, DirectInjectDone):
            self._on_direct_inject(event)

    def _on_delivered(self, outcome: DeliveryOutcome) -> None:
        run = self._active.get(outcome.dictation_id)
        if run is None:
            return
        d = run.dictation
        if d.state in _TERMINAL:
            self._settle(run, status=d.state.value)
            return
        run_status = outcome.status
        if run_status is DeliveryStatus.INSERTED:
            if d.state is DictationState.POSTPROCESSING:
                self._transition(d, DictationState.INSERTING)
            self._transition(d, DictationState.INSERTED)
            if outcome.message is not None:
                msg = outcome.message
            elif outcome.rewrite_skipped:
                msg = _REWRITE_MESSAGES.get(outcome.rewrite_skipped, Msg.REWRITE_SKIPPED)
            elif run.mic_lost:
                msg = Msg.MIC_LOST
            else:
                msg = Msg.INSERTED
            self._show_outcome(run, msg)
        elif run_status is DeliveryStatus.FAILED_INSERT:
            self._transition(d, DictationState.FAILED_INSERT)
            self._show_outcome(run, outcome.message or Msg.INSERT_FAILED_CLIPBOARD, error=True)
            if outcome.rewrite_skipped:
                self._notice(_REWRITE_MESSAGES.get(outcome.rewrite_skipped, Msg.REWRITE_SKIPPED))
        elif run_status is DeliveryStatus.NO_SPEECH:
            self._transition(d, DictationState.DISCARDED)
            self._show_outcome(run, Msg.NOTHING_DETECTED)
        else:  # CANCELLED
            self._transition(d, DictationState.CANCELLED)
        self._settle(run, status=str(run_status), outcome=outcome)

    def _owns_overlay(self, run: _Run) -> bool:
        """A dictation may use the overlay unless a newer one is recording."""
        return self._recording is None or self._recording is run

    def _show_outcome(self, run: _Run, msg: Msg, *, error: bool = False) -> None:
        if self._owns_overlay(run):
            self.overlay.show_outcome(msg)
        elif error:  # never hide a failure behind the newer recording's overlay
            self._notice(msg, error=True)

    def _settle(self, run: _Run, *, status: str, outcome: DeliveryOutcome | None = None) -> None:
        """A dictation reached a terminal state: log it, forget it, run deferred changes."""
        d = run.dictation
        if self._active.pop(d.id, None) is None:
            return
        now = self._clock.now_ms()
        total_ms = now - run.finished_ms if run.finished_ms is not None else 0
        fields: dict[str, Any] = {
            "id": d.id,
            "mode": d.mode.value,
            "record_ms": run.record_ms,
            "speech_s": round(run.speech_s, 2),
            "asr_ms": run.asr_ms,
        }
        if outcome is not None:
            fields |= {
                "post_ms": outcome.post_ms,
                "llm_ms": outcome.llm_ms,
                "inject_ms": outcome.inject_ms,
            }
            if outcome.rewrite_skipped:
                fields["rewrite_skipped"] = outcome.rewrite_skipped
        fields |= {"total_ms": total_ms, "status": status}
        if outcome is not None:
            fields["text"] = outcome.text
        slow = (
            total_ms > SLOW_TOTAL_MS
            and self._device is not None
            and self._device.kind == "cuda"
            and status == "inserted"
        )
        log_event(
            logger, "dictation_finished", level=logging.WARNING if slow else logging.INFO, **fields
        )
        if not self._active:
            self._run_deferred()

    # -- re-paste and history ------------------------------------------------------------

    def repaste(self) -> None:
        """The re-paste hotkey: insert the newest history entry (works while PAUSED)."""
        if self._recording is not None:
            log_event(logger, "repaste_ignored", reason="recording")
            return
        entry = self.history.last()
        if entry is None:
            self._notice(Msg.NOTHING_TO_REPASTE, overlay=True)
            return
        self._inject_direct(entry.text, "repaste")

    def insert_text(self, text: str) -> None:
        """Tray history "Inserir"."""
        if self._recording is not None:
            log_event(logger, "history_insert_ignored", reason="recording")
            return
        self._inject_direct(text, "history")

    def _inject_direct(self, text: str, source: str) -> None:
        log_event(logger, "inject_direct", source=source, text=text)
        bridge = self.bridge

        def done(result: InjectResult) -> None:
            bridge.delivery_event.emit(DirectInjectDone(source, result))

        self.delivery.inject_text(text, self.config.settings, done)

    def _on_direct_inject(self, event: DirectInjectDone) -> None:
        result = event.result
        log_event(logger, "inject_direct_result", source=event.source, status=str(result.status))
        msg = message_for(result)
        if result.status is not InjectStatus.INSERTED and msg is not None:
            self._notice(msg, overlay=self._recording is None, error=True)

    def clear_history(self) -> None:
        self.history.clear()
        log_event(logger, "history_cleared")

    # -- pause ---------------------------------------------------------------------------

    def _on_pause_requested(self, paused: bool) -> None:
        if paused:
            self.pause()
        else:
            self.resume()

    def pause(self) -> None:
        """The trigger becomes a normal key; a recording is cancelled; the model stays loaded."""
        if self.state is AppState.PAUSED or self._pause_requested:
            return
        if self._recording is not None:
            self._stop_recording(self._recording, DictationState.CANCELLED, notice=Msg.CANCELLED)
        self.hook.set_enabled(False)
        if self.state is AppState.READY:
            self._set_state(AppState.PAUSED, reason="user")
        else:
            self._pause_requested = True  # applied when the model becomes ready (UT-240)
            log_event(logger, "pause_deferred", state=self.state.name)

    def resume(self) -> None:
        if self.state is AppState.PAUSED:
            self._pause_requested = False
            self.hook.set_enabled(True)
            self._set_state(AppState.READY, reason="user")
        elif self._pause_requested:
            self._pause_requested = False
            self.hook.set_enabled(True)

    # -- settings ------------------------------------------------------------------------

    def guard_toggle(self, key: str, value: Any) -> bool:
        """Tray toggle veto: the AI rewrite only turns on with a local endpoint (UT-186)."""
        if key == KEY_REWRITE and value:
            rejected = any(e.key == "llm.endpoint" for e in self.config.last_report.errors)
            if rejected or not is_loopback_endpoint(self.config.settings.llm.endpoint):
                self._notice(Msg.LLM_ENDPOINT_NOT_LOCAL, error=True)
                return False
        return True

    def on_setting_changed(self, key: str, value: Any, saved: bool) -> None:
        """A tray toggle was applied; it takes effect from the next dictation (UT-172)."""
        log_event(logger, "setting_changed", key=key, value=value, saved=saved)
        if not saved:
            self._notice(Msg.CONFIG_SAVE_FAILED, error=True)
        settings = self.config.settings
        if key == KEY_REWRITE and value:
            self._prewarm(settings)
        elif key == KEY_MICROPHONE:
            self.recorder.microphone = settings.recording.microphone

    def _prewarm(self, settings: Settings) -> None:
        try:
            self._rewriter_factory(settings.llm).prewarm()
        except Exception as exc:  # an invalid endpoint never breaks the app
            log_event(logger, "prewarm_failed", exc=type(exc).__name__)

    def _busy(self) -> bool:
        return bool(self._active)

    def reload_config(self) -> None:
        """Tray "Recarregar config"; deferred while a dictation is active (UT-171)."""
        if self._busy():
            self._defer("reload", None)
            return
        before = self.config.settings
        report = self.config.reload()
        self._log_reload(report)
        self._notice(
            Msg.CONFIG_RELOADED if report.ok else Msg.CONFIG_RELOADED_WITH_ERRORS,
            report.notice(),
            error=not report.ok,
        )
        self._apply(before, self.config.settings)

    def import_config(self, path: Path) -> None:
        if self._busy():
            self._defer("import", path)
            return
        before = self.config.settings
        result = self.config.import_config(Path(path))
        log_event(
            logger, "config_import", result="applied" if result.applied else result.reason or ""
        )
        if not result.applied:
            self._notice(Msg.IMPORT_REJECTED, result.message, error=True)
            return
        self._notice(Msg.CONFIG_IMPORTED, result.message)
        if result.report is not None:
            self._log_reload(result.report)
            if not result.report.ok:
                self._notice(Msg.CONFIG_RELOADED_WITH_ERRORS, result.report.notice(), error=True)
        self._apply(before, self.config.settings)

    def export_config(self, path: Path) -> None:
        result = self.config.export_config(Path(path))
        log_event(logger, "config_export", result="ok" if result.saved else result.reason or "")
        if result.saved:
            self._notice(Msg.CONFIG_EXPORTED, Msg.CONFIG_EXPORTED.format(path=path))
        else:
            self._notice(Msg.EXPORT_FAILED, Msg.EXPORT_FAILED.format(path=path), error=True)

    def _defer(self, op: str, path: Path | None) -> None:
        log_event(logger, "config_deferred", op=op, active=len(self._active))
        self._deferred.ops.append((op, path))

    def _run_deferred(self) -> None:
        ops, self._deferred.ops = self._deferred.ops, []
        for op, path in ops:
            if op == "reload":
                self.reload_config()
            elif path is not None:
                self.import_config(path)

    def _log_reload(self, report: ValidationReport) -> None:
        result = "rejected" if report.fatal else "ok" if report.ok else "partial"
        log_event(
            logger,
            "config_reload",
            result=result,
            errors=len(report.errors),
            warnings=len(report.warnings),
        )

    def _apply(self, before: Settings, after: Settings) -> None:
        """Apply what a reload or import changed beyond the per-dictation snapshot."""
        if after == before:
            return
        set_log_text(after.debug.log_text)
        self.sounds.configure(after.recording)
        self.history.set_size(after.history.size)
        self.recorder.microphone = after.recording.microphone
        if after.trigger != before.trigger or after.history != before.history:
            self.hook.update_settings(after.trigger, after.history.repaste_hotkey)
        if self._injector is not None:
            self._injector.trigger_vk = trigger_vk(after.trigger.key)
        if (after.asr.model, after.asr.device) != (before.asr.model, before.asr.device):
            log_event(logger, "model_switch", model=after.asr.model, device=after.asr.device)
            self.asr.reload(after.asr.model, after.asr.device)  # EngineReady tells the rest
        elif (
            after.asr.vocabulary
            and after.asr.vocabulary != before.asr.vocabulary
            and self._engine_ready
            and not self._supports_vocabulary
        ):
            self._notice(Msg.VOCAB_IGNORED, Msg.VOCAB_IGNORED.format(model=after.asr.model))
        if after.llm.enabled and (not before.llm.enabled or after.llm != before.llm):
            self._prewarm(after)
        self.tray.refresh()

    # -- autostart and files -------------------------------------------------------------

    def set_autostart(self, enabled: bool) -> None:
        try:
            if enabled:
                autostart.enable(self._autostart_venv)
            else:
                autostart.disable()
        except autostart.AutostartError:
            self._notice(Msg.AUTOSTART_BLOCKED, error=True)

    def _open(self, path: Path) -> None:
        try:
            self._open_path(path)
        except OSError as exc:
            log_event(logger, "open_failed", path=path, exc=type(exc).__name__)

    # -- onboarding ----------------------------------------------------------------------

    def _load_state(self) -> _State:
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _State()
        shown = raw.get("onboarding_shown") if isinstance(raw, dict) else None
        return _State(onboarding_shown=shown is True)

    def _maybe_onboard(self) -> None:
        """The one-time "hold Right Ctrl" notice on the first READY (UT-179)."""
        state = self._load_state()
        if state.onboarding_shown:
            return
        self._notice(Msg.ONBOARDING)
        payload = json.dumps({"onboarding_shown": True}).encode("utf-8")
        try:
            atomic_write(self._state_file, payload)
        except OSError as exc:
            log_event(logger, "state_save_failed", exc=type(exc).__name__)

    # -- notices -------------------------------------------------------------------------

    def _notice(
        self, msg: Msg, text: str | None = None, *, error: bool = False, overlay: bool = False
    ) -> None:
        """Log and show a user notice: on the overlay when the user is looking at it."""
        text = msg.text if text is None else text
        log_event(logger, "notice", level=logging.WARNING if error else logging.INFO, key=msg.name)
        if overlay:
            self.overlay.show_message(text)
        else:
            self.tray.notify(text, error=error)


def _startfile(path: Path) -> None:
    os.startfile(path)  # opens the config in the user's editor


def _quit_qt() -> None:
    instance = QApplication.instance()
    if instance is not None:
        instance.quit()


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


class _FixedStore:
    """`ModelStore` stand-in for an injected engine: nothing to download."""

    def ensure(
        self,
        model_id: str,
        progress_cb: Any = None,
        *,
        load: Callable[[Path], object] | None = None,
    ) -> Path:
        del model_id, progress_cb
        path = Path()
        if load is not None:
            load(path)
        return path

    def cancel(self) -> None:
        pass


class _Late:
    """Forwards callbacks to the `App` built after the components that call them."""

    app: App | None = None

    def guard(self, key: str, value: Any) -> bool:
        return self.app.guard_toggle(key, value) if self.app is not None else True

    def report(self, text: str) -> None:
        if self.app is not None:
            self.app.tray.notify(text)


def build_app(
    settings_home: Path,
    *,
    engine: SpeechEngine | None = None,
    injector: Injector | None = None,
    recorder: Recorder | None = None,
    show_tray: bool = True,
) -> App:
    """Build the full app inside an existing `QApplication`.

    `engine` replaces model download and loading (e.g. a `FakeEngine`); `injector` and
    `recorder` replace the real ones. Call `App.start()` afterwards.
    """
    paths = Paths(Path(settings_home)).ensure()
    config = ConfigManager(paths.home)
    settings = config.settings
    set_log_text(settings.debug.log_text)
    history = HistoryStore(paths.home, size=settings.history.size)
    bridge = Bridge()
    late = _Late()

    overlay = Overlay()
    overlay_hwnd = overlay.hwnd
    if injector is None:
        injector = Injector(
            overlay_hwnd=lambda: overlay_hwnd, trigger_vk=trigger_vk(settings.trigger.key)
        )
    delivery = DeliveryThread(
        injector=injector,
        history=history,
        on_outcome=bridge.delivery_event.emit,
        on_inject_started=bridge.post_inject_started,
        on_inserting=bridge.post_inserting,
    )
    if engine is None:
        asr = AsrWorker(on_event=bridge.worker_event.emit)
    else:
        fixed = engine
        asr = AsrWorker(
            store=_FixedStore(),
            engine_factory=lambda *_args: fixed,
            device_selector=lambda _model, _requested: Device(fixed.device, "int8", "forced"),
            on_event=bridge.worker_event.emit,
        )
    if recorder is None:
        recorder = create_recorder(settings.recording.microphone)
    recorder.on_level = bridge.level.emit
    recorder.on_device_lost = bridge.device_lost.emit
    hook = HookThread(
        trigger=settings.trigger,
        repaste_hotkey=settings.history.repaste_hotkey,
        on_effects=bridge.post_effects,
        on_event=bridge.hook_event.emit,
    )
    sounds = Sounds(settings.recording, report=late.report)

    from talktype.audio import list_input_devices

    tray = Tray(
        config,
        history=history.entries,
        devices=_safe_devices(list_input_devices),
        autostart_enabled=autostart.is_enabled,
        toggle_guard=late.guard,
    )
    app = App(
        config=config,
        history=history,
        recorder=recorder,
        asr=asr,
        delivery=delivery,
        hook=hook,
        overlay=overlay,
        tray=tray,
        sounds=sounds,
        bridge=bridge,
        injector=injector,
    )
    late.app = app
    if show_tray:
        tray.show()
    active = seams.log_active(logger)
    if active:
        log_event(logger, "seams", active=",".join(active))
    log_event(
        logger, "app_built", home=paths.home, pid=os.getpid(), main_thread=threading.get_native_id()
    )
    return app


def _safe_devices(list_devices: Callable[[], list[str]]) -> Callable[[], list[str]]:
    def devices() -> list[str]:
        try:
            return list_devices()
        except Exception as exc:
            log_event(logger, "device_list_failed", exc=type(exc).__name__)
            return []

    return devices
