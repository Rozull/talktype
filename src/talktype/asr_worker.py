"""`AsrWorker`: the single recognition thread.

- Final jobs always run before preview jobs; finals run in FIFO order.
- At most one preview per dictation is queued or running; finishing a dictation drops its
  pending previews and suppresses any preview result that arrives afterwards.
- Loads run on a separate loader thread, so a reload keeps serving finals with the old
  engine until the new one is loaded and warmed up, then the worker swaps it in.
- CUDA out-of-memory retries the dictation once on a lazily built CPU engine; a second
  consecutive CUDA failure triggers a full reload through `select_device()`.

Results leave the worker through `on_event` (default: the thread-safe `events` queue);
nothing crosses the thread boundary as an exception. Task 6 marshals events into Qt.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from talktype import vad
from talktype.asr import (
    SAMPLE_RATE,
    Audio,
    DeviceKind,
    EngineError,
    EngineOOM,
    SpeechEngine,
    create_engine,
)
from talktype.asr_device import Device, cpu, select_device
from talktype.dictation import AsrOutcome, Dictation
from talktype.logging_setup import get_logger, log_event
from talktype.models import DownloadCancelled, DownloadError, ModelStore, ProgressCallback
from talktype.strings import Msg

logger = get_logger("asr_worker")

# Consecutive CUDA failures that trigger a full reload (driver reset).
CUDA_FAILURES_BEFORE_RELOAD = 2
_FINISHED_IDS_KEPT = 256


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineLoading:
    model_id: str


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    model_id: str
    downloaded: int
    total: int


@dataclass(frozen=True, slots=True)
class EngineReady:
    model_id: str
    device: Device
    supports_vocabulary: bool
    load_ms: int = 0
    warmup_ms: int = 0


@dataclass(frozen=True, slots=True)
class EngineFailed:
    """No engine could be loaded; the app goes to ERROR (or offers a retry for downloads)."""

    model_id: str
    msg: Msg
    text: str


@dataclass(frozen=True, slots=True)
class ReloadFailed:
    """A model switch failed; the previous engine stays active."""

    model_id: str
    msg: Msg
    text: str


@dataclass(frozen=True, slots=True)
class GpuReset:
    """Repeated CUDA failures; the engine is being reloaded through `select_device()`."""

    model_id: str


@dataclass(frozen=True, slots=True)
class PreviewResult:
    dictation_id: int
    seq: int
    text: str


@dataclass(frozen=True, slots=True)
class VocabTruncated:
    dropped: int


WorkerEvent = (
    EngineLoading
    | DownloadProgress
    | EngineReady
    | EngineFailed
    | ReloadFailed
    | GpuReset
    | PreviewResult
    | VocabTruncated
    | AsrOutcome
)


# --------------------------------------------------------------------------------------
# Collaborators
# --------------------------------------------------------------------------------------


class Store(Protocol):
    def ensure(
        self,
        model_id: str,
        progress_cb: ProgressCallback | None = None,
        *,
        load: Callable[[Path], object] | None = None,
    ) -> Path: ...

    def cancel(self) -> None: ...


EngineFactory = Callable[[str, DeviceKind, str, Path | None], SpeechEngine]
DeviceSelector = Callable[[str, str], Device]
SpeechSeconds = Callable[[Audio], float]


@dataclass(slots=True)
class _Final:
    dictation: Dictation
    audio: Audio


@dataclass(slots=True)
class _Preview:
    dictation: Dictation
    audio: Audio
    seq: int


@dataclass(slots=True)
class _Loaded:
    generation: int
    model_id: str
    requested: str
    engine: SpeechEngine
    device: Device
    load_ms: int
    warmup_ms: int


class AsrWorker:
    def __init__(
        self,
        *,
        store: Store | None = None,
        engine_factory: EngineFactory = create_engine,
        device_selector: DeviceSelector = select_device,
        speech_seconds: SpeechSeconds = vad.speech_seconds,
        on_event: Callable[[WorkerEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events: queue.Queue[WorkerEvent] = queue.Queue()
        self._emit = on_event or self.events.put
        self._store: Store = store or ModelStore()
        self._factory = engine_factory
        self._select = device_selector
        self._speech_seconds = speech_seconds
        self._clock = clock

        self._cond = threading.Condition()
        self._finals: deque[_Final] = deque()
        self._previews: deque[_Preview] = deque()
        self._preview_busy: set[int] = set()  # a preview is queued or running
        self._finished: set[int] = set()
        self._seq: dict[int, int] = {}
        self._stopping = False
        self._thread: threading.Thread | None = None

        self._engine: SpeechEngine | None = None
        self._device: Device | None = None
        self._active: tuple[str, str] | None = None  # (model, requested device) in use
        self._target: tuple[str, str] | None = None  # what is loaded or being loaded
        self._generation = 0
        self._loading = False
        self._pending: _Loaded | None = None
        self._cpu_engine: SpeechEngine | None = None
        self._cuda_failures = 0
        self._vocab_dropped = 0

    # --- lifecycle --------------------------------------------------------------------

    def start(self, model_id: str | None = None, device: str = "auto") -> None:
        """Start the worker thread and, when `model_id` is given, the first load."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="talktype-asr", daemon=True)
            self._thread.start()
        if model_id is not None:
            self.reload(model_id, device)

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stopping = True
            self._generation += 1
            self._cond.notify_all()
        self._store.cancel()
        if self._thread is not None:
            self._thread.join(timeout)
        for engine in (self._engine, self._cpu_engine):
            if engine is not None:
                engine.close()
        self._engine = self._cpu_engine = None

    @property
    def engine(self) -> SpeechEngine | None:
        return self._engine

    @property
    def device(self) -> Device | None:
        return self._device

    @property
    def ready(self) -> bool:
        return self._engine is not None

    def queued(self) -> tuple[int, int]:
        """(final jobs, preview jobs) waiting in the queue."""
        with self._cond:
            return len(self._finals), len(self._previews)

    # --- loading ----------------------------------------------------------------------

    def reload(self, model_id: str, device: str = "auto", *, force: bool = False) -> bool:
        """Load `model_id` in the background; the current engine serves until it is ready.

        Returns False (and does nothing) when that model and device are already active or
        loading, unless `force`.
        """
        with self._cond:
            if self._stopping or (not force and self._target == (model_id, device)):
                return False
            cancel = self._loading
            self._target = (model_id, device)
            self._generation += 1
            generation = self._generation
            self._loading = True
        if cancel:
            self._store.cancel()
        threading.Thread(
            target=self._load,
            args=(generation, model_id, device),
            name="talktype-asr-loader",
            daemon=True,
        ).start()
        return True

    def _build(self, model_id: str, device: Device, progress: ProgressCallback) -> SpeechEngine:
        built: list[SpeechEngine] = []

        def load(path: Path) -> None:
            engine = self._factory(model_id, device.kind, device.compute_type, path)
            engine.load()
            built.append(engine)

        self._store.ensure(model_id, progress, load=load)
        return built[-1]

    def _load(self, generation: int, model_id: str, requested: str) -> None:
        self._emit(EngineLoading(model_id))

        def progress(downloaded: int, total: int) -> None:
            self._emit(DownloadProgress(model_id, downloaded, total))

        engine: SpeechEngine | None = None
        try:
            device = self._select(model_id, requested)
            started = self._clock()
            try:
                engine = self._build(model_id, device, progress)
            except EngineOOM as exc:
                if device.kind != "cuda":
                    raise
                log_event(logger, "cuda_load_failed", model=model_id, exc=str(exc)[:200])
                device = cpu("insufficient_vram", device.free_mib)
                engine = self._build(model_id, device, progress)
            load_ms = round((self._clock() - started) * 1000)
            started = self._clock()
            engine.warm_up()
            warmup_ms = round((self._clock() - started) * 1000)
        except DownloadCancelled:
            if engine is not None:
                engine.close()
            self._load_done(generation, None)
            return
        except Exception as exc:
            if engine is not None:
                engine.close()
            self._load_failed(generation, model_id, exc)
            return
        self._load_done(
            generation,
            _Loaded(generation, model_id, requested, engine, device, load_ms, warmup_ms),
        )

    def _load_done(self, generation: int, loaded: _Loaded | None) -> None:
        with self._cond:
            current = generation == self._generation and not self._stopping
            if current:
                self._loading = False
                if loaded is not None:
                    self._pending = loaded
                    self._cond.notify_all()
        if not current and loaded is not None:
            loaded.engine.close()

    def _load_failed(self, generation: int, model_id: str, exc: Exception) -> None:
        log_event(
            logger, "model_load_failed", level=logging.ERROR, model=model_id, exc=str(exc)[:300]
        )
        with self._cond:
            if generation != self._generation or self._stopping:
                return
            self._loading = False
            self._target = self._active
            has_engine = self._engine is not None or self._pending is not None
            orphans = [] if has_engine else list(self._finals)
            if not has_engine:
                self._finals.clear()
        if has_engine:
            if isinstance(exc, DownloadError):  # e.g. the cache folder is not writable
                self._emit(ReloadFailed(model_id, exc.msg, exc.text))
                return
            text = Msg.MODEL_SWITCH_FAILED.format(model=model_id)
            self._emit(ReloadFailed(model_id, Msg.MODEL_SWITCH_FAILED, text))
            return
        if isinstance(exc, DownloadError):
            self._emit(EngineFailed(model_id, exc.msg, exc.text))
        else:
            reason = str(exc)[:200] or type(exc).__name__
            self._emit(EngineFailed(model_id, Msg.APP_ERROR, Msg.APP_ERROR.format(reason=reason)))
        for job in orphans:
            self._emit(AsrOutcome(job.dictation.id, "error"))

    def _swap(self, loaded: _Loaded) -> None:
        for old in (self._engine, self._cpu_engine):
            if old is not None and old is not loaded.engine:
                old.close()
        self._engine = loaded.engine
        self._device = loaded.device
        self._active = (loaded.model_id, loaded.requested)
        self._cpu_engine = None
        self._cuda_failures = 0
        log_event(
            logger,
            "model_loaded",
            model=loaded.model_id,
            device=loaded.device.kind,
            load_ms=loaded.load_ms,
            warmup_ms=loaded.warmup_ms,
        )
        self._emit(
            EngineReady(
                loaded.model_id,
                loaded.device,
                loaded.engine.supports_vocabulary,
                loaded.load_ms,
                loaded.warmup_ms,
            )
        )

    # --- submission (any thread) ------------------------------------------------------

    def submit_final(self, dictation: Dictation, audio: Audio) -> None:
        """Queue the final transcription; drops the dictation's pending previews."""
        with self._cond:
            self._mark_finished(dictation.id)
            self._finals.append(_Final(dictation, audio))
            self._cond.notify_all()

    def submit_preview(self, dictation: Dictation, audio: Audio) -> bool:
        """Queue a preview unless one is already queued or running for this dictation."""
        with self._cond:
            did = dictation.id
            if (
                self._stopping
                or self._engine is None
                or did in self._finished
                or did in self._preview_busy
            ):
                return False
            seq = self._seq.get(did, 0) + 1
            self._seq[did] = seq
            self._preview_busy.add(did)
            self._previews.append(_Preview(dictation, audio, seq))
            self._cond.notify_all()
            return True

    def drop_previews(self, dictation_id: int) -> None:
        """Forget a dictation's previews (cancelled or discarded dictation)."""
        with self._cond:
            self._mark_finished(dictation_id)

    def _mark_finished(self, dictation_id: int) -> None:
        self._finished.add(dictation_id)
        self._seq.pop(dictation_id, None)
        self._previews = deque(p for p in self._previews if p.dictation.id != dictation_id)
        if len(self._finished) > _FINISHED_IDS_KEPT:
            self._finished.discard(min(self._finished))

    # --- worker thread ----------------------------------------------------------------

    def _next(self) -> _Loaded | _Final | _Preview | None:
        with self._cond:
            while True:
                if self._stopping:
                    return None
                if self._pending is not None:
                    loaded, self._pending = self._pending, None
                    return loaded
                if self._engine is not None:
                    if self._finals:
                        return self._finals.popleft()
                    if self._previews:
                        return self._previews.popleft()
                self._cond.wait()

    def _loop(self) -> None:
        while (job := self._next()) is not None:
            try:
                if isinstance(job, _Loaded):
                    self._swap(job)
                elif isinstance(job, _Final):
                    self._run_final(job)
                else:
                    self._run_preview(job)
            except Exception as exc:
                log_event(
                    logger,
                    "worker_error",
                    level=logging.ERROR,
                    thread="asr",
                    exc=type(exc).__name__,
                    exc_info=True,
                )
                if isinstance(job, _Final):
                    self._emit(AsrOutcome(job.dictation.id, "error"))
                elif isinstance(job, _Preview):
                    self._preview_finished(job.dictation.id)

    def _elapsed_ms(self, started: float) -> int:
        return round((self._clock() - started) * 1000)

    def _run_final(self, job: _Final) -> None:
        settings = job.dictation.settings
        did = job.dictation.id
        started = self._clock()
        speech_s: float | None
        try:
            speech_s = self._speech_seconds(job.audio)
        except Exception as exc:
            speech_s = None
            log_event(logger, "vad_error", level=logging.WARNING, id=did, exc=type(exc).__name__)
        if speech_s is not None and speech_s < settings.recording.min_speech_seconds:
            self._emit(
                AsrOutcome(did, "no_speech", speech_s=speech_s, asr_ms=self._elapsed_ms(started))
            )
            return
        engine = self._engine
        assert engine is not None
        language, vocabulary = settings.asr.language, settings.asr.vocabulary
        used_cpu = False
        try:
            text = engine.transcribe(
                job.audio, language=language, vocabulary=vocabulary, fast=False
            )
            if engine.device == "cuda":
                self._cuda_failures = 0
        except EngineError as exc:
            log_event(
                logger,
                "transcribe_failed",
                level=logging.WARNING,
                id=did,
                device=engine.device,
                oom=isinstance(exc, EngineOOM),
                exc=str(exc)[:200],
            )
            if engine.device != "cuda":
                self._emit(AsrOutcome(did, "error", speech_s=speech_s or 0.0))
                return
            self._cuda_failure()
            if not isinstance(exc, EngineOOM):
                self._emit(AsrOutcome(did, "error", speech_s=speech_s or 0.0))
                return
            try:
                text = self._cpu_fallback(engine).transcribe(
                    job.audio, language=language, vocabulary=vocabulary, fast=False
                )
            except Exception as cpu_exc:
                log_event(
                    logger, "cpu_fallback_failed", level=logging.ERROR, exc=str(cpu_exc)[:200]
                )
                self._emit(AsrOutcome(did, "error", speech_s=speech_s or 0.0))
                return
            used_cpu = True
        self._report_vocab(engine)
        self._emit(
            AsrOutcome(
                did,
                "text",
                text,
                speech_s=speech_s if speech_s is not None else 0.0,
                asr_ms=self._elapsed_ms(started),
                used_cpu_fallback=used_cpu,
            )
        )

    def _cuda_failure(self) -> None:
        self._cuda_failures += 1
        if self._cuda_failures < CUDA_FAILURES_BEFORE_RELOAD or self._active is None:
            return
        self._cuda_failures = 0
        model_id, requested = self._active
        log_event(logger, "gpu_reset_reload", level=logging.WARNING, model=model_id)
        self._emit(GpuReset(model_id))
        self.reload(model_id, requested, force=True)

    def _cpu_fallback(self, engine: SpeechEngine) -> SpeechEngine:
        if self._cpu_engine is None:
            path = self._store.ensure(engine.model_id)
            fallback = self._factory(engine.model_id, "cpu", "int8", path)
            fallback.load()
            self._cpu_engine = fallback
        return self._cpu_engine

    def _report_vocab(self, engine: SpeechEngine) -> None:
        dropped = int(getattr(engine, "last_vocab_dropped", 0) or 0)
        if dropped != self._vocab_dropped:
            self._vocab_dropped = dropped
            if dropped:
                self._emit(VocabTruncated(dropped))

    def _preview_finished(self, dictation_id: int) -> bool:
        """Release the dictation's preview slot; True when its result may still be shown."""
        with self._cond:
            self._preview_busy.discard(dictation_id)
            return dictation_id not in self._finished

    def _run_preview(self, job: _Preview) -> None:
        settings = job.dictation.settings
        window = settings.overlay.preview_window_seconds * SAMPLE_RATE
        audio = job.audio[-window:] if job.audio.size > window else job.audio
        engine = self._engine
        assert engine is not None
        started = self._clock()
        text: str | None
        try:
            text = engine.transcribe(
                np.ascontiguousarray(audio),
                language=settings.asr.language,
                vocabulary=_NO_VOCABULARY,
                fast=True,
            )
        except Exception as exc:
            text = None
            log_event(logger, "preview_error", id=job.dictation.id, exc=type(exc).__name__)
        asr_ms = self._elapsed_ms(started)
        if self._preview_finished(job.dictation.id) and text is not None:
            log_event(logger, "preview", id=job.dictation.id, seq=job.seq, asr_ms=asr_ms)
            self._emit(PreviewResult(job.dictation.id, job.seq, text))


_NO_VOCABULARY: Sequence[str] = ()
