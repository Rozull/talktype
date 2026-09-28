"""`AsrWorker` scheduling with `FakeEngine` (UT-110–UT-124)."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import numpy as np
import pytest

from talktype.asr import EngineError, EngineOOM, ModelLoadError
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
    WorkerEvent,
)
from talktype.config import AsrSettings, OverlaySettings, RecordingSettings, Settings
from talktype.dictation import AsrOutcome, Dictation, DictationMode
from talktype.models import DownloadError
from talktype.strings import Msg
from tests.fakes import FakeEngine, FakeEngineFactory, FakeStore

pytestmark = pytest.mark.unit

E = TypeVar("E")
TURBO = "large-v3-turbo"
CUDA = Device("cuda", "int8_float16", "auto")


def audio(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(round(seconds * 16000), dtype=np.float32)


def dictation(did: int, **asr: Any) -> Dictation:
    settings = Settings(
        asr=AsrSettings(**asr),
        recording=RecordingSettings(min_speech_seconds=0.3),
        overlay=OverlaySettings(preview_window_seconds=15),
    )
    return Dictation(did, DictationMode.HOLD, settings, committed_at_ms=0)


class Events:
    def __init__(self, worker: AsrWorker) -> None:
        self.worker = worker
        self.seen: list[WorkerEvent] = []

    def wait(
        self, kind: type[E], pred: Callable[[E], bool] = lambda e: True, timeout: float = 5.0
    ) -> E:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"no {kind.__name__} in {self.seen}"
            try:
                event = self.worker.events.get(timeout=remaining)
            except queue.Empty:
                continue
            self.seen.append(event)
            if isinstance(event, kind) and pred(event):
                return event

    def drain(self, seconds: float = 0.2) -> list[WorkerEvent]:
        deadline = time.monotonic() + seconds
        got: list[WorkerEvent] = []
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                got.append(self.worker.events.get(timeout=remaining))
            except queue.Empty:
                break
        self.seen.extend(got)
        return got

    def outcome(self, did: int) -> AsrOutcome:
        return self.wait(AsrOutcome, lambda o: o.dictation_id == did)


class Harness:
    def __init__(
        self,
        *,
        speech: Callable[[np.ndarray], float] = lambda a: 5.0,
        prepare: Callable[[FakeEngine], None] | None = None,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> None:
        self.factory = FakeEngineFactory(prepare)
        self.store = FakeStore()
        self.selections: list[tuple[str, str]] = []
        self.worker = AsrWorker(
            store=self.store,
            engine_factory=self.factory,
            device_selector=self.select,
            speech_seconds=speech,
            on_event=on_event,
        )
        self.events = Events(self.worker)

    def select(self, model_id: str, requested: str) -> Device:
        self.selections.append((model_id, requested))
        if model_id == "parakeet-v3" or requested == "cpu":
            return Device("cpu", "int8", "forced")
        return CUDA

    def start(self, model_id: str = TURBO, *, wait: bool = True) -> None:
        self.worker.start(model_id)
        if wait:
            self.events.wait(EngineReady)

    @property
    def engine(self) -> FakeEngine:
        engine = self.worker.engine
        assert isinstance(engine, FakeEngine)
        return engine


@pytest.fixture
def harness() -> Iterator[Harness]:
    h = Harness()
    yield h
    h.worker.stop()


def make(**kwargs: Any) -> Harness:
    return Harness(**kwargs)


@pytest.fixture
def cleanup() -> Iterator[list[Harness]]:
    made: list[Harness] = []
    yield made
    for h in made:
        h.worker.stop()


def test_ut110_final_emits_text_outcome(harness: Harness) -> None:
    harness.start()
    d1 = dictation(1)

    harness.worker.submit_final(d1, audio())

    outcome = harness.events.outcome(1)
    assert outcome == AsrOutcome(
        1, "text", "olá", speech_s=5.0, asr_ms=outcome.asr_ms, used_cpu_fallback=False
    )
    call = harness.factory.calls[-1]
    assert (call.language, call.fast) == ("pt", False)


def test_ut111_final_outranks_queued_preview(harness: Harness) -> None:
    harness.start()
    gate = threading.Event()
    harness.engine.transcribe_gate = gate
    harness.worker.submit_final(dictation(1), audio())
    assert harness.engine.transcribing.wait(5)  # the engine is busy with d1

    d2, d3 = dictation(2), dictation(3)
    assert harness.worker.submit_preview(d2, audio())
    assert harness.worker.submit_preview(d3, audio())
    harness.worker.submit_final(d2, audio())
    assert harness.worker.queued() == (1, 1)
    gate.set()

    harness.events.outcome(2)
    calls = harness.factory.calls
    assert [c.fast for c in calls[:2]] == [False, False]  # d1 final, then d2 final
    harness.events.wait(PreviewResult, lambda r: r.dictation_id == 3)
    assert calls[2].fast is True


def test_ut112_one_pending_preview_per_dictation(harness: Harness) -> None:
    harness.start()
    gate = threading.Event()
    harness.engine.transcribe_gate = gate
    harness.worker.submit_final(dictation(9), audio())
    assert harness.engine.transcribing.wait(5)
    d1 = dictation(1)

    assert harness.worker.submit_preview(d1, audio()) is True
    before = harness.worker.queued()
    assert harness.worker.submit_preview(d1, audio()) is False
    assert harness.worker.queued() == before == (0, 1)
    gate.set()


def test_ut113_finishing_drops_pending_previews(harness: Harness) -> None:
    harness.start()
    gate = threading.Event()
    harness.engine.transcribe_gate = gate
    harness.worker.submit_final(dictation(9), audio())
    assert harness.engine.transcribing.wait(5)
    d1 = dictation(1)
    assert harness.worker.submit_preview(d1, audio())

    harness.worker.submit_final(d1, audio())

    assert harness.worker.queued() == (1, 0)
    assert harness.worker.submit_preview(d1, audio()) is False
    gate.set()
    harness.events.outcome(1)
    assert not any(
        isinstance(e, PreviewResult) and e.dictation_id == 1 for e in harness.events.drain()
    )
    assert not any(isinstance(e, PreviewResult) for e in harness.events.seen)


def test_ut113_running_preview_result_is_dropped_after_finish(harness: Harness) -> None:
    harness.start()
    gate = threading.Event()
    harness.engine.transcribe_gate = gate
    d1 = dictation(1)
    assert harness.worker.submit_preview(d1, audio())
    assert harness.engine.transcribing.wait(5)  # the preview is running

    harness.worker.submit_final(d1, audio())
    gate.set()

    harness.events.outcome(1)
    harness.events.drain()
    assert not any(isinstance(e, PreviewResult) for e in harness.events.seen)


def test_ut114_no_speech_skips_transcription() -> None:
    h = make(speech=lambda a: 0.1)
    try:
        h.start()
        for did in (1, 2, 3):
            h.worker.submit_final(dictation(did), audio())
            outcome = h.events.outcome(did)
            assert outcome.kind == "no_speech"
            assert outcome.speech_s == pytest.approx(0.1)
        assert h.factory.calls == []  # warm-up is not a transcribe call on FakeEngine
    finally:
        h.worker.stop()


def test_ut115_exactly_min_speech_is_transcribed() -> None:
    h = make(speech=lambda a: 0.3)
    try:
        h.start()
        h.worker.submit_final(dictation(1), audio())
        outcome = h.events.outcome(1)
        assert outcome.kind == "text"
        assert len(h.factory.calls) == 1
    finally:
        h.worker.stop()


def test_ut116_vad_error_transcribes_anyway(logs: list[str]) -> None:
    def broken(a: np.ndarray) -> float:
        raise RuntimeError("onnxruntime failed")

    h = make(speech=broken)
    try:
        h.start()
        h.worker.submit_final(dictation(1), audio())
        outcome = h.events.outcome(1)
        assert outcome.kind == "text"
        assert len(h.factory.calls) == 1
        assert any(line.startswith("vad_error id=1") for line in logs)
    finally:
        h.worker.stop()


def test_ut117_cuda_oom_retries_once_on_cpu(harness: Harness) -> None:
    harness.start()
    cuda_engine = harness.engine
    cuda_engine.errors = [EngineOOM("CUDA failed with error out of memory")]

    harness.worker.submit_final(dictation(1), audio())

    outcome = harness.events.outcome(1)
    assert outcome.kind == "text"
    assert outcome.used_cpu_fallback is True
    cpu_engines = harness.factory.by(TURBO, "cpu")
    assert len(cpu_engines) == 1
    assert cpu_engines[0].order == ["load", "transcribe"]
    assert harness.worker.engine is cuda_engine
    assert "close" not in cuda_engine.order

    harness.worker.submit_final(dictation(2), audio())
    assert harness.events.outcome(2).used_cpu_fallback is False
    assert len(harness.factory.by(TURBO, "cpu")) == 1


def test_ut118_two_consecutive_cuda_failures_reload(harness: Harness) -> None:
    harness.start()
    first = harness.engine
    first.errors = [EngineOOM("out of memory"), EngineError("CUDA error: device reset")]

    harness.worker.submit_final(dictation(1), audio())
    assert harness.events.outcome(1).used_cpu_fallback is True
    harness.worker.submit_final(dictation(2), audio())
    assert harness.events.outcome(2).kind == "error"

    ready = harness.events.wait(EngineReady)
    assert ready.model_id == TURBO
    assert GpuReset(TURBO) in harness.events.seen
    assert harness.selections == [(TURBO, "auto"), (TURBO, "auto")]
    assert harness.worker.engine is not first
    assert "close" in first.order


def test_ut118_a_success_resets_the_failure_count(harness: Harness) -> None:
    harness.start()
    harness.engine.errors = [EngineOOM("out of memory")]
    harness.worker.submit_final(dictation(1), audio())
    harness.events.outcome(1)
    harness.worker.submit_final(dictation(2), audio())  # succeeds on CUDA
    harness.events.outcome(2)
    harness.engine.errors = [EngineOOM("out of memory")]
    harness.worker.submit_final(dictation(3), audio())
    harness.events.outcome(3)

    harness.events.drain()
    assert not any(isinstance(e, GpuReset) for e in harness.events.seen)


def test_ut119_finals_are_fifo(harness: Harness) -> None:
    harness.start()
    gate = threading.Event()
    harness.engine.transcribe_gate = gate
    harness.engine.text = lambda a: f"{a.size}"
    harness.worker.submit_final(dictation(1), audio(3.0))
    assert harness.engine.transcribing.wait(5)
    harness.worker.submit_final(dictation(2), audio(0.5))
    harness.worker.submit_final(dictation(3), audio(2.0))
    gate.set()

    harness.events.outcome(3)
    order = [e.dictation_id for e in harness.events.seen if isinstance(e, AsrOutcome)]
    assert order == [1, 2, 3]


def test_ut120_reload_keeps_serving_until_the_new_engine_is_ready() -> None:
    gate = threading.Event()

    def prepare(engine: FakeEngine) -> None:
        if engine.model_id == "small":
            engine.load_gate = gate

    h = make(prepare=prepare)
    try:
        h.start()
        turbo = h.engine
        assert h.worker.reload("small") is True

        h.worker.submit_final(dictation(1), audio())
        h.events.outcome(1)
        assert h.factory.calls[-1].engine is turbo

        gate.set()
        ready = h.events.wait(EngineReady)
        assert ready.model_id == "small"
        h.worker.submit_final(dictation(2), audio())
        h.events.outcome(2)
        assert h.factory.calls[-1].engine.model_id == "small"
        assert "close" in turbo.order
    finally:
        gate.set()
        h.worker.stop()


def test_ut120_failed_switch_keeps_the_previous_engine() -> None:
    def prepare(engine: FakeEngine) -> None:
        if engine.model_id == "small":
            engine.load_error = ModelLoadError("small: unable to open model.bin")

    h = make(prepare=prepare)
    try:
        h.start()
        turbo = h.engine
        h.worker.reload("small")

        failed = h.events.wait(ReloadFailed)
        assert failed.msg is Msg.MODEL_SWITCH_FAILED
        assert failed.text == Msg.MODEL_SWITCH_FAILED.format(model="small")
        assert h.worker.engine is turbo
        h.worker.submit_final(dictation(1), audio())
        h.events.outcome(1)
        assert h.factory.calls[-1].engine is turbo
        assert h.worker.reload("small") is True  # the failed target can be retried
    finally:
        h.worker.stop()


def test_ut121_reloading_the_active_model_is_a_no_op(harness: Harness) -> None:
    harness.start()
    created = len(harness.factory.created)

    assert harness.worker.reload(TURBO) is False

    assert len(harness.factory.created) == created
    assert harness.engine.order.count("load") == 1


def test_ut122_preview_seq_increases_per_dictation(harness: Harness) -> None:
    harness.start()
    d1, d2 = dictation(1), dictation(2)
    seqs: list[int] = []
    for _ in range(3):
        assert harness.worker.submit_preview(d1, audio())
        seqs.append(harness.events.wait(PreviewResult, lambda r: r.dictation_id == 1).seq)
    assert harness.worker.submit_preview(d2, audio())
    other = harness.events.wait(PreviewResult, lambda r: r.dictation_id == 2)

    assert seqs == [1, 2, 3]
    assert other.seq == 1


def test_preview_transcribes_only_the_last_window_fast(harness: Harness) -> None:
    harness.start()
    d1 = dictation(1, vocabulary=["GitHub"])

    assert harness.worker.submit_preview(d1, audio(300))
    harness.events.wait(PreviewResult)

    call = harness.factory.calls[-1]
    assert call.samples == 15 * 16000
    assert call.fast is True
    assert call.vocabulary == ()


def test_ut123_warm_up_runs_once_before_ready() -> None:
    snapshots: list[tuple[str, list[str]]] = []
    h: Harness

    def on_event(event: WorkerEvent) -> None:
        if isinstance(event, EngineReady):
            engine = h.worker.engine
            assert isinstance(engine, FakeEngine)
            snapshots.append((event.model_id, list(engine.order)))
        h.worker.events.put(event)

    h = make(on_event=on_event)
    try:
        h.start()
        assert snapshots == [(TURBO, ["load", "warm_up"])]
        loading = [e for e in h.events.seen if isinstance(e, EngineLoading)]
        assert loading == [EngineLoading(TURBO)]
    finally:
        h.worker.stop()


def test_ut124_final_before_first_ready_waits_for_the_engine() -> None:
    gate = threading.Event()

    def prepare(engine: FakeEngine) -> None:
        engine.load_gate = gate

    h = make(prepare=prepare)
    try:
        h.start(wait=False)
        h.worker.submit_final(dictation(1), audio())
        assert not any(isinstance(e, AsrOutcome) for e in h.events.drain(0.2))
        assert h.worker.queued() == (1, 0)

        gate.set()
        h.events.wait(EngineReady)
        assert h.events.outcome(1).kind == "text"
    finally:
        gate.set()
        h.worker.stop()


def test_first_load_failure_reports_and_fails_queued_finals() -> None:
    def prepare(engine: FakeEngine) -> None:
        engine.load_error = ModelLoadError("corrupt twice")

    h = make(prepare=prepare)
    try:
        h.worker.submit_final(dictation(1), audio())
        h.start(wait=False)
        failed = h.events.wait(EngineFailed)
        assert failed.model_id == TURBO
        assert h.events.outcome(1).kind == "error"
    finally:
        h.worker.stop()


def test_download_error_is_reported_with_its_message() -> None:
    h = make()

    def offline(*args: Any, **kwargs: Any) -> Any:
        raise DownloadError(Msg.DOWNLOAD_OFFLINE, Msg.DOWNLOAD_OFFLINE.format())

    h.store.ensure = offline  # type: ignore[method-assign]
    try:
        h.start(wait=False)
        failed = h.events.wait(EngineFailed)
        assert failed.msg is Msg.DOWNLOAD_OFFLINE
    finally:
        h.worker.stop()


def test_download_progress_is_forwarded() -> None:
    h = make()
    original = h.store.ensure

    def with_progress(model_id: str, progress_cb: Any = None, *, load: Any = None) -> Any:
        progress_cb(50, 100)
        return original(model_id, progress_cb, load=load)

    h.store.ensure = with_progress  # type: ignore[method-assign]
    try:
        h.start()
        assert DownloadProgress(TURBO, 50, 100) in h.events.seen
    finally:
        h.worker.stop()


def test_vocabulary_truncation_is_reported_once(harness: Harness) -> None:
    harness.start()
    harness.engine.last_vocab_dropped = 4

    for did in (1, 2):
        harness.worker.submit_final(dictation(did), audio())
        harness.events.outcome(did)

    truncated = [e for e in harness.events.seen if isinstance(e, VocabTruncated)]
    assert truncated == [VocabTruncated(4)]
