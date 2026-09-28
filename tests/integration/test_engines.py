"""Real engines, the hallucination guard and the model store (IT-035–IT-038, marker `gpu`).

They reuse the Hugging Face cache already populated on the target machine (turbo and
Parakeet). E2E-001 covers the full path through the WAV audio seam.
"""

from __future__ import annotations

import socket
import time
import tomllib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from talktype import win32
from talktype.asr import ParakeetEngine, WhisperEngine
from talktype.asr_device import prepare_cuda_dlls, select_device
from talktype.asr_worker import AsrWorker, EngineFailed, EngineReady
from talktype.audio import read_wav
from talktype.config import AsrSettings, Settings
from talktype.dictation import AsrOutcome, Dictation, DictationMode
from talktype.models import ModelStore
from tests.conftest import FIXTURES_AUDIO, normalize

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

GIB = 1024**3
TURBO = "large-v3-turbo"
HALLUCINATIONS = ("obrigado por assistir", "legendas", "inscreva", "amara org", "tchau")


def expected() -> dict[str, dict[str, Any]]:
    return tomllib.loads((FIXTURES_AUDIO / "expected.toml").read_text(encoding="utf-8"))


def fixture(name: str) -> np.ndarray:
    samples, rate = read_wav(FIXTURES_AUDIO / f"{name}.wav")
    assert rate == 16000
    return samples


@pytest.fixture(scope="module")
def offline_store() -> ModelStore:
    prepare_cuda_dlls()
    return ModelStore()


@pytest.fixture(scope="module")
def turbo(offline_store: ModelStore) -> Iterator[tuple[WhisperEngine, int]]:
    """Turbo on CUDA, loaded and warmed up; yields the engine and the NVML used delta."""
    before = win32.nvml_memory().used
    path = offline_store.ensure(TURBO)
    engine = WhisperEngine(TURBO, "cuda", "int8_float16", path)
    engine.load()
    engine.warm_up()
    engine.transcribe(fixture("deploy_pr"), language="pt", vocabulary=[], fast=False)
    delta = win32.nvml_memory().used - before
    yield engine, delta
    engine.close()


def test_it035_turbo_on_cuda_transcribes_the_key_terms(
    turbo: tuple[WhisperEngine, int],
) -> None:
    engine, delta = turbo
    audio = fixture("deploy_pr")

    started = time.perf_counter()
    text = engine.transcribe(audio, language="pt", vocabulary=["GitHub", "CI"], fast=False)
    elapsed = time.perf_counter() - started

    normalized = normalize(text)
    for term in expected()["deploy_pr"]["key_terms"]:
        assert term in normalized, (term, text)
    assert elapsed <= 1.0, f"warm latency {elapsed:.2f} s"
    assert delta <= 2.5 * GIB, f"VRAM delta {delta / GIB:.2f} GiB"


def test_it035_device_selection_picks_cuda_on_this_machine() -> None:
    assert select_device(TURBO, "auto").kind == "cuda"


def _dictation(did: int) -> Dictation:
    return Dictation(did, DictationMode.HOLD, Settings(asr=AsrSettings()), committed_at_ms=0)


def _wait(worker: AsrWorker, kind: type, did: int | None = None, timeout: float = 120.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = worker.events.get(timeout=max(0.1, deadline - time.monotonic()))
        if isinstance(event, EngineFailed):
            pytest.fail(f"engine failed: {event.text}")
        if isinstance(event, kind) and (did is None or getattr(event, "dictation_id", None) == did):
            return event
    pytest.fail(f"no {kind.__name__} within {timeout} s")


def test_it036_silence_is_no_speech_and_long_silences_do_not_hallucinate(
    offline_store: ModelStore,
) -> None:
    worker = AsrWorker(store=offline_store)
    worker.start(TURBO, "cuda")
    try:
        ready = _wait(worker, EngineReady)
        assert ready.device.kind == "cuda"

        worker.submit_final(_dictation(1), fixture("silencio_3s"))
        assert _wait(worker, AsrOutcome, 1).kind == "no_speech"

        gap = np.zeros(20 * 16000, dtype=np.float32)
        parts: list[np.ndarray] = []
        names = ["deploy_pr", "curta_sim", "nova_linha", "hesitacao"]
        while sum(p.size for p in parts) < 5 * 60 * 16000:
            parts.extend([fixture(names[len(parts) // 2 % len(names)]), gap])
        recording = np.concatenate(parts)[: 5 * 60 * 16000]

        worker.submit_final(_dictation(2), recording)
        outcome = _wait(worker, AsrOutcome, 2, timeout=300)
    finally:
        worker.stop()

    assert outcome.kind == "text"
    text = normalize(outcome.text)
    for phrase in HALLUCINATIONS:
        assert phrase not in text, (phrase, outcome.text)
    allowed = set(normalize(" ".join(str(t["text"]) for t in expected().values())).split())
    extra = [w for w in text.split() if w not in allowed]
    assert len(extra) <= 0.1 * len(text.split()), (extra, outcome.text)


def test_it037_parakeet_on_cpu(offline_store: ModelStore) -> None:
    path = offline_store.ensure("parakeet-v3")
    engine = ParakeetEngine("parakeet-v3", path)
    engine.load()
    try:
        engine.warm_up()
        text = engine.transcribe(fixture("deploy_pr"), language="pt", vocabulary=[], fast=False)
    finally:
        engine.close()

    normalized = normalize(text)
    assert "deploy" in normalized, text
    assert "pull request" in normalized, text


def test_it037_parakeet_long_audio_goes_through_vad(offline_store: ModelStore) -> None:
    path = offline_store.ensure("parakeet-v3")
    engine = ParakeetEngine("parakeet-v3", path)
    engine.load()
    gap = np.zeros(3 * 16000, dtype=np.float32)
    try:
        text = engine.transcribe(
            np.concatenate([fixture("deploy_pr"), gap, fixture("nova_linha")]),
            language="pt",
            vocabulary=[],
            fast=False,
        )
    finally:
        engine.close()

    normalized = normalize(text)
    assert "deploy" in normalized, text
    assert "segundo item" in normalized, text


def test_it038_cached_model_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ModelStore.ensure opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    progress: list[tuple[int, int]] = []

    started = time.perf_counter()
    path = ModelStore().ensure(TURBO, lambda d, t: progress.append((d, t)))
    elapsed = time.perf_counter() - started

    assert elapsed <= 2.0
    assert (path / "model.bin").is_file()
    assert progress == []
