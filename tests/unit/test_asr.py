"""Engine parameters, vocabulary budget, Parakeet behavior and OOM mapping (UT-101–UT-104)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from talktype.asr import (
    VOCAB_PROMPT_BUDGET,
    EngineError,
    EngineOOM,
    ModelLoadError,
    ParakeetEngine,
    WhisperEngine,
    vocabulary_prompt,
)

pytestmark = pytest.mark.unit

AUDIO = np.zeros(16000, dtype=np.float32)


@dataclass
class _Segment:
    text: str


class FakeWhisperModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init = (args, kwargs)
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterator[_Segment], object]:
        self.calls.append(kwargs)

        def segments() -> Iterator[_Segment]:
            if self.error is not None:
                raise self.error  # CTranslate2 fails while the segments are decoded
            yield _Segment(" faz o deploy")
            yield _Segment(" e abre o PR.")

        return segments(), object()


def whisper(**kwargs: Any) -> tuple[WhisperEngine, FakeWhisperModel]:
    models: list[FakeWhisperModel] = []

    def factory(*args: Any, **kw: Any) -> FakeWhisperModel:
        models.append(FakeWhisperModel(*args, **kw))
        return models[-1]

    engine = WhisperEngine("large-v3-turbo", "cuda", model_factory=factory, **kwargs)
    engine.load()
    return engine, models[0]


def test_ut101_whisper_parameters() -> None:
    engine, model = whisper()

    text = engine.transcribe(AUDIO, language="pt", vocabulary=["PR", "Kubernetes"], fast=False)

    assert text == "faz o deploy e abre o PR."
    call = model.calls[-1]
    assert call["language"] == "pt"
    assert call["task"] == "transcribe"
    assert call["initial_prompt"] == "PR, Kubernetes"
    assert call["beam_size"] == 5
    assert call["vad_filter"] is True

    engine.transcribe(AUDIO, language="pt", vocabulary=["PR"], fast=True)
    assert model.calls[-1]["beam_size"] == 1
    assert model.calls[-1]["vad_filter"] is False

    engine.transcribe(AUDIO, language="pt", vocabulary=[], fast=False)
    assert model.calls[-1]["initial_prompt"] is None

    engine.transcribe(AUDIO, language="en", vocabulary=[], fast=False)
    assert model.calls[-1]["language"] == "en"
    assert model.calls[-1]["task"] == "transcribe"


def test_ut101_whisper_load_uses_device_and_compute_type() -> None:
    _engine, model = whisper()

    args, kwargs = model.init
    assert args == ("large-v3-turbo",)
    assert kwargs["device"] == "cuda"
    assert kwargs["compute_type"] == "int8_float16"
    assert kwargs["local_files_only"] is True


def test_ut101_warm_up_transcribes_silence_without_vad() -> None:
    engine, model = whisper()

    engine.warm_up()

    assert model.calls[-1]["vad_filter"] is False
    assert model.calls[-1]["initial_prompt"] is None


def test_ut102_vocabulary_is_truncated_to_the_prompt_budget(logs: list[str]) -> None:
    engine, model = whisper()
    vocabulary = [f"termo{i:05d}" for i in range(300)]  # 300 entries of 10 characters

    engine.transcribe(AUDIO, language="pt", vocabulary=vocabulary, fast=False)

    prompt = model.calls[-1]["initial_prompt"]
    assert len(prompt) <= VOCAB_PROMPT_BUDGET
    dropped = engine.last_vocab_dropped
    assert dropped > 0
    assert prompt.split(", ") == vocabulary[: 300 - dropped]
    assert f"vocab_truncated dropped={dropped}" in logs


def test_ut102_prompt_budget_boundary() -> None:
    fits = ["a" * 398, "b" * 400]  # 398 + 2 + 400 == 800

    assert vocabulary_prompt(fits) == ("a" * 398 + ", " + "b" * 400, 0)
    assert vocabulary_prompt([*fits, "c"]) == ("a" * 398 + ", " + "b" * 400, 1)


class FakeOnnxModel:
    def __init__(self) -> None:
        self.recognized: list[int] = []
        self.with_vad_calls: list[tuple[Any, dict[str, Any]]] = []

    def recognize(self, audio: Any, **kwargs: Any) -> str:
        self.recognized.append(len(audio))
        return " olá mundo "

    def with_vad(self, vad: Any, **kwargs: Any) -> Any:
        self.with_vad_calls.append((vad, kwargs))
        model = self

        class _Adapter:
            def recognize(self, audio: Any, **kw: Any) -> Iterator[_Segment]:
                model.recognized.append(len(audio))
                return iter([_Segment(" primeira parte"), _Segment(" segunda parte ")])

        return _Adapter()


def test_ut103_parakeet_ignores_vocabulary_and_uses_vad_for_long_audio() -> None:
    model = FakeOnnxModel()
    loads: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    vad = object()

    def loader(*args: Any, **kwargs: Any) -> FakeOnnxModel:
        loads.append((args, kwargs))
        return model

    engine = ParakeetEngine(loader=loader, vad_factory=lambda: vad)
    engine.load()

    assert engine.supports_vocabulary is False
    assert engine.device == "cpu"
    assert loads[0][0][0] == "nemo-parakeet-tdt-0.6b-v3"
    assert loads[0][1]["quantization"] == "int8"
    assert loads[0][1]["providers"] == ["CPUExecutionProvider"]

    short = engine.transcribe(AUDIO, language="pt", vocabulary=["X"], fast=False)
    assert short == "olá mundo"
    assert model.with_vad_calls == []

    long_audio = np.zeros(16000 * 21, dtype=np.float32)
    text = engine.transcribe(long_audio, language="pt", vocabulary=["X"], fast=False)
    assert text == "primeira parte segunda parte"
    assert model.with_vad_calls[0][0] is vad
    assert "X" not in repr(model.with_vad_calls)


def test_ut103_parakeet_load_failure_is_model_load_error() -> None:
    def loader(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("encoder-model.int8.onnx: protobuf parsing failed")

    engine = ParakeetEngine(loader=loader, vad_factory=object)

    with pytest.raises(ModelLoadError):
        engine.load()


def test_ut104_cuda_out_of_memory_maps_to_engine_oom() -> None:
    engine, model = whisper()
    model.error = RuntimeError("CUDA failed with error out of memory")

    with pytest.raises(EngineOOM):
        engine.transcribe(AUDIO, language="pt", vocabulary=[], fast=False)


def test_ut104_other_failures_map_to_engine_error() -> None:
    engine, model = whisper()
    model.error = ValueError("invalid input shape")

    with pytest.raises(EngineError) as info:
        engine.transcribe(AUDIO, language="pt", vocabulary=[], fast=False)
    assert not isinstance(info.value, EngineOOM)


def test_ut104_cuda_failure_while_loading_is_engine_oom() -> None:
    def factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA failed with error out of memory")

    engine = WhisperEngine("large-v3-turbo", "cuda", model_factory=factory)

    with pytest.raises(EngineOOM):
        engine.load()


def test_ut104_corrupt_model_while_loading_is_model_load_error() -> None:
    def factory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Unable to open file 'model.bin' in model")

    engine = WhisperEngine("large-v3-turbo", "cuda", model_factory=factory)

    with pytest.raises(ModelLoadError):
        engine.load()
