"""Speech engines: the `SpeechEngine` protocol, `MODEL_CATALOG`, Whisper and Parakeet.

The heavy runtimes (faster-whisper / CTranslate2, onnx-asr) are imported lazily inside
`load()`, after `prepare_cuda_dlls()`, so importing this module never touches the GPU stack.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from talktype.logging_setup import get_logger, log_event

logger = get_logger("asr")

SAMPLE_RATE = 16_000
GIB = 1024**3
MIB = 1024**2

# The vocabulary is joined into Whisper's `initial_prompt`; the prompt window is 224
# tokens, so the joined text is capped (UT-102).
VOCAB_PROMPT_BUDGET = 800
VOCAB_SEPARATOR = ", "

# Parakeet transcribes up to this much audio in one pass; longer audio goes through VAD.
PARAKEET_MAX_SECONDS = 20

EngineKind = Literal["whisper", "parakeet"]
DeviceKind = Literal["cuda", "cpu"]

WHISPER_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)
# The same patterns onnx-asr's resolver uses for the int8 TDT files.
PARAKEET_FILES = (
    "config.json",
    "config.yaml",
    "encoder-model?int8.onnx",
    "encoder-model?int8.onnx?data",
    "decoder_joint-model?int8.onnx",
    "decoder_joint-model?int8.onnx?data",
    "vocab.txt",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str  # the `asr.model` config value
    engine: EngineKind
    repo_id: str  # Hugging Face repository
    allow_patterns: tuple[str, ...]
    download_bytes: int  # approximate snapshot size, for the "no space" message
    vram_bytes: int | None = None  # CUDA requirement; None means CPU only
    engine_id: str = ""  # the faster-whisper alias or the onnx-asr model name


MODEL_CATALOG: dict[str, ModelSpec] = {
    "large-v3-turbo": ModelSpec(
        "large-v3-turbo",
        "whisper",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        WHISPER_FILES,
        download_bytes=int(1.6 * GIB),
        vram_bytes=int(1.6 * GIB),
        engine_id="large-v3-turbo",
    ),
    "large-v3": ModelSpec(
        "large-v3",
        "whisper",
        "Systran/faster-whisper-large-v3",
        WHISPER_FILES,
        download_bytes=int(3.1 * GIB),
        vram_bytes=int(3.2 * GIB),
        engine_id="large-v3",
    ),
    "medium": ModelSpec(
        "medium",
        "whisper",
        "Systran/faster-whisper-medium",
        WHISPER_FILES,
        download_bytes=int(1.5 * GIB),
        vram_bytes=int(1.8 * GIB),
        engine_id="medium",
    ),
    "small": ModelSpec(
        "small",
        "whisper",
        "Systran/faster-whisper-small",
        WHISPER_FILES,
        download_bytes=int(0.5 * GIB),
        vram_bytes=int(0.8 * GIB),
        engine_id="small",
    ),
    "parakeet-v3": ModelSpec(
        "parakeet-v3",
        "parakeet",
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        PARAKEET_FILES,
        download_bytes=int(0.67 * GIB),
        vram_bytes=None,
        engine_id="nemo-parakeet-tdt-0.6b-v3",
    ),
}


def model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        raise ValueError(f"unknown model: {model_id}") from None


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ModelLoadError(Exception):
    """The model files could not be loaded (missing, corrupt or incompatible)."""


class EngineError(Exception):
    """A transcription failed for a reason other than GPU memory."""


class EngineOOM(EngineError):  # noqa: N818 - public name
    """CTranslate2 ran out of GPU memory or hit a CUDA error; retry on the CPU."""


def is_cuda_failure(exc: BaseException) -> bool:
    """CTranslate2 errors whose message contains "out of memory" or "CUDA"."""
    message = str(exc)
    return "out of memory" in message.lower() or "CUDA" in message


# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------

Audio = npt.NDArray[np.float32]


class SpeechEngine(Protocol):
    model_id: str
    device: DeviceKind
    supports_vocabulary: bool

    def load(self) -> None: ...  # raises ModelLoadError

    def warm_up(self) -> None: ...

    def transcribe(
        self, audio: Audio, *, language: str, vocabulary: Sequence[str], fast: bool
    ) -> str: ...  # raises EngineOOM, EngineError

    def close(self) -> None: ...


def vocabulary_prompt(
    vocabulary: Sequence[str], budget: int = VOCAB_PROMPT_BUDGET
) -> tuple[str | None, int]:
    """Join the vocabulary into a prompt of at most `budget` characters.

    Returns ``(prompt or None, dropped)``; terms that do not fit are dropped, in order.
    """
    kept: list[str] = []
    length = 0
    for term in vocabulary:
        extra = len(term) + (len(VOCAB_SEPARATOR) if kept else 0)
        if length + extra > budget:
            break
        kept.append(term)
        length += extra
    dropped = len(vocabulary) - len(kept)
    return (VOCAB_SEPARATOR.join(kept) or None), dropped


def silence(seconds: float = 1.0) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


# --------------------------------------------------------------------------------------
# Whisper (faster-whisper on CTranslate2)
# --------------------------------------------------------------------------------------

WhisperFactory = Callable[..., Any]


def _default_whisper_factory(*args: Any, **kwargs: Any) -> Any:
    from talktype.asr_device import prepare_cuda_dlls

    prepare_cuda_dlls()
    from faster_whisper import WhisperModel

    return WhisperModel(*args, **kwargs)


class WhisperEngine:
    supports_vocabulary = True

    def __init__(
        self,
        model_id: str,
        device: DeviceKind = "cuda",
        compute_type: str | None = None,
        path: Path | None = None,
        *,
        model_factory: WhisperFactory | None = None,
    ) -> None:
        self.model_id = model_id
        self.device: DeviceKind = device
        self.compute_type = compute_type or ("int8_float16" if device == "cuda" else "int8")
        self.path = path
        self.last_vocab_dropped = 0
        self._factory = model_factory or _default_whisper_factory
        self._model: Any = None

    def load(self) -> None:
        source = str(self.path) if self.path is not None else model_spec(self.model_id).engine_id
        try:
            self._model = self._factory(
                source,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )
        except Exception as exc:
            if self.device == "cuda" and is_cuda_failure(exc):
                raise EngineOOM(str(exc)) from exc
            raise ModelLoadError(f"{self.model_id}: {exc}") from exc

    def warm_up(self) -> None:
        """Transcribe 1 s of silence without VAD so the first dictation runs warm."""
        self._run(silence(), language="pt", prompt=None, beam_size=5, vad_filter=False)

    def transcribe(
        self, audio: Audio, *, language: str, vocabulary: Sequence[str], fast: bool
    ) -> str:
        prompt, dropped = vocabulary_prompt(vocabulary)
        self.last_vocab_dropped = dropped
        if dropped:
            log_event(logger, "vocab_truncated", level=logging.WARNING, dropped=dropped)
        return self._run(
            audio,
            language=language,
            prompt=prompt,
            beam_size=1 if fast else 5,
            vad_filter=not fast,
        )

    def _run(
        self,
        audio: Audio,
        *,
        language: str,
        prompt: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> str:
        if self._model is None:
            raise EngineError("engine not loaded")
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                task="transcribe",
                initial_prompt=prompt,
                beam_size=beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=False,
            )
            # Decoding runs lazily while the segments are consumed.
            return " ".join(text for s in segments if (text := s.text.strip()))
        except EngineError:
            raise
        except Exception as exc:
            if is_cuda_failure(exc):
                raise EngineOOM(str(exc)) from exc
            raise EngineError(str(exc)) from exc

    def close(self) -> None:
        self._model = None
        gc.collect()


# --------------------------------------------------------------------------------------
# Parakeet (onnx-asr, CPU only)
# --------------------------------------------------------------------------------------

ParakeetLoader = Callable[..., Any]
VadFactory = Callable[[], Any]


def _default_parakeet_loader(*args: Any, **kwargs: Any) -> Any:
    import onnx_asr

    return onnx_asr.load_model(*args, **kwargs)


def _default_vad_factory() -> Any:
    from talktype.vad import BundledSileroVad

    return BundledSileroVad()


class ParakeetEngine:
    """Parakeet TDT 0.6B v3 int8 on the CPU; ignores the vocabulary and the language."""

    supports_vocabulary = False

    def __init__(
        self,
        model_id: str = "parakeet-v3",
        path: Path | None = None,
        *,
        loader: ParakeetLoader | None = None,
        vad_factory: VadFactory | None = None,
    ) -> None:
        self.model_id = model_id
        self.device: DeviceKind = "cpu"
        self.path = path
        self._loader = loader or _default_parakeet_loader
        self._vad_factory = vad_factory or _default_vad_factory
        self._model: Any = None
        self._vad: Any = None

    def load(self) -> None:
        spec = model_spec(self.model_id)
        try:
            self._model = self._loader(
                spec.engine_id,
                self.path,
                quantization="int8",
                providers=["CPUExecutionProvider"],
            )
            self._vad = self._vad_factory()
        except Exception as exc:
            raise ModelLoadError(f"{self.model_id}: {exc}") from exc

    def warm_up(self) -> None:
        self.transcribe(silence(), language="pt", vocabulary=(), fast=False)

    def transcribe(
        self, audio: Audio, *, language: str, vocabulary: Sequence[str], fast: bool
    ) -> str:
        del language, vocabulary, fast  # Parakeet detects the language and takes no prompt
        if self._model is None:
            raise EngineError("engine not loaded")
        try:
            if len(audio) > PARAKEET_MAX_SECONDS * SAMPLE_RATE:
                segments = self._model.with_vad(
                    self._vad, max_speech_duration_s=PARAKEET_MAX_SECONDS
                ).recognize(audio, sample_rate=SAMPLE_RATE)
                return " ".join(text for s in segments if (text := s.text.strip()))
            return str(self._model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()
        except Exception as exc:
            raise EngineError(str(exc)) from exc

    def close(self) -> None:
        self._model = None
        self._vad = None
        gc.collect()


def create_engine(
    model_id: str, device: DeviceKind, compute_type: str, path: Path | None
) -> SpeechEngine:
    """The engine for `model_id` on `device` (Parakeet always runs on the CPU)."""
    spec = model_spec(model_id)
    if spec.engine == "parakeet":
        return ParakeetEngine(model_id, path)
    return WhisperEngine(model_id, device, compute_type, path)
