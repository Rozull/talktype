"""Speech detection with faster-whisper's bundled Silero ONNX model.

No `silero-vad` package and no torch: `faster_whisper.vad.get_speech_timestamps` runs the
bundled ``silero_vad_v6.onnx`` on the CPU `onnxruntime`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from onnx_asr.vad import BaseVad

SAMPLE_RATE = 16_000


def _speech_timestamps(audio: npt.NDArray[np.float32], **options: Any) -> list[dict[str, int]]:
    from talktype.asr_device import prepare_cuda_dlls

    prepare_cuda_dlls()  # importing faster_whisper imports ctranslate2
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(audio, VadOptions(**options), sampling_rate=SAMPLE_RATE)


def speech_seconds(audio: npt.NDArray[np.float32]) -> float:
    """Seconds of detected speech in 16 kHz mono float32 `audio`.

    Padding is disabled so the value measures voice only, which is what the
    `min_speech_seconds` rule compares against.
    """
    if audio.size == 0:
        return 0.0
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    chunks = _speech_timestamps(samples, speech_pad_ms=0)
    return sum(c["end"] - c["start"] for c in chunks) / SAMPLE_RATE


class BundledSileroVad(BaseVad):
    """onnx-asr VAD adapter over the bundled Silero model, used by Parakeet for long audio."""

    def segment_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        sample_rate: Literal[8_000, 16_000],
        **kwargs: float,
    ) -> Iterator[Iterator[tuple[int, int]]]:
        for waveform, length in zip(waveforms, waveforms_len, strict=True):
            samples = np.ascontiguousarray(waveform[: int(length)], dtype=np.float32)
            chunks = _speech_timestamps(samples, speech_pad_ms=0)
            segments = iter([(c["start"], c["end"]) for c in chunks])
            yield self._merge_segments(segments, int(length), sample_rate, **kwargs)
