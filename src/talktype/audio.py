"""Microphone capture: `Recorder`, `WavAudioSource`, device resolution and level metering.

16 kHz mono float32 in 32 ms blocks; the Windows default input or a pinned device;
the device list is refreshed on every open; the ConsentStore privacy value is checked first.
Audio lives in memory only and is never written to disk.

Threads: `open()`, `stop()`, `discard()` and `snapshot()` are called from the Qt GUI thread;
the stream callback, `on_level` and `on_device_lost` run on the PortAudio (or WAV) thread.
"""

from __future__ import annotations

import math
import threading
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol

import numpy as np
import numpy.typing as npt

from talktype import win32
from talktype.logging_setup import get_logger, log_event

logger = get_logger("audio")

SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.032
MIN_DBFS = -120.0

# PortAudio error codes (portaudio.h)
PA_INVALID_DEVICE = -9996
PA_DEVICE_UNAVAILABLE = -9985

Audio = npt.NDArray[np.float32]
MicErrorKind = Literal["blocked", "no_device", "busy"]


class MicError(Exception):
    """The microphone cannot be opened; `kind` selects the user-facing message."""

    def __init__(self, kind: MicErrorKind, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind: MicErrorKind = kind


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class _RealClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class SoundBackend(Protocol):
    """The subset of the `sounddevice` module the recorder uses (faked in tests)."""

    default: Any

    @property
    def PortAudioError(self) -> type[Exception]: ...  # noqa: N802 (sounddevice API)

    def query_devices(self) -> Any: ...

    def check_input_settings(
        self, device: Any = ..., channels: Any = ..., dtype: Any = ..., samplerate: Any = ...
    ) -> None: ...

    def InputStream(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: N802 (sounddevice API)


def _sounddevice() -> SoundBackend:
    import sounddevice

    return sounddevice  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Level:
    rms: float
    peak_dbfs: float


def block_level(block: npt.NDArray[np.floating[Any]]) -> Level:
    """RMS and peak (dBFS) of one block; silence is `MIN_DBFS`."""
    if block.size == 0:
        return Level(0.0, MIN_DBFS)
    samples = block.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    peak_dbfs = 20.0 * math.log10(peak) if peak > 0 else MIN_DBFS
    return Level(rms, max(peak_dbfs, MIN_DBFS))


def _lowpass_taps(cutoff: float, taps: int = 63) -> npt.NDArray[np.float64]:
    """Hamming-windowed sinc low-pass; `cutoff` in cycles per sample (< 0.5)."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff * n) * np.hamming(taps)
    return h / h.sum()


def resample(
    audio: npt.NDArray[np.floating[Any]], src_rate: int, dst_rate: int = SAMPLE_RATE
) -> Audio:
    """Band-limited linear resampling to `dst_rate`; output length is ``round(n * dst/src)``."""
    samples = np.asarray(audio, dtype=np.float64)
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    if src_rate > dst_rate:
        samples = np.convolve(samples, _lowpass_taps(0.475 * dst_rate / src_rate), mode="same")
    n_out = round(samples.size * dst_rate / src_rate)
    positions = np.arange(n_out) * (src_rate / dst_rate)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    index: int
    name: str
    samplerate: float
    pinned_missing: bool = False


def _input_devices(devices: Sequence[Any]) -> list[tuple[int, Any]]:
    return [(i, d) for i, d in enumerate(devices) if int(d["max_input_channels"]) > 0]


def _default_input_index(backend: SoundBackend, inputs: list[tuple[int, Any]]) -> int:
    default = backend.default.device  # an (input, output) pair; -1 means none
    try:
        index = int(default[0])
    except (TypeError, IndexError, KeyError):
        index = int(default) if isinstance(default, int) else -1
    return index if any(i == index for i, _ in inputs) else inputs[0][0]


def list_input_devices(backend: SoundBackend | None = None) -> list[str]:
    """Input device names on the default device's host API, for the tray menu."""
    backend = backend or _sounddevice()
    devices = backend.query_devices()
    inputs = _input_devices(devices)
    if not inputs:
        return []
    default = _default_input_index(backend, inputs)
    hostapi = devices[default]["hostapi"]
    return [str(d["name"]) for _, d in inputs if d["hostapi"] == hostapi]


def resolve_device(backend: SoundBackend, pinned: str = "") -> DeviceChoice:
    """The pinned input device when present, otherwise the Windows default.

    Candidates are the inputs of the default device's host API. A pinned name matches
    exactly (case-insensitive) or as a substring in either direction. Raises
    `MicError("no_device")` when there is no input at all.
    """
    devices = backend.query_devices()
    inputs = _input_devices(devices)
    if not inputs:
        raise MicError("no_device")
    default = _default_input_index(backend, inputs)
    hostapi = devices[default]["hostapi"]
    candidates = [(i, d) for i, d in inputs if d["hostapi"] == hostapi]
    pinned_missing = False
    chosen = default
    wanted = pinned.strip().casefold()
    if wanted:
        names = [(i, str(d["name"]).casefold()) for i, d in candidates]
        match = next((i for i, n in names if n == wanted), None)
        if match is None:
            match = next((i for i, n in names if wanted in n or n in wanted), None)
        if match is None:
            pinned_missing = True
        else:
            chosen = match
    device = devices[chosen]
    return DeviceChoice(
        chosen, str(device["name"]), float(device["default_samplerate"]), pinned_missing
    )


# --------------------------------------------------------------------------------------
# WAV test seam (TALKTYPE_AUDIO_SOURCE=wav:<path>)
# --------------------------------------------------------------------------------------

StreamCallback = Callable[[Any, int, Any, Any], None]


def read_wav(source: Path | BinaryIO) -> tuple[Audio, int]:
    """Read a 16-bit PCM WAV (a path or a binary file object) into mono float32 samples."""
    with wave.open(str(source) if isinstance(source, Path) else source, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{source}: only 16-bit PCM WAV is supported")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return samples, rate


class WavAudioSource:
    """Stream stand-in that plays a WAV into the callback in real time, then silence.

    It mimics the parts of `sounddevice.InputStream` the recorder uses.
    """

    def __init__(
        self,
        source: Path | BinaryIO,
        *,
        callback: StreamCallback,
        clock: Clock | None = None,
        block_seconds: float = BLOCK_SECONDS,
    ) -> None:
        self.samples, self.samplerate = read_wav(source)
        self.blocksize = round(self.samplerate * block_seconds)
        self.block_seconds = self.blocksize / self.samplerate
        self._callback = callback
        self._clock = clock or _RealClock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def blocks(self) -> Iterator[Audio]:
        """The file in `blocksize` blocks (the last one zero-padded), then silence forever."""
        size = self.blocksize
        for start in range(0, self.samples.size, size):
            block = self.samples[start : start + size]
            if block.size < size:
                block = np.concatenate([block, np.zeros(size - block.size, dtype=np.float32)])
            yield block
        silence = np.zeros(size, dtype=np.float32)
        while True:
            yield silence

    def run(self, max_blocks: int | None = None) -> int:
        """Deliver blocks at real-time pace until stopped (or `max_blocks`); returns the count."""
        start = self._clock.now()
        count = 0
        for block in self.blocks():
            if self._stop.is_set() or (max_blocks is not None and count >= max_blocks):
                break
            self._callback(block.reshape(-1, 1), self.blocksize, None, None)
            count += 1
            delay = start + count * self.block_seconds - self._clock.now()
            if delay > 0:
                self._clock.sleep(delay)
        return count

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="talktype-wav-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def close(self) -> None:
        self.stop()


# --------------------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenInfo:
    device: str
    samplerate: int
    pinned_missing: bool = False


class Recorder:
    """Owns one input stream per dictation; buffers blocks in memory only."""

    def __init__(
        self,
        microphone: str = "",
        *,
        backend: SoundBackend | None = None,
        consent: Callable[[], str | None] | None = None,
        audio_source: Path | None = None,
        clock: Clock | None = None,
        on_level: Callable[[Level], None] | None = None,
        on_device_lost: Callable[[], None] | None = None,
        refresh_portaudio: bool = True,
    ) -> None:
        self.microphone = microphone
        self._backend = backend
        self._consent = consent or win32.read_mic_consent
        self._audio_source = audio_source
        self._clock = clock or _RealClock()
        self.on_level = on_level
        self.on_device_lost = on_device_lost
        self._refresh_portaudio = refresh_portaudio
        self._lock = threading.Lock()
        self._blocks: list[Audio] = []
        self._stream: Any = None
        self._rate = SAMPLE_RATE
        self._stopping = False
        self._lost = False
        self._opened_at = 0.0
        self._first_block_ms: int | None = None
        self._open_fields: dict[str, Any] | None = None
        self.level = Level(0.0, MIN_DBFS)

    @property
    def backend(self) -> SoundBackend:
        if self._backend is None:
            self._backend = _sounddevice()
        return self._backend

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def samplerate(self) -> int:
        return self._rate

    @property
    def buffer_bytes(self) -> int:
        """Bytes of audio currently buffered (zero after `stop()` or `discard()`)."""
        with self._lock:
            return sum(block.nbytes for block in self._blocks)

    @property
    def device_lost(self) -> bool:
        """True when the stream ended by itself (device unplugged) while recording."""
        if self._lost:
            return True
        stream = self._stream
        return stream is not None and not self._stopping and not bool(stream.active)

    # --- stream callbacks (PortAudio thread) ------------------------------------------

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info, status
        data = np.asarray(indata, dtype=np.float32)
        block = (data[:, 0] if data.ndim == 2 else data).copy()
        with self._lock:
            self._blocks.append(block)
        self.level = block_level(block)
        if self._first_block_ms is None:
            with self._lock:
                self._first_block_ms = round((self._clock.now() - self._opened_at) * 1000)
            self._log_opened()
        if self.on_level is not None:
            self.on_level(self.level)

    def _finished(self) -> None:
        if not self._stopping:
            self._lost = True
            log_event(logger, "mic_device_lost")
            if self.on_device_lost is not None:
                self.on_device_lost()

    # --- public API (GUI thread) ------------------------------------------------------

    def open(self) -> OpenInfo:
        """Start capturing. Raises `MicError` (``blocked``, ``no_device``, ``busy``)."""
        if self._stream is not None:
            self.discard()
        with self._lock:
            self._blocks = []
        self._stopping = False
        self._lost = False
        self._first_block_ms = None
        self._open_fields = None
        self._opened_at = self._clock.now()
        try:
            info = self._open_wav() if self._audio_source else self._open_device()
        except MicError as exc:
            log_event(logger, "mic_error", kind=exc.kind)
            raise
        with self._lock:
            self._open_fields = {
                "device": info.device,
                "rate": info.samplerate,
                "open_ms": round((self._clock.now() - self._opened_at) * 1000),
                "pinned_missing": info.pinned_missing,
            }
        self._log_opened()
        return info

    def _log_opened(self) -> None:
        """Log `mic_opened` once both the open time and the first block time are known."""
        with self._lock:
            if self._open_fields is None or self._first_block_ms is None:
                return
            fields, self._open_fields = self._open_fields, None
            fields["first_block_ms"] = self._first_block_ms
        log_event(logger, "mic_opened", **fields)

    def _open_wav(self) -> OpenInfo:
        assert self._audio_source is not None
        source = WavAudioSource(self._audio_source, callback=self._callback, clock=self._clock)
        self._rate = source.samplerate
        self._stream = source
        source.start()
        return OpenInfo(f"wav:{self._audio_source.name}", source.samplerate)

    def _open_device(self) -> OpenInfo:
        try:
            consent = self._consent()
        except OSError:
            consent = None
        if consent == "Deny":
            raise MicError("blocked")
        backend = self.backend
        if self._refresh_portaudio:
            _reinitialize_portaudio(backend)
        choice = resolve_device(backend, self.microphone)
        rate = self._pick_rate(backend, choice)
        try:
            stream = backend.InputStream(
                device=choice.index,
                channels=1,
                samplerate=rate,
                dtype="float32",
                blocksize=round(rate * BLOCK_SECONDS),
                callback=self._callback,
                finished_callback=self._finished,
            )
            stream.start()
        except backend.PortAudioError as exc:
            code = exc.args[1] if len(exc.args) > 1 else None
            kind: MicErrorKind = "no_device" if code == PA_INVALID_DEVICE else "busy"
            raise MicError(kind, str(exc)) from exc
        self._rate = rate
        self._stream = stream
        return OpenInfo(choice.name, rate, choice.pinned_missing)

    @staticmethod
    def _pick_rate(backend: SoundBackend, choice: DeviceChoice) -> int:
        """16 kHz when the device accepts it, otherwise its native rate (resampled later)."""
        try:
            backend.check_input_settings(
                device=choice.index, channels=1, dtype="float32", samplerate=SAMPLE_RATE
            )
            return SAMPLE_RATE
        except Exception:
            return round(choice.samplerate)

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        self._stopping = True
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            self._lost = True
            log_event(logger, "mic_close_failed", exc=type(exc).__name__)

    def _collect(self, max_samples: int | None = None) -> Audio:
        with self._lock:
            blocks = list(self._blocks)
        if max_samples is not None:
            picked: list[Audio] = []
            total = 0
            for block in reversed(blocks):
                if total >= max_samples:
                    break
                picked.append(block)
                total += block.size
            blocks = picked[::-1]
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def snapshot(self, seconds: float | None = None) -> Audio:
        """A 16 kHz copy of the audio so far (only the last `seconds` when given)."""
        if seconds is None:
            return resample(self._collect(), self._rate)
        margin = 64  # filter warm-up samples at the native rate
        native = self._collect(math.ceil(seconds * self._rate) + margin)
        audio = resample(native, self._rate)
        return audio[-round(seconds * SAMPLE_RATE) :]

    def stop(self) -> Audio:
        """Close the stream and return the recording as 16 kHz mono float32."""
        self._close_stream()
        audio = resample(self._collect(), self._rate)
        with self._lock:
            self._blocks = []
        return audio

    def discard(self) -> None:
        """Close the stream and drop the buffered audio."""
        self._close_stream()
        with self._lock:
            self._blocks = []


def _reinitialize_portaudio(backend: SoundBackend) -> None:
    """Re-enumerate devices so a hot-plugged default is seen (PortAudio caches the list)."""
    terminate = getattr(backend, "_terminate", None)
    initialize = getattr(backend, "_initialize", None)
    if terminate is None or initialize is None:
        return
    try:
        terminate()
        initialize()
    except Exception as exc:
        log_event(logger, "mic_refresh_failed", exc=type(exc).__name__)


def create_recorder(microphone: str = "", *, audio_wav: Path | None = None) -> Recorder:
    """The app's recorder; `audio_wav` is the `TALKTYPE_AUDIO_SOURCE=wav:<path>` seam."""
    if audio_wav is None:
        from talktype import seams

        audio_wav = seams.current().audio_wav
    return Recorder(microphone, audio_source=audio_wav)
