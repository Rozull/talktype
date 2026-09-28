"""`Recorder` and `WavAudioSource` (UT-188–UT-194, UT-241, UT-247)."""

from __future__ import annotations

import builtins
import io
import os
import wave
from typing import Any

import numpy as np
import pytest

from talktype import win32
from talktype.audio import (
    PA_DEVICE_UNAVAILABLE,
    PA_INVALID_DEVICE,
    Level,
    MicError,
    Recorder,
    WavAudioSource,
    block_level,
    list_input_devices,
    resample,
    resolve_device,
)
from tests.conftest import FakeClock
from tests.fakes import FakeRegistry, FakeSoundDevice, PortAudioError, device, sine

pytestmark = pytest.mark.unit


def recorder(sd: FakeSoundDevice, microphone: str = "", **kwargs: Any) -> Recorder:
    kwargs.setdefault("consent", lambda: None)
    return Recorder(microphone, backend=sd, **kwargs)


def devices_with_jabra() -> FakeSoundDevice:
    return FakeSoundDevice(
        [
            device("Alto-falantes", inputs=0),
            device("Microfone (Realtek)"),
            device("Headset (Jabra Evolve2 65)"),
            device("Headset (Jabra Evolve2 65)", hostapi=2),  # same device on WASAPI
        ],
        default_input=1,
    )


def test_ut188_resolve_device() -> None:
    sd = devices_with_jabra()

    default = resolve_device(sd, "")
    assert (default.index, default.name, default.pinned_missing) == (
        1,
        "Microfone (Realtek)",
        False,
    )

    jabra = resolve_device(sd, "Jabra")
    assert (jabra.index, jabra.pinned_missing) == (2, False)
    assert resolve_device(sd, "headset (jabra evolve2 65)").index == 2

    sd.devices = [d for d in sd.devices if "Jabra" not in d["name"]]
    missing = resolve_device(sd, "Jabra")
    assert (missing.index, missing.pinned_missing) == (1, True)

    sd.devices = [device("Alto-falantes", inputs=0)]
    with pytest.raises(MicError) as info:
        resolve_device(sd, "")
    assert info.value.kind == "no_device"


def test_ut188_invalid_default_uses_the_first_input() -> None:
    sd = FakeSoundDevice([device("Saída", inputs=0), device("USB Mic")], default_input=-1)

    assert resolve_device(sd, "").index == 1
    assert list_input_devices(sd) == ["USB Mic"]


def test_ut188_open_reports_a_missing_pinned_device() -> None:
    sd = FakeSoundDevice([device("Microfone (Realtek)")])

    info = recorder(sd, "Jabra").open()

    assert info.pinned_missing is True
    assert info.device == "Microfone (Realtek)"
    assert sd.stream.kwargs["device"] == 0


def test_ut188_no_device_on_open() -> None:
    sd = FakeSoundDevice([])

    with pytest.raises(MicError) as info:
        recorder(sd).open()
    assert info.value.kind == "no_device"
    assert sd.streams == []


@pytest.mark.parametrize("key", [win32.MIC_CONSENT_KEY, win32.MIC_CONSENT_NONPACKAGED_KEY])
def test_ut189_privacy_denied_blocks_the_microphone(key: str) -> None:
    registry = FakeRegistry({(key, "Value"): "Deny"})
    sd = FakeSoundDevice()
    rec = recorder(sd, consent=lambda: win32.read_mic_consent(registry))

    with pytest.raises(MicError) as info:
        rec.open()

    assert info.value.kind == "blocked"
    assert sd.streams == []
    assert (win32.MIC_CONSENT_KEY, "Value") in registry.reads


def test_ut189_consent_values() -> None:
    assert win32.read_mic_consent(FakeRegistry()) is None
    allow = FakeRegistry({(win32.MIC_CONSENT_KEY, "Value"): "Allow"})
    assert win32.read_mic_consent(allow) == "Allow"
    mixed = FakeRegistry(
        {
            (win32.MIC_CONSENT_KEY, "Value"): "Allow",
            (win32.MIC_CONSENT_NONPACKAGED_KEY, "Value"): "Deny",
        }
    )
    assert win32.read_mic_consent(mixed) == "Deny"


@pytest.mark.parametrize(
    ("code", "kind"), [(PA_DEVICE_UNAVAILABLE, "busy"), (PA_INVALID_DEVICE, "no_device")]
)
def test_ut190_portaudio_errors_map_to_mic_errors(code: int, kind: str, logs: list[str]) -> None:
    sd = FakeSoundDevice()
    sd.start_error = PortAudioError("Device unavailable", code)

    with pytest.raises(MicError) as info:
        recorder(sd).open()

    assert info.value.kind == kind
    assert f"mic_error kind={kind}" in logs


def test_ut191_level_of_a_half_amplitude_sine() -> None:
    block = sine(0.032, amplitude=0.5, freq=1000.0)
    assert block.size == 512

    level = block_level(block)

    assert level.rms == pytest.approx(0.35, abs=0.01)
    assert level.peak_dbfs == pytest.approx(-6.0, abs=0.1)
    assert block_level(np.zeros(512, dtype=np.float32)) == Level(0.0, -120.0)


def test_ut191_recorder_meters_each_block() -> None:
    sd = FakeSoundDevice()
    levels: list[Level] = []
    rec = recorder(sd, on_level=levels.append)
    rec.open()

    sd.stream.feed(sine(0.032, amplitude=0.5, freq=1000.0))

    assert len(levels) == 1
    assert rec.level.rms == pytest.approx(0.35, abs=0.01)


def wav_bytes(seconds: float, rate: int = 16000) -> io.BytesIO:
    """An in-memory 16-bit WAV (nothing is written to disk)."""
    samples = (sine(seconds, rate=rate, amplitude=0.25) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    buffer.seek(0)
    return buffer


def test_ut192_wav_source_plays_in_real_time_then_silence() -> None:
    clock = FakeClock(100.0)
    blocks: list[np.ndarray] = []
    times: list[float] = []

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        assert indata.shape == (512, 1) and frames == 512
        blocks.append(indata[:, 0].copy())
        times.append(clock.now())

    source = WavAudioSource(wav_bytes(2.0), callback=callback, clock=clock)
    assert source.run(max_blocks=70) == 70

    with_audio = [i for i, b in enumerate(blocks) if np.any(b)]
    assert with_audio == list(range(63))  # 32000 samples = 62.5 blocks of 32 ms
    assert all(not np.any(b) for b in blocks[63:])
    assert times[0] == 100.0
    assert times[63] == pytest.approx(100.0 + 63 * 0.032)
    assert clock.now() == pytest.approx(100.0 + 70 * 0.032)


def test_ut192_recorder_uses_the_wav_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[WavAudioSource] = []
    monkeypatch.setattr(WavAudioSource, "start", lambda self: started.append(self))
    from tests.conftest import FIXTURES_AUDIO

    rec = Recorder(audio_source=FIXTURES_AUDIO / "curta_sim.wav", consent=lambda: "Deny")
    info = rec.open()

    assert info.device == "wav:curta_sim.wav"
    assert rec.samplerate == 16000
    source = started[0]
    source.run(max_blocks=10)
    assert rec.stop().size == 10 * 512


def test_ut193_stop_concatenates_blocks_and_discard_empties() -> None:
    sd = FakeSoundDevice()
    rec = recorder(sd)
    rec.open()
    b1, b2, b3 = (np.full(512, v, dtype=np.float32) for v in (0.1, 0.2, 0.3))
    for block in (b1, b2, b3):
        sd.stream.feed(block)

    audio = rec.stop()

    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, np.concatenate([b1, b2, b3]))
    assert sd.stream.kwargs["samplerate"] == 16000
    assert sd.stream.kwargs["blocksize"] == 512
    assert sd.stream.kwargs["channels"] == 1
    assert sd.stream.closed

    rec.open()
    sd.stream.feed(b1)
    rec.discard()
    assert rec.stop().size == 0
    assert not rec.is_open


def test_ut193_snapshot_returns_the_last_seconds() -> None:
    sd = FakeSoundDevice()
    rec = recorder(sd)
    rec.open()
    for _ in range(100):  # 3.2 s
        sd.stream.feed(np.ones(512, dtype=np.float32))

    assert rec.snapshot(1.0).size == 16000
    assert rec.snapshot().size == 100 * 512
    assert rec.is_open


def test_ut194_native_rate_is_resampled_to_16k() -> None:
    sd = FakeSoundDevice([device("USB Mic", rate=44100.0)], rates={0: [44100]})
    rec = recorder(sd)

    info = rec.open()
    assert info.samplerate == 44100
    assert sd.stream.kwargs["samplerate"] == 44100
    assert sd.stream.kwargs["blocksize"] == round(44100 * 0.032)
    tone = sine(2.0, rate=44100, amplitude=0.5)
    for start in range(0, tone.size, 1411):
        sd.stream.feed(tone[start : start + 1411])

    audio = rec.stop()

    assert abs(audio.size - 2.0 * 16000) <= 1
    assert audio.dtype == np.float32
    assert block_level(audio[1000:-1000]).rms == pytest.approx(0.35, abs=0.01)


def test_ut194_resample_length_and_identity() -> None:
    x = sine(1.0, rate=48000)
    assert resample(x, 48000).size == 16000
    assert resample(sine(1.0, rate=8000), 8000).size == 16000
    same = sine(0.5)
    np.testing.assert_array_equal(resample(same, 16000), same)


def test_ut241_each_open_refreshes_the_device_list() -> None:
    sd = FakeSoundDevice([device("Microfone (Realtek)")])
    rec = recorder(sd)
    rec.open()
    rec.stop()
    assert sd.query_count == 1

    sd.devices = [device("Microfone (Realtek)"), device("Headset (Jabra)")]
    sd.default.device = [1, -1]  # the headset became the Windows default
    info = rec.open()
    rec.stop()

    assert sd.query_count == 2
    assert info.device == "Headset (Jabra)"


def test_mic_opened_is_logged_with_open_and_first_block_times(logs: list[str]) -> None:
    clock = FakeClock(10.0)
    sd = FakeSoundDevice([device("Microfone (Realtek)")])
    rec = recorder(sd, clock=clock)
    rec.open()
    assert not any(line.startswith("mic_opened") for line in logs)

    clock.advance(0.05)
    sd.stream.feed(np.zeros(512, dtype=np.float32))
    sd.stream.feed(np.zeros(512, dtype=np.float32))

    opened = [line for line in logs if line.startswith("mic_opened")]
    assert opened == [
        'mic_opened device="Microfone (Realtek)" rate=16000 open_ms=0 '
        "pinned_missing=False first_block_ms=50"
    ]


def test_device_loss_is_detected_and_keeps_the_captured_audio() -> None:
    sd = FakeSoundDevice()
    lost: list[bool] = []
    rec = recorder(sd, on_device_lost=lambda: lost.append(True))
    rec.open()
    sd.stream.feed(np.ones(512, dtype=np.float32))
    assert rec.device_lost is False

    sd.stream.unplug()

    assert rec.device_lost is True
    assert lost == [True]
    assert rec.stop().size == 512


def test_ut247_audio_never_touches_the_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []

    def forbidden(name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            writes.append(name)
            raise AssertionError(f"{name} called during recording")

        return fail

    sd = FakeSoundDevice()
    rec = recorder(sd)
    monkeypatch.setattr(builtins, "open", forbidden("open"))
    monkeypatch.setattr(io, "open", forbidden("io.open"))
    monkeypatch.setattr(os, "open", forbidden("os.open"))
    monkeypatch.setattr(wave, "open", forbidden("wave.open"))
    try:
        import soundfile  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        monkeypatch.setattr(soundfile, "write", forbidden("soundfile.write"))

    rec.open()
    for _ in range(10):
        sd.stream.feed(sine(0.032))
    audio = rec.stop()
    rec.open()
    sd.stream.feed(sine(0.032))
    rec.snapshot(1.0)
    rec.discard()
    monkeypatch.undo()

    assert audio.size == 10 * 512
    assert writes == []
