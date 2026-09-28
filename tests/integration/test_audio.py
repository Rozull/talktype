"""Real audio devices, the ConsentStore read and the real VAD (IT-029, IT-030, IT-040)."""

from __future__ import annotations

import subprocess
import sys
import time
import winreg
from typing import Any

import numpy as np
import pytest

from talktype import win32
from talktype.audio import Recorder, list_input_devices, read_wav
from talktype.vad import speech_seconds
from tests.conftest import FIXTURES_AUDIO

pytestmark = pytest.mark.integration

# Another "app" holding the default input in shared mode while the recorder opens it.
HOLDER = """
import sys, time
import sounddevice as sd
with sd.InputStream(channels=1, samplerate=16000, dtype="float32"):
    print("holding", flush=True)
    time.sleep(float(sys.argv[1]))
"""


def _has_input() -> bool:
    import sounddevice as sd

    try:
        return any(int(d["max_input_channels"]) > 0 for d in sd.query_devices())
    except Exception:
        return False


def _record(seconds: float) -> np.ndarray:
    recorder = Recorder()
    info = recorder.open()
    assert info.samplerate > 0
    time.sleep(seconds)
    return recorder.stop()


@pytest.mark.skipif(not _has_input(), reason="no input device on this machine")
def test_it029_records_from_the_default_device() -> None:
    if win32.read_mic_consent() == "Deny":
        pytest.skip("microphone access is denied in Windows privacy settings")
    assert list_input_devices()

    audio = _record(0.5)

    assert audio.dtype == np.float32
    assert audio.size / 16000 > 0.1


@pytest.mark.skipif(not _has_input(), reason="no input device on this machine")
def test_it029_shared_device_still_records() -> None:
    if win32.read_mic_consent() == "Deny":
        pytest.skip("microphone access is denied in Windows privacy settings")
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, "5"], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "holding"

        audio = _record(0.5)

        assert audio.size / 16000 > 0.1
        assert holder.poll() is None  # the other app kept its stream
    finally:
        holder.kill()
        holder.wait(10)


def test_it040_consent_read_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_writes(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the consent check must never write to the registry")

    for name in ("SetValueEx", "SetValue", "CreateKey", "CreateKeyEx", "DeleteValue", "DeleteKey"):
        monkeypatch.setattr(winreg, name, no_writes)

    assert win32.read_mic_consent() in ("Allow", "Deny", None)


def fixture(name: str) -> np.ndarray:
    samples, rate = read_wav(FIXTURES_AUDIO / f"{name}.wav")
    assert rate == 16000
    return samples


def test_it030_speech_seconds_on_the_fixtures() -> None:
    assert speech_seconds(fixture("silencio_3s")) == 0.0
    assert speech_seconds(fixture("ruido_3s")) < 0.3
    assert speech_seconds(fixture("deploy_pr")) > 8.0
    assert speech_seconds(fixture("curta_sim")) >= 0.3


def test_it030_constant_tone_is_not_speech() -> None:
    t = np.arange(3 * 16000) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    clipped = np.clip(tone * 4, -1, 1).astype(np.float32)

    assert speech_seconds(tone) < 0.3
    assert speech_seconds(clipped) < 0.3
