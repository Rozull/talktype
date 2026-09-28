"""Start and end cues with a fake `PlaySound` (UT-195, UT-196)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from talktype import win32
from talktype.config import RecordingSettings
from talktype.sounds import PLAY_FLAGS, VOLUME, Cue, Sounds, builtin_path, wav_image
from talktype.strings import Msg

pytestmark = pytest.mark.unit


class FakePlaySound:
    """Records each played image by the cue file it came from; `fail` or `raises` break it."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.flags: list[int] = []
        self.names: dict[bytes, str] = {}
        self.fail = False
        self.raises = False

    def __call__(self, sound: bytes | None, flags: int) -> bool:
        if self.raises:
            raise RuntimeError("no audio output device")
        if sound is not None:
            self.log.append(f"play {self.names.get(sound, '?')}")
            self.flags.append(flags)
        return not self.fail


class Harness:
    def __init__(self, settings: RecordingSettings) -> None:
        self.reports: list[str] = []
        self.player = FakePlaySound()
        self.sounds = Sounds(settings, report=self.reports.append, play_sound=self.player)
        self.name_images()

    def name_images(self) -> None:
        for cue in Cue:
            image = wav_image(self.sounds.source(cue))
            self.player.names[image] = self.sounds.source(cue).name

    @property
    def log(self) -> list[str]:
        return self.player.log

    def dictation(self) -> None:
        self.sounds.play_start()  # commit
        self.sounds.play_end()  # finish


def _samples(image: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(image), "rb") as source:
        return np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)


def test_builtin_cues_are_packaged() -> None:
    for cue in Cue:
        data = builtin_path(cue).read_bytes()
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def test_cues_play_at_the_configured_volume() -> None:
    path = builtin_path(Cue.START)
    original = _samples(path.read_bytes())
    scaled = _samples(wav_image(path))
    assert len(scaled) == len(original)
    assert np.abs(scaled).max() == round(np.abs(original).max() * VOLUME)


def test_ut195_sounds_off_plays_nothing() -> None:
    harness = Harness(RecordingSettings(sounds=False))
    harness.dictation()
    harness.dictation()
    assert harness.log == []
    assert harness.reports == []


def test_ut195_missing_custom_sound_uses_builtin_and_reports_once() -> None:
    harness = Harness(RecordingSettings(start_sound="C:/missing.wav"))

    for _ in range(3):
        harness.dictation()

    assert harness.sounds.source(Cue.START) == builtin_path(Cue.START)
    assert harness.log.count("play start.wav") == 3
    assert harness.reports == [Msg.SOUND_MISSING.format(path="C:/missing.wav")]


def test_ut195_existing_custom_sound_is_used(tmp_path: Path) -> None:
    custom = tmp_path / "meu-inicio.wav"
    with wave.open(str(custom), "wb") as target:
        target.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        target.writeframes(np.full(800, 1000, dtype=np.int16).tobytes())
    harness = Harness(RecordingSettings(start_sound=str(custom)))

    harness.dictation()

    assert harness.sounds.source(Cue.START) == custom
    assert harness.log[0] == "play meu-inicio.wav"
    assert harness.reports == []


def test_ut195_undecodable_custom_sound_falls_back_once(tmp_path: Path) -> None:
    custom = tmp_path / "quebrado.wav"
    custom.write_bytes(b"not a wav")
    harness = Harness(RecordingSettings(end_sound=str(custom)))

    harness.dictation()
    harness.dictation()

    assert harness.sounds.source(Cue.END) == builtin_path(Cue.END)
    assert harness.log.count("play end.wav") == 2
    assert harness.reports == [Msg.SOUND_MISSING.format(path=str(custom))]


def test_ut195_reload_turning_sounds_off_applies() -> None:
    harness = Harness(RecordingSettings())
    harness.sounds.configure(RecordingSettings(sounds=False))
    harness.dictation()
    assert harness.log == []


def test_ut196_missing_output_device_never_raises(logs: list[str]) -> None:
    harness = Harness(RecordingSettings())
    harness.player.fail = True  # PlaySound finds no output device
    harness.sounds.play_start()
    harness.player.fail = False
    harness.player.raises = True
    for _ in range(2):
        harness.dictation()  # must not raise

    assert harness.reports == []
    assert [line for line in logs if line.startswith("sound_error")] == [
        "sound_error cue=start reason=play_failed",
        "sound_error cue=end reason=RuntimeError",
    ]


def test_new_cue_stops_the_previous_one() -> None:
    harness = Harness(RecordingSettings())
    harness.sounds.play_end()
    harness.sounds.play_start()  # end is still playing: PlaySound stops it
    assert harness.log == ["play end.wav", "play start.wav"]
    # Without SND_NOSTOP, PlaySound stops the sound still playing in this process.
    assert harness.player.flags == [PLAY_FLAGS, PLAY_FLAGS]
    assert PLAY_FLAGS & win32.SND_ASYNC and PLAY_FLAGS & win32.SND_MEMORY
