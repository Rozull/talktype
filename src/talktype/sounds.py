"""Start and end cues through the Windows ``PlaySound`` API.

The built-in cues are the packaged ``assets/start.wav`` and ``assets/end.wav``. A custom
``recording.start_sound`` / ``end_sound`` that is missing or cannot be decoded falls back to
the built-in cue and is reported once per path. Each cue is decoded once, scaled to
`VOLUME` and kept in memory. PlaySound plays one sound per process, so a new cue stops the
one still playing and cues never overlap. A missing output device is logged and otherwise
ignored: dictation goes on silently. Qt GUI thread only.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Callable
from enum import StrEnum
from importlib import resources
from pathlib import Path

import numpy as np

from talktype import win32
from talktype.config import RecordingSettings
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg

VOLUME = 0.6
PLAY_FLAGS = win32.SND_MEMORY | win32.SND_ASYNC | win32.SND_NODEFAULT

_log = get_logger("sounds")

# PCM sample width in bytes -> numpy sample type (8-bit WAV is unsigned).
_SAMPLE_TYPES: dict[int, type[np.integer]] = {1: np.uint8, 2: np.int16, 4: np.int32}

PlaySound = Callable[[bytes | None, int], bool]
"""`win32.PlaySoundW`'s shape (a seam for tests)."""


class Cue(StrEnum):
    START = "start"
    END = "end"


def builtin_path(cue: Cue) -> Path:
    return Path(str(resources.files("talktype").joinpath("assets", f"{cue.value}.wav")))


def wav_image(path: Path, volume: float = VOLUME) -> bytes:
    """The PCM WAV file at `path`, rewritten in memory with its samples scaled by `volume`.

    Raises `OSError`, `EOFError` or `wave.Error` when the file cannot be read or decoded.
    """
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(params.nframes)
    sample_type = _SAMPLE_TYPES.get(params.sampwidth)
    if sample_type is None:
        raise wave.Error(f"unsupported sample width: {params.sampwidth} bytes")
    samples = np.frombuffer(frames, dtype=sample_type).astype(np.float64)
    center = 128.0 if params.sampwidth == 1 else 0.0
    scaled = np.round((samples - center) * volume + center).astype(sample_type)
    image = io.BytesIO()
    with wave.open(image, "wb") as target:
        target.setparams(params)
        target.writeframes(scaled.tobytes())
    return image.getvalue()


class Sounds:
    def __init__(
        self,
        settings: RecordingSettings,
        *,
        report: Callable[[str], None] | None = None,
        play_sound: PlaySound | None = None,
    ) -> None:
        self._report = report or (lambda _text: None)
        self._play_sound: PlaySound = play_sound or win32.PlaySoundW
        self._images: dict[Cue, bytes] = {}
        self._sources: dict[Cue, Path] = {}
        self._custom: dict[Cue, str] = {}  # configured custom path ("" = built-in)
        self._using_custom: dict[Cue, bool] = {}
        self._playing: bytes | None = None  # PlaySound reads it until the next cue
        self._reported: set[str] = set()
        self._errors_logged: set[Cue] = set()
        self._enabled = True
        self.configure(settings)

    # -- configuration -------------------------------------------------------------------

    def configure(self, settings: RecordingSettings) -> None:
        """Apply `recording.sounds`, `start_sound` and `end_sound` (at start and reload)."""
        self._enabled = settings.sounds
        for cue, custom in ((Cue.START, settings.start_sound), (Cue.END, settings.end_sound)):
            custom = custom.strip()
            if custom == self._custom.get(cue) and cue in self._using_custom:
                continue
            self._custom[cue] = custom
            self._using_custom[cue] = bool(custom) and self._load(cue, Path(custom).expanduser())
            if not self._using_custom[cue]:
                self._load(cue, builtin_path(cue))

    def _load(self, cue: Cue, path: Path) -> bool:
        try:
            self._images[cue] = wav_image(path)
        except (OSError, EOFError, wave.Error):
            if path == builtin_path(cue):
                raise
            return False
        self._sources[cue] = path
        return True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def source(self, cue: Cue) -> Path:
        """The file `cue` currently plays."""
        return self._sources[cue]

    # -- playback ------------------------------------------------------------------------

    def play_start(self) -> None:
        self.play(Cue.START)

    def play_end(self) -> None:
        self.play(Cue.END)

    def play(self, cue: Cue) -> None:
        """Play `cue`, stopping any other cue first. Never raises."""
        if not self._enabled:
            return
        self._report_missing(cue)
        self._playing = self._images[cue]
        try:
            played = self._play_sound(self._playing, PLAY_FLAGS)
        except Exception as exc:  # a broken audio backend must never break a dictation
            self._log_error(cue, type(exc).__name__)
            return
        if not played:
            self._log_error(cue, "play_failed")

    # -- failures ------------------------------------------------------------------------

    def _report_missing(self, cue: Cue) -> None:
        custom = self._custom.get(cue, "")
        if custom and not self._using_custom.get(cue, False) and custom not in self._reported:
            self._reported.add(custom)
            self._report(Msg.SOUND_MISSING.format(path=custom))

    def _log_error(self, cue: Cue, reason: str) -> None:
        if cue not in self._errors_logged:
            self._errors_logged.add(cue)
            log_event(_log, "sound_error", cue=cue.value, reason=reason)
