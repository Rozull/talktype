"""Test seams read from `TALKTYPE_*` environment variables.

Every seam is inert unless its variable is set. `current()` reads the environment once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

AUDIO_SOURCE_ENV = "TALKTYPE_AUDIO_SOURCE"
ACCEPT_INJECTED_ENV = "TALKTYPE_TEST_ACCEPT_INJECTED"
NO_MSGBOX_ENV = "TALKTYPE_TEST_NO_MSGBOX"


@dataclass(frozen=True, slots=True)
class Seams:
    audio_source: str | None = None  # raw "wav:<path>" value
    accept_injected: bool = False
    no_msgbox: bool = False

    @property
    def audio_wav(self) -> Path | None:
        """The WAV path of `TALKTYPE_AUDIO_SOURCE=wav:<path>`, or None."""
        if self.audio_source and self.audio_source.startswith("wav:"):
            return Path(self.audio_source.removeprefix("wav:"))
        return None

    def active(self) -> list[str]:
        names: list[str] = []
        if self.audio_source:
            names.append("audio_source")
        if self.accept_injected:
            names.append("accept_injected")
        if self.no_msgbox:
            names.append("no_msgbox")
        return names


def read(env: Mapping[str, str] | None = None) -> Seams:
    env = os.environ if env is None else env
    return Seams(
        audio_source=env.get(AUDIO_SOURCE_ENV) or None,
        accept_injected=env.get(ACCEPT_INJECTED_ENV) == "1",
        no_msgbox=env.get(NO_MSGBOX_ENV) == "1",
    )


@cache
def current() -> Seams:
    return read()


def active(seams: Seams | None = None) -> list[str]:
    """The names of the seams that are set."""
    return (current() if seams is None else seams).active()


def log_active(logger: logging.Logger, seams: Seams | None = None) -> list[str]:
    """Log one `test_seam_active` line per active seam and return their names."""
    from talktype.logging_setup import log_event

    names = active(seams)
    for name in names:
        log_event(logger, "test_seam_active", level=logging.WARNING, seam=name)
    return names
