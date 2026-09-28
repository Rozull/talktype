"""Shared fixtures and fakes."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from talktype import logging_setup

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_AUDIO = REPO_ROOT / "tests" / "fixtures" / "audio"


class FakeClock:
    """Injectable wall clock. `sleep()` advances time and runs an optional hook."""

    def __init__(self, now: float, on_sleep: Callable[[], None] | None = None) -> None:
        self._now = now
        self.sleeps: list[float] = []
        self.on_sleep = on_sleep

    def now(self) -> float:
        return self._now

    def now_ms(self) -> int:
        return int(self._now * 1000)

    def set(self, now: float) -> None:
        self._now = now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def normalize(text: str) -> str:
    """Text comparison form for ASR assertions: casefold, no punctuation, "pra" == "para"."""
    words = re.sub(r"[^\w\s]", " ", text.casefold()).split()
    return " ".join("para" if w == "pra" else w for w in words)


@pytest.fixture
def clock() -> FakeClock:
    """A clock far enough ahead of real time that no file looks freshly written."""
    return FakeClock(time.time() + 3600)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "talktype-home"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    logging_setup.set_log_text(False)
    logging_setup.close_logging()


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Every `talktype.*` log line (rendered key=value, no timestamp) during the test."""
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    lines: list[str] = []
    formatter = logging_setup.KeyValueFormatter()

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(formatter.format(record).split(" ", 2)[2])

    handler = _Collector()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield lines
    logger.removeHandler(handler)
    logger.setLevel(previous)
