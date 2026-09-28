"""End-to-end harness.

- **Subprocess mode** (`launch`): `python -m talktype` with a fresh `TALKTYPE_HOME` and the
  seams `TALKTYPE_AUDIO_SOURCE`, `TALKTYPE_TEST_ACCEPT_INJECTED=1` and
  `TALKTYPE_TEST_NO_MSGBOX=1`, for keyboard journeys.
- **In-process mode** (`inprocess`): `build_app()` inside the test's `QApplication`, for
  journeys that trigger tray actions with `QAction.trigger()`.
- **Targets**: `TextHarness` (a `QTextEdit` that records its key events) and `SubmitEdit`
  (a chat-like box that emits `submitted` on a bare Enter).
- **Input**: real `SendInput` through `tests.desktop`.
- **Assertions**: the log (`AppProcess.log_lines`), `history.json`, the clipboard and
  `normalize()`.

The suite takes over the keyboard focus and the clipboard, so it runs serially: a lock file
keeps two e2e sessions from overlapping, and every test releases the modifiers it pressed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import msvcrt
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QTextEdit

from talktype import logging_setup, seams, win32
from talktype.config import ConfigManager, set_value
from talktype.paths import Paths
from tests import desktop
from tests.conftest import FIXTURES_AUDIO, REPO_ROOT, normalize

READY_TIMEOUT_S = 90.0  # a cold start loads the model from the HF cache
EXPECTED = tomllib.loads((FIXTURES_AUDIO / "expected.toml").read_text(encoding="utf-8"))


def fixture_wav(name: str) -> Path:
    return FIXTURES_AUDIO / f"{name}.wav"


def key_terms(name: str) -> list[str]:
    return list(EXPECTED[name]["key_terms"])


def duration_ms(name: str) -> int:
    return round(float(EXPECTED[name]["duration_s"]) * 1000)


def contains_terms(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return all(normalize(term) in normalized for term in terms)


__all__ = [
    "AppProcess",
    "InProcess",
    "SubmitEdit",
    "TextHarness",
    "configure",
    "contains_terms",
    "duration_ms",
    "fixture_wav",
    "history",
    "key_terms",
    "normalize",
]


# --------------------------------------------------------------------------------------
# Serial execution and desktop hygiene
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _serial_e2e() -> Iterator[None]:
    """One e2e session at a time on this desktop."""
    path = Path(tempfile.gettempdir()) / "talktype_e2e.lock"
    with path.open("a+b") as handle:
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(1)
        yield


@pytest.fixture(autouse=True)
def _release_keys() -> Iterator[None]:
    yield
    desktop.release_modifiers()


# --------------------------------------------------------------------------------------
# Config, history and log
# --------------------------------------------------------------------------------------


def configure(home: Path, **values: Any) -> None:
    """Create `home/config.toml` from the template and set dotted keys (``a__b`` = ``a.b``)."""
    ConfigManager(home)
    for key, value in values.items():
        assert set_value(Paths(home).config, key.replace("__", "."), value).saved


def history(home: Path) -> list[dict[str, Any]]:
    path = Paths(home).history
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(home: Path) -> list[str]:
    path = Paths(home).log_file
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def fields(line: str) -> dict[str, str]:
    """``key=value`` pairs of a rendered log line (quoted values unquoted)."""
    pairs = re.findall(r'(\w+)=("(?:[^"\\]|\\.)*"|\S*)', line)
    return {k: v[1:-1] if v.startswith('"') else v for k, v in pairs}


class LogView:
    def __init__(self, home: Path) -> None:
        self.home = home

    def lines(self) -> list[str]:
        return read_log(self.home)

    def events(self, event: str) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        for line in self.lines():
            parts = line.split(" ", 3)
            if len(parts) >= 3 and parts[2] == event:
                found.append(fields(parts[3] if len(parts) > 3 else ""))
        return found

    def has(self, event: str, **match: str) -> bool:
        return any(all(f.get(k) == v for k, v in match.items()) for f in self.events(event))

    def wait(
        self, event: str, timeout_s: float = 30.0, *, fail_on: str | None = None, **match: str
    ) -> dict[str, str]:
        """The first `event` whose fields equal `match`.

        Fails at once when a `fail_on` event (with the same `match`) is logged first.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            for item in self.events(event):
                if all(item.get(k) == v for k, v in match.items()):
                    return item
            if fail_on is not None and any(
                all(item.get(k) == v for k, v in match.items()) for item in self.events(fail_on)
            ):
                tail = "\n".join(self.lines()[-30:])
                raise AssertionError(f"{fail_on} {match} before {event}:\n{tail}")
            if time.monotonic() > deadline:
                tail = "\n".join(self.lines()[-30:])
                raise AssertionError(f"no {event} {match} in the log:\n{tail}")
            desktop.pump(50)

    def count(self, event: str, **match: str) -> int:
        return sum(1 for f in self.events(event) if all(f.get(k) == v for k, v in match.items()))


# --------------------------------------------------------------------------------------
# Subprocess mode
# --------------------------------------------------------------------------------------


def seam_env(home: Path, wav: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    env["TALKTYPE_HOME"] = str(home)
    env["TALKTYPE_TEST_ACCEPT_INJECTED"] = "1"
    env["TALKTYPE_TEST_NO_MSGBOX"] = "1"
    env.pop("TALKTYPE_AUDIO_SOURCE", None)
    if wav is not None:
        env["TALKTYPE_AUDIO_SOURCE"] = f"wav:{wav}"
    env["PYTHONUTF8"] = "1"
    return env


@dataclass
class AppProcess:
    home: Path
    proc: subprocess.Popen[bytes]
    log: LogView
    main_thread: int = 0
    stopped: bool = False

    @property
    def pid(self) -> int:
        return self.proc.pid

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.log.has("app_state", to="READY"):
            if self.proc.poll() is not None:
                raise AssertionError(f"talktype exited with {self.proc.returncode}")
            if time.monotonic() > deadline:
                raise AssertionError("talktype did not reach READY:\n" + "\n".join(
                    self.log.lines()[-30:]
                ))  # fmt: skip
            desktop.pump(100)
        built = self.log.events("app_built")
        self.main_thread = int(built[-1].get("main_thread", "0")) if built else 0

    def stop(self, timeout_s: float = 15.0) -> int | None:
        """Quit through the Qt event loop (WM_QUIT), then kill if it does not exit."""
        if self.stopped:
            return self.proc.returncode
        self.stopped = True
        if self.proc.poll() is None and self.main_thread:
            with contextlib.suppress(OSError):
                win32.PostThreadMessageW(self.main_thread, win32.WM_QUIT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout_s)
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        desktop.release_modifiers()
        return self.proc.returncode


@pytest.fixture
def launch() -> Iterator[Callable[..., AppProcess]]:
    """Start `python -m talktype` for a home; every process is stopped after the test."""
    started: list[AppProcess] = []

    def start(home: Path, wav: Path | None = None, *, ready: bool = True) -> AppProcess:
        home.mkdir(parents=True, exist_ok=True)
        out = (home.parent / f"{home.name}-stdout.txt").open("ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "talktype"],
            cwd=REPO_ROOT,
            env=seam_env(home, wav),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        out.close()
        app = AppProcess(home, proc, LogView(home))
        started.append(app)
        if ready:
            app.wait_ready()
        return app

    yield start
    for app in started:
        app.stop()


# --------------------------------------------------------------------------------------
# In-process mode
# --------------------------------------------------------------------------------------


@dataclass
class InProcess:
    home: Path
    app: Any  # talktype.app.App
    log: LogView

    def action(self, text: str) -> Any:
        found = self.app.tray.find_action(text)
        assert found is not None, f"no tray action {text!r}"
        return found

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        from talktype.app import AppState

        desktop.wait_for(
            lambda: self.app.state in (AppState.READY, AppState.PAUSED),
            timeout_s,
            message="in-process app did not reach READY",
        )


@pytest.fixture
def inprocess(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., InProcess]]:
    """`build_app()` in this process with the e2e seams; shut down after the test."""
    del qtbot  # the QApplication
    built: list[InProcess] = []

    def build(home: Path, wav: Path | None = None, *, ready: bool = True) -> InProcess:
        from talktype.app import build_app
        from talktype.audio import Recorder

        for name, value in seam_env(home, wav).items():
            if name.startswith("TALKTYPE_"):
                monkeypatch.setenv(name, value)
        if wav is None:
            monkeypatch.delenv("TALKTYPE_AUDIO_SOURCE", raising=False)
        seams.current.cache_clear()
        paths = Paths(home).ensure()
        logging_setup.setup_logging(paths.log_file, console=False, level=logging.INFO)
        recorder = Recorder(audio_source=wav) if wav is not None else None
        app = build_app(home, recorder=recorder, show_tray=False)
        handle = InProcess(home, app, LogView(home))
        built.append(handle)
        app.start()
        if ready:
            handle.wait_ready()
        return handle

    yield build
    for handle in built:
        handle.app.shutdown()
    desktop.release_modifiers()
    logging_setup.close_logging()
    seams.current.cache_clear()


# --------------------------------------------------------------------------------------
# Target windows
# --------------------------------------------------------------------------------------


@dataclass
class KeyLog:
    events: list[tuple[str, int]] = field(default_factory=list)

    def pressed(self, key: Qt.Key) -> bool:
        return ("press", int(key)) in self.events

    def released(self, key: Qt.Key) -> bool:
        return ("release", int(key)) in self.events


class _KeySpy(QObject):
    def __init__(self, log: KeyLog) -> None:
        super().__init__()
        self.log = log

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        del watched
        if isinstance(event, QKeyEvent):
            if event.type() == QEvent.Type.KeyPress:
                self.log.events.append(("press", event.key()))
            elif event.type() == QEvent.Type.KeyRelease:
                self.log.events.append(("release", event.key()))
        return False


class TextHarness(QTextEdit):
    """The focused target: a plain `QTextEdit` that records every key event it receives."""

    def __init__(self, title: str = "talktype e2e") -> None:
        super().__init__()
        self.keys = KeyLog()
        self._spy = _KeySpy(self.keys)
        self.installEventFilter(self._spy)
        self.setWindowTitle(title)
        self.resize(640, 360)

    @property
    def hwnd(self) -> int:
        return int(self.winId())

    def open(self) -> TextHarness:
        self.show()
        desktop.focus(self.hwnd)
        self.setFocus()
        desktop.pump(150)
        return self

    def text(self) -> str:
        return self.toPlainText()


class SubmitEdit(TextHarness):
    """A chat-like box: a bare Enter emits `submitted` and inserts nothing."""

    submitted = Signal()

    def __init__(self) -> None:
        super().__init__("talktype e2e submit")
        self.submits = 0
        self.submitted.connect(self._count)

    def _count(self) -> None:
        self.submits += 1

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802 - Qt override
        enter = e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if enter and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        if enter:  # Shift+Enter: a literal line break
            self.insertPlainText("\n")
            return
        super().keyPressEvent(e)


@pytest.fixture
def harness(qtbot: Any) -> Iterator[TextHarness]:
    widget = TextHarness()
    qtbot.addWidget(widget)
    widget.open()
    yield widget
    widget.close()


# --------------------------------------------------------------------------------------
# Trigger gestures
# --------------------------------------------------------------------------------------

RCTRL = win32.VK_RCONTROL


def hold_trigger(log: LogView, hold_ms: int, *, commit_timeout_s: float = 4.0) -> float:
    """Hold Right Ctrl until `dictation_committed`, keep holding `hold_ms`, then release.

    Returns the seconds between key-down and the committed log line.
    """
    start = time.monotonic()
    before = log.count("dictation_committed")
    desktop.down(RCTRL)
    try:
        deadline = start + commit_timeout_s
        while log.count("dictation_committed") == before:
            if time.monotonic() > deadline:
                raise AssertionError("no dictation_committed while holding Right Ctrl")
            desktop.pump(10)
        committed_s = time.monotonic() - start
        desktop.pump(hold_ms)
    finally:
        desktop.up(RCTRL)
    return committed_s


def double_tap(tap_ms: int = 80, gap_ms: int = 150) -> None:
    desktop.tap(RCTRL, tap_ms)
    desktop.pump(gap_ms)
    desktop.tap(RCTRL, tap_ms)


def wait_text(
    widget: TextHarness, predicate: Callable[[str], bool], timeout_s: float = 20.0
) -> str:
    desktop.wait_for(
        lambda: predicate(widget.text()),
        timeout_s,
        message=f"harness text never matched; last: {widget.text()!r}",
    )
    return widget.text()
