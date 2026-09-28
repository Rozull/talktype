"""The desktop shell against real Windows: Run key, mutex, overlay window, tray menu.

IT-024 … IT-028. IT-025 and IT-027 start subprocesses; IT-027 needs `QT_SCALE_FACTOR`
set before its `QApplication` exists.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
import tomllib
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from talktype import autostart, win32
from talktype.config import ConfigManager
from talktype.history import HistoryStore
from talktype.overlay import Overlay
from talktype.tray import Tray
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


def _run_python(code: str, *args: str, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------
# IT-024: autostart round trip
# --------------------------------------------------------------------------------------


@pytest.fixture
def run_value_name() -> Iterator[str]:
    name = f"talktype-test-{uuid.uuid4()}"
    try:
        yield name
    finally:
        win32.reg_delete_value(win32.HKEY_CURRENT_USER, win32.RUN_KEY, name)


def test_it024_autostart_round_trip_in_the_real_run_key(run_value_name: str) -> None:
    assert autostart.is_enabled(name=run_value_name) is False

    autostart.enable(name=run_value_name)
    assert autostart.is_enabled(name=run_value_name) is True
    value = win32.reg_read_str(win32.HKEY_CURRENT_USER, win32.RUN_KEY, run_value_name)
    assert value == autostart.command_for(autostart.default_venv())

    autostart.enable(name=run_value_name)  # twice: still one value
    autostart.disable(name=run_value_name)
    assert autostart.is_enabled(name=run_value_name) is False
    assert win32.reg_read_str(win32.HKEY_CURRENT_USER, win32.RUN_KEY, run_value_name) is None


# --------------------------------------------------------------------------------------
# IT-025: single instance across processes
# --------------------------------------------------------------------------------------

MUTEX = "Local\\talktype-singleton-test"

INSTANCE = """
    import sys
    from pathlib import Path
    from talktype import logging_setup, singleinstance

    name, log_file = sys.argv[1], Path(sys.argv[2])
    logging_setup.setup_logging(log_file, console=False)
    guard = singleinstance.acquire(name)
    if guard is None:
        logging_setup.log_event(logging_setup.get_logger("main"), "already_running")
        logging_setup.close_logging()
        print("already_running", flush=True)
        sys.exit(0)
    print("acquired", flush=True)
    sys.stdin.readline()  # hold the mutex until stdin closes or the process is killed
"""


def _instance(log_file: Path) -> subprocess.Popen[str]:
    return _run_python(INSTANCE, MUTEX, str(log_file))


def _first_line(proc: subprocess.Popen[str]) -> str:
    assert proc.stdout is not None
    return proc.stdout.readline().strip()


def _release(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(OSError):
        assert proc.stdin is not None
        proc.stdin.close()
    proc.wait(timeout=20)


def test_it025_single_instance_across_processes(tmp_path: Path) -> None:
    a = _instance(tmp_path / "a.log")
    try:
        assert _first_line(a) == "acquired"

        b = _instance(tmp_path / "b.log")
        assert _first_line(b) == "already_running"
        assert b.wait(timeout=20) == 0
        assert "already_running" in (tmp_path / "b.log").read_text(encoding="utf-8")

        handle = win32.OpenProcess(win32.PROCESS_TERMINATE | win32.SYNCHRONIZE, a.pid)
        try:
            win32.TerminateProcess(handle, 1)
        finally:
            win32.CloseHandle(handle)
        a.wait(timeout=20)

        c = _instance(tmp_path / "c.log")  # a crashed holder never blocks the next launch
        assert _first_line(c) == "acquired"
        _release(c)
    finally:
        if a.poll() is None:
            a.kill()
            a.wait(timeout=20)


def test_it025_simultaneous_launches_leave_one_holder(tmp_path: Path) -> None:
    procs = [_instance(tmp_path / f"p{i}.log") for i in range(3)]
    try:
        lines = [_first_line(p) for p in procs]
        assert sorted(lines) == ["acquired", "already_running", "already_running"]
    finally:
        for proc in procs:
            _release(proc)
    assert [p.returncode for p in procs] == [0, 0, 0]


# --------------------------------------------------------------------------------------
# IT-026: overlay window style, focus and click-through
# --------------------------------------------------------------------------------------


class ClickTarget(QWidget):
    """A plain window placed under the overlay; records the clicks it receives."""

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("background: #ffffff;")
        self.clicks = 0

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.clicks += 1
        event.accept()


def test_it026_overlay_is_non_activating_and_click_through(qtbot: QtBot) -> None:
    overlay = Overlay(foreground=lambda: None)
    target = ClickTarget()
    qtbot.addWidget(overlay)
    qtbot.addWidget(target)

    foreground_before = win32.GetForegroundWindow()
    overlay.show_recording(1)
    qtbot.waitExposed(overlay)
    assert win32.GetForegroundWindow() == foreground_before

    ex_style = win32.GetWindowLongPtrW(overlay.hwnd, win32.GWL_EXSTYLE)
    for flag in (win32.WS_EX_NOACTIVATE, win32.WS_EX_TRANSPARENT, win32.WS_EX_TOPMOST):
        assert ex_style & flag == flag

    target.setGeometry(overlay.frameGeometry().adjusted(-40, -30, 40, 30))
    target.show()
    qtbot.waitExposed(target)
    overlay.show_recording(2)  # raises the overlay back above the target
    qtbot.wait(50)
    assert win32.GetForegroundWindow() == foreground_before

    left, top, right, bottom = win32.GetWindowRect(overlay.hwnd)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    under = win32.WindowFromPoint(cx, cy)
    assert win32.GetAncestor(under) == target.winId()

    cursor = win32.GetCursorPos()
    try:
        win32.SetCursorPos(cx, cy)
        win32.SendInput(
            [
                win32.mouse_input(win32.MOUSEEVENTF_LEFTDOWN),
                win32.mouse_input(win32.MOUSEEVENTF_LEFTUP),
            ]
        )
        qtbot.waitUntil(lambda: target.clicks == 1, timeout=3000)
    finally:
        win32.SetCursorPos(*cursor)


# --------------------------------------------------------------------------------------
# IT-027: placement and legibility at 150% scaling
# --------------------------------------------------------------------------------------

SCALED = """
    import json
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    from talktype.overlay import PREVIEW_WIDTH, Overlay

    overlay = Overlay()
    overlay.show_recording(1)
    app.processEvents()
    words = [f"palavra{i:03d}" for i in range(46)]
    text = " ".join(words)[:500]
    overlay.show_preview(1, 0, text)
    app.processEvents()
    label = overlay._preview_label
    screen = overlay.screen()
    print(json.dumps({
        "ratio": screen.devicePixelRatio(),
        "inside": screen.availableGeometry().contains(overlay.frameGeometry()),
        "text_len": len(text),
        "last_word": text.split()[-1],
        "preview": overlay.preview_text,
        "advance": label.fontMetrics().horizontalAdvance(overlay.preview_text),
        "preview_width": PREVIEW_WIDTH,
        "label_width": label.width(),
        "label_height": label.height(),
        "line_height": label.fontMetrics().height(),
        "overlay_width": overlay.width(),
    }), flush=True)
"""


def test_it027_overlay_fits_and_elides_at_150_percent() -> None:
    env = {**os.environ, "QT_SCALE_FACTOR": "1.5"}
    env.pop("QT_QPA_PLATFORM", None)
    proc = _run_python(SCALED, env=env)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    result = json.loads(out.strip().splitlines()[-1])

    assert result["ratio"] >= 1.5
    assert result["inside"] is True
    assert result["text_len"] == 500
    assert result["preview"].startswith("… ")
    assert result["preview"].endswith(result["last_word"])
    assert result["advance"] <= result["preview_width"] == result["label_width"]
    assert result["label_height"] >= result["line_height"]  # one full line, not clipped


# --------------------------------------------------------------------------------------
# IT-028: tray menu contents, check states and persistence
# --------------------------------------------------------------------------------------

MENU_TEXTS = [
    "Modo: colar",
    "Modo: digitar",
    "Preview ao vivo",
    "Limpeza",
    "Comandos falados",
    "Revisão por IA",
    "Pausar",
    "Iniciar com o Windows",
    "Microfone",
    "Histórico",
    "Abrir config",
    "Recarregar config",
    "Abrir pasta de config",
    "Exportar config",
    "Importar config",
    "Sair",
]


def test_it028_tray_menu_checks_and_persisted_toggle(qtbot: QtBot, home: Path) -> None:
    del qtbot
    config = ConfigManager(home)
    history = HistoryStore(home)
    tray = Tray(
        config,
        history=history.entries,
        devices=lambda: ["Microfone (Realtek)"],
        autostart_enabled=lambda: False,
    )
    tray.set_status("large-v3-turbo", "cuda")
    tray.menu.aboutToShow.emit()

    texts = [a.text() for a in tray.menu.actions()]
    for text in MENU_TEXTS:
        assert text in texts, text
    assert texts[0] == "Modelo: large-v3-turbo · GPU"

    def checked(text: str) -> bool:
        action = tray.find_action(text)
        assert action is not None and action.isCheckable()
        return action.isChecked()

    settings = config.settings
    assert checked("Modo: colar") is (settings.injection.mode == "paste") is True
    assert checked("Modo: digitar") is False
    assert checked("Preview ao vivo") is True  # "auto" on CUDA
    assert checked("Limpeza") is settings.cleanup.enabled is True
    assert checked("Comandos falados") is settings.commands.enabled is True
    assert checked("Revisão por IA") is settings.llm.enabled is False
    assert checked("Iniciar com o Windows") is False

    before = config.path.read_text(encoding="utf-8")
    action = tray.find_action("Limpeza")
    assert action is not None
    action.trigger()

    after = config.path.read_text(encoding="utf-8")
    assert tomllib.loads(after)["cleanup"]["enabled"] is False
    assert config.settings.cleanup.enabled is False
    assert not action.isChecked()
    assert after.count("#") == before.count("#")  # comments kept
    assert ConfigManager(home).settings.cleanup.enabled is False  # survives a restart
