"""Clipboard and injection on real Windows (IT-001 … IT-012, IT-044).

These tests take over the clipboard, the keyboard and the window focus: run them under the
desktop lock. Injections run on a real `DeliveryThread` (which pumps Win32 messages for
delayed rendering) while the test thread runs the Qt event loop for the target widgets.
The developer's clipboard is captured before the module and restored afterwards.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QTextEdit, QWidget
from pytestqt.qtbot import QtBot

from talktype import win32
from talktype.clipboard import (
    MARKER_NAMES,
    ClipboardLockedError,
    ClipboardSnapshot,
    open_clipboard,
    read_text,
)
from talktype.config import InjectionSettings, Settings
from talktype.delivery import DeliveryThread
from talktype.foreground import probe_target
from talktype.history import HistoryStore
from talktype.injector import InjectMode, Injector, InjectResult, InjectStatus
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

PASTE = InjectMode.PASTE
TYPE = InjectMode.TYPE


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def keep_developer_clipboard() -> Iterator[None]:
    try:
        saved: ClipboardSnapshot | None = ClipboardSnapshot.capture()
    except ClipboardLockedError:
        saved = None
    yield
    if saved is not None:
        with contextlib.suppress(ClipboardLockedError):
            saved.restore()


def seed(formats: dict[int, bytes]) -> None:
    """Replace the clipboard content (no owner window, like a console app)."""
    assert open_clipboard(0)
    try:
        win32.EmptyClipboard()
        for fmt, data in formats.items():
            assert win32.SetClipboardData(fmt, win32.global_from_bytes(data))
    finally:
        win32.CloseClipboard()


def seed_text(text: str) -> None:
    seed({win32.CF_UNICODETEXT: win32.unicode_text_bytes(text)})


def clipboard_bytes(fmt: int) -> bytes | None:
    assert open_clipboard(0)
    try:
        handle = win32.GetClipboardData(fmt)
        return win32.global_to_bytes(handle) if handle else None
    finally:
        win32.CloseClipboard()


def bring_to_front(hwnd: int) -> None:
    """`SetForegroundWindow`, attached to the current foreground thread's input."""
    foreground = win32.GetForegroundWindow()
    if foreground == hwnd:
        return
    other = win32.GetWindowThreadProcessId(foreground)[0] if foreground else 0
    me = win32.GetCurrentThreadId()
    attached = bool(other) and other != me and win32.AttachThreadInput(me, other, True)
    try:
        win32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            win32.AttachThreadInput(me, other, False)


def focus(qtbot: QtBot, widget: QWidget) -> int:
    """Show `widget` as the foreground window with keyboard focus; returns its HWND.

    The caller must keep a reference to `widget`: `qtbot` only holds a weak one, and a
    collected widget hands the foreground back to the terminal running the tests.
    """
    qtbot.addWidget(widget)
    widget.resize(640, 360)
    widget.show()
    qtbot.waitExposed(widget)
    hwnd = int(widget.winId())
    for _ in range(3):
        bring_to_front(hwnd)
        widget.activateWindow()
        widget.setFocus()
        with contextlib.suppress(Exception):
            qtbot.waitUntil(lambda: win32.GetForegroundWindow() == hwnd, timeout=1500)
            break
    assert win32.GetForegroundWindow() == hwnd
    qtbot.waitUntil(widget.hasFocus, timeout=3000)
    return hwnd


class Runner:
    """Runs injections on a real delivery thread while the test thread pumps Qt."""

    def __init__(self, qtbot: QtBot, home: Path) -> None:
        self.qtbot = qtbot
        self.injector = Injector(overlay_hwnd=0)
        self.delivery = DeliveryThread(
            injector=self.injector, history=HistoryStore(home), on_outcome=lambda _o: None
        )
        self.delivery.start()

    def start(
        self, text: str, mode: InjectMode, **injection: int
    ) -> Callable[[], tuple[InjectResult, float] | None]:
        settings = Settings(injection=InjectionSettings(mode=mode.value, **injection))
        box: list[tuple[InjectResult, float]] = []
        started = time.perf_counter()
        self.delivery.inject_text(
            text, settings, lambda result: box.append((result, time.perf_counter() - started))
        )
        return lambda: box[0] if box else None

    def inject(
        self, text: str, mode: InjectMode, *, own_target: bool = True, **injection: int
    ) -> tuple[InjectResult, float]:
        """Inject and wait for the result.

        With `own_target`, the foreground must still be a window of this process afterwards.
        Otherwise another window (usually the terminal, after a click or key press during the
        run) took the focus and received the keys, and the test fails with that window's class.
        """
        pending = self.start(text, mode, **injection)
        self.qtbot.waitUntil(lambda: pending() is not None, timeout=30000)
        done = pending()
        assert done is not None
        if own_target:
            foreground = win32.GetForegroundWindow()
            _, pid = win32.GetWindowThreadProcessId(foreground) if foreground else (0, 0)
            if pid != os.getpid():
                try:
                    name = win32.GetClassNameW(foreground)
                except OSError:
                    name = "?"
                pytest.fail(f"focus stolen by {name!r} during the injection: keep the desktop idle")
        return done

    def close(self) -> None:
        self.delivery.stop()


@pytest.fixture
def runner(qtbot: QtBot, tmp_path: Path) -> Iterator[Runner]:
    r = Runner(qtbot, tmp_path)
    yield r
    r.close()


def run_helper(code: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def first_line(proc: subprocess.Popen[str]) -> str:
    assert proc.stdout is not None
    return proc.stdout.readline().strip()


class _EnterFilter(QObject):
    """Handles Enter for `SubmitEdit`; every other key goes to Qt untouched."""

    def __init__(self, edit: SubmitEdit) -> None:
        super().__init__(edit)
        self._edit = edit

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = cast(QKeyEvent, event)
        if key.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return False
        if key.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._edit.textCursor().insertBlock()
        else:
            self._edit.submitted.emit()
        return True


class SubmitEdit(QTextEdit):
    """Enter "submits"; Shift+Enter inserts a paragraph (like a chat input).

    Enter is caught by an event filter, not a `keyPressEvent` override: Ctrl+V must reach
    `QTextEdit` without Python code on the stack. Otherwise the paste reads the clipboard
    while holding the GIL, and the delivery thread in this same process cannot run its
    ``WM_RENDERFORMAT`` handler (Windows gives up after about 30 s).
    """

    submitted = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._filter = _EnterFilter(self)
        self.installEventFilter(self._filter)


class DeafWidget(QWidget):
    """Takes the focus but has no text input: Ctrl+V does nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


# --------------------------------------------------------------------------------------
# IT-001: snapshot round trip
# --------------------------------------------------------------------------------------


def _dib_2x2() -> bytes:
    header = struct.pack("<IiiHHIIiiII", 40, 2, 2, 1, 32, 0, 16, 0, 0, 0, 0)
    return header + bytes([0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0])


def _hdrop(path: Path) -> bytes:
    header = struct.pack("<IiiII", 20, 0, 0, 0, 1)  # DROPFILES, wide names
    return header + (str(path) + "\0\0").encode("utf-16-le")


def _html(fragment: str) -> bytes:
    body = f"<html><body><!--StartFragment-->{fragment}<!--EndFragment--></body></html>"
    template = "Version:0.9\r\nStartHTML:{:08d}\r\nEndHTML:{:08d}\r\n"
    template += "StartFragment:{:08d}\r\nEndFragment:{:08d}\r\n"
    size = len(template.format(0, 0, 0, 0).encode("utf-8"))
    raw = body.encode("utf-8")
    start = size + raw.index(b"<!--StartFragment-->") + len(b"<!--StartFragment-->")
    end = size + raw.index(b"<!--EndFragment-->")
    return template.format(size, size + len(raw), start, end).encode("utf-8") + raw + b"\0"


def test_it001_snapshot_round_trip(tmp_path: Path) -> None:
    dropped = tmp_path / "arquivo.txt"
    dropped.write_text("x", encoding="utf-8")
    html = win32.RegisterClipboardFormatW(win32.CFSTR_HTML)
    seeded = {
        win32.CF_UNICODETEXT: win32.unicode_text_bytes("sentinela"),
        html: _html("<b>sentinela</b>"),
        win32.CF_DIB: _dib_2x2(),
        win32.CF_HDROP: _hdrop(dropped),
    }
    seed(seeded)

    snapshot = ClipboardSnapshot.capture()
    seed({})
    assert clipboard_bytes(win32.CF_UNICODETEXT) is None
    snapshot.restore()

    assert snapshot.partial is False
    for fmt, data in seeded.items():
        captured = snapshot.get(fmt)
        restored = clipboard_bytes(fmt)
        assert captured is not None and restored is not None
        assert restored == captured
        assert restored[: len(data)] == data


# --------------------------------------------------------------------------------------
# IT-002 … IT-005, IT-008, IT-009: paste mode
# --------------------------------------------------------------------------------------


def test_it002_paste_is_confirmed_and_clipboard_restored(qtbot: QtBot, runner: Runner) -> None:
    edit = QTextEdit()
    focus(qtbot, edit)
    seed_text("sentinela")
    text = "Olá, mundo 😀\nsegunda linha"

    result, _ = runner.inject(text, PASTE)

    assert result.status is InjectStatus.INSERTED
    assert result.clipboard_restored is True
    qtbot.waitUntil(lambda: edit.toPlainText() == text, timeout=3000)
    qtbot.wait(300)
    assert read_text() == "sentinela"


def test_it002_back_to_back_pastes(qtbot: QtBot, runner: Runner) -> None:
    edit = QTextEdit()
    focus(qtbot, edit)
    seed_text("sentinela")

    first, _ = runner.inject("primeiro ditado. ", PASTE)
    second, _ = runner.inject("segundo ditado.", PASTE)

    assert (first.status, second.status) == (InjectStatus.INSERTED, InjectStatus.INSERTED)
    qtbot.waitUntil(lambda: edit.toPlainText() == "primeiro ditado. segundo ditado.", timeout=3000)
    qtbot.wait(300)
    assert edit.toPlainText() == "primeiro ditado. segundo ditado."
    assert read_text() == "sentinela"


def test_it003_ignored_paste_leaves_text_on_clipboard(qtbot: QtBot, runner: Runner) -> None:
    deaf = DeafWidget()
    focus(qtbot, deaf)
    seed_text("sentinela")

    result, _ = runner.inject("perdido?", PASTE)

    assert result.status is InjectStatus.FAILED_UNCONFIRMED
    assert result.text_on_clipboard is True
    assert read_text() == "perdido?"


COPY_HELPER = """
    import sys
    from talktype.clipboard import put_text

    print("ready", flush=True)
    sys.stdin.readline()
    print("done" if put_text("novo") else "failed", flush=True)
"""


def test_it004_user_copy_during_restore_window_is_kept(qtbot: QtBot, runner: Runner) -> None:
    helper = run_helper(COPY_HELPER)
    try:
        assert first_line(helper) == "ready"

        def signal_helper() -> None:
            assert helper.stdin is not None
            helper.stdin.write("go\n")
            helper.stdin.flush()

        # `textChanged` fires once the paste has read (and so rendered) the clipboard. A
        # signal, not an `insertFromMimeData` override: see `SubmitEdit` for why.
        edit = QTextEdit()
        edit.textChanged.connect(signal_helper)
        focus(qtbot, edit)
        seed_text("sentinela")

        result, _ = runner.inject("ditado", PASTE, restore_delay_ms=600)

        assert first_line(helper) == "done"
    finally:
        helper.kill()
        helper.wait(10)
    assert result.status is InjectStatus.INSERTED
    assert result.clipboard_restored is False
    assert edit.toPlainText() == "ditado"
    assert read_text() == "novo"


LOCK_HELPER = """
    import sys
    import time
    from talktype import win32

    assert win32.OpenClipboard(0)
    print("locked", flush=True)
    time.sleep(float(sys.argv[1]))
    win32.CloseClipboard()
"""


def test_it005_locked_clipboard_fails_fast(qtbot: QtBot, runner: Runner) -> None:
    edit = QTextEdit()
    focus(qtbot, edit)
    helper = run_helper(LOCK_HELPER, "2")
    try:
        assert first_line(helper) == "locked"

        result, elapsed = runner.inject("x", PASTE)
    finally:
        helper.wait(10)

    assert result.status is InjectStatus.FAILED_CLIPBOARD_LOCKED
    assert result.text_on_clipboard is False
    assert elapsed < 0.200


def test_it008_paste_line_breaks_never_submit(qtbot: QtBot, runner: Runner) -> None:
    edit = SubmitEdit()
    submits: list[bool] = []
    edit.submitted.connect(lambda: submits.append(True))
    focus(qtbot, edit)

    result, _ = runner.inject("a\nb", PASTE)

    assert result.status is InjectStatus.INSERTED
    qtbot.waitUntil(lambda: edit.toPlainText() == "a\nb", timeout=3000)
    assert submits == []


def test_it009_exclusion_markers_during_the_paste_window(qtbot: QtBot, runner: Runner) -> None:
    deaf = DeafWidget()
    focus(qtbot, deaf)
    markers = {win32.RegisterClipboardFormatW(name) for name in MARKER_NAMES}
    seen: list[list[int]] = []

    def offered() -> bool:
        if not open_clipboard(0):
            return False
        try:
            owner = win32.GetClipboardOwner()
            formats = win32.clipboard_formats()  # enumerating does not render
        finally:
            win32.CloseClipboard()
        try:
            ours = win32.GetClassNameW(owner).startswith("talktype-clipboard-")
        except OSError:
            ours = False
        if ours and win32.CF_UNICODETEXT in formats:
            seen.append(formats)
            return True
        return False

    pending = runner.start("marcadores", PASTE, paste_confirm_timeout_ms=2000)
    qtbot.waitUntil(offered, timeout=1800)
    qtbot.waitUntil(lambda: pending() is not None, timeout=5000)

    assert markers <= set(seen[0])
    done = pending()
    assert done is not None and done[0].status is InjectStatus.FAILED_UNCONFIRMED


# --------------------------------------------------------------------------------------
# IT-006, IT-007: type mode
# --------------------------------------------------------------------------------------


NATIVE_EDIT_HELPER = """
    import ctypes
    import json
    import sys
    import threading
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT
    ]
    # A top-level multiline EDIT (WS_OVERLAPPEDWINDOW | WS_VISIBLE | ES_MULTILINE).
    hwnd = user32.CreateWindowExW(
        0, "EDIT", None, 0x00CF0000 | 0x10000000 | 0x0004, 100, 100, 600, 200,
        None, None, None, None,
    )

    def serve():
        for line in sys.stdin:
            if line.strip() != "read":
                break
            buf = ctypes.create_unicode_buffer(4096)
            user32.SendMessageW(hwnd, 0x000D, 4096, ctypes.addressof(buf))  # WM_GETTEXT
            print(json.dumps(buf.value), flush=True)
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE

    print(hwnd, flush=True)
    threading.Thread(target=serve, daemon=True).start()
    msg = wintypes.MSG()
    while user32.IsWindow(hwnd) and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
"""


def test_it006_type_mode_is_exact_and_leaves_the_clipboard(qtbot: QtBot, runner: Runner) -> None:
    """The target is a native Win32 EDIT in another process, as in most Windows apps.

    Qt widgets are not used here: Qt's key mapper delivers ``KEYEVENTF_UNICODE`` surrogate
    pairs ahead of the BMP characters sent in the same batch, so a `QTextEdit` receives
    "😀ñ" for "ñ😀" (a Qt quirk, see the workflow memory's "Open Risks").
    """
    helper = run_helper(NATIVE_EDIT_HELPER)
    try:
        hwnd = int(first_line(helper))
        assert _wait_foreground(qtbot, hwnd)
        text = "ação, coração — ñ 😀"
        sequence = win32.GetClipboardSequenceNumber()

        result, _ = runner.inject(text, TYPE, own_target=False)

        assert result.status is InjectStatus.INSERTED
        assert helper.stdin is not None
        typed = ""
        for _ in range(30):
            helper.stdin.write("read\n")
            helper.stdin.flush()
            typed = json.loads(first_line(helper))
            if typed == text:
                break
            qtbot.wait(100)
        assert typed == text
        assert win32.GetClipboardSequenceNumber() == sequence
    finally:
        with contextlib.suppress(OSError):
            helper.communicate("quit\n", timeout=10)
        if helper.poll() is None:
            helper.kill()


def test_it007_type_line_breaks_never_submit(qtbot: QtBot, runner: Runner) -> None:
    edit = SubmitEdit()
    submits: list[bool] = []
    edit.submitted.connect(lambda: submits.append(True))
    focus(qtbot, edit)
    text = "p1\n\np2\np3\np4\np5"

    result, _ = runner.inject(text, TYPE)

    assert result.status is InjectStatus.INSERTED
    qtbot.waitUntil(lambda: edit.toPlainText() == text, timeout=3000)
    qtbot.wait(100)
    assert submits == []
    blocks = edit.toPlainText().split("\n")
    assert [b for b in blocks if b] == ["p1", "p2", "p3", "p4", "p5"]


# --------------------------------------------------------------------------------------
# IT-010, IT-011: probe_target against real windows
# --------------------------------------------------------------------------------------


def _new_window(class_name: str, before: set[int], timeout_s: float = 15.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        fresh = [h for h in win32.find_windows(class_name) if h not in before]
        if fresh:
            return fresh[0]
        time.sleep(0.1)
    return 0


def _wait_foreground(qtbot: QtBot, hwnd: int) -> bool:
    for _ in range(5):
        bring_to_front(hwnd)
        qtbot.wait(200)
        if win32.GetForegroundWindow() == hwnd:
            return True
    return False


def test_it010_probe_real_notepad_and_desktop(qtbot: QtBot) -> None:
    before = set(win32.find_windows("Notepad"))
    proc = subprocess.Popen(["notepad.exe"])
    hwnd = _new_window("Notepad", before)
    try:
        if not hwnd:
            pytest.skip("Notepad opened no new top-level window (tabbed into an existing one)")
        assert _wait_foreground(qtbot, hwnd)
        probe = probe_target()
        assert (probe.status, probe.class_name, probe.hwnd) == ("ok", "Notepad", hwnd)

        desktop = win32.FindWindowExW(0, 0, "Progman")
        assert desktop
        bring_to_front(desktop)
        qtbot.waitUntil(
            lambda: win32.GetClassNameW(win32.GetForegroundWindow()) in win32.SHELL_WINDOW_CLASSES,
            timeout=5000,
        )
        assert probe_target().status == "no_focus"
    finally:
        if hwnd and win32.IsWindow(hwnd):
            win32.PostMessageW(hwnd, win32.WM_CLOSE)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(10)
        if proc.poll() is None:
            proc.kill()


@pytest.mark.skipif(
    os.environ.get("TALKTYPE_TEST_ELEVATED") != "1",
    reason="needs an elevated Notepad without a UAC prompt (TALKTYPE_TEST_ELEVATED=1)",
)
def test_it011_elevated_target(qtbot: QtBot, runner: Runner) -> None:
    before = set(win32.find_windows("Notepad"))
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Start-Process notepad -Verb RunAs"],
        check=True,
        timeout=30,
    )
    hwnd = _new_window("Notepad", before)
    assert hwnd, "the elevated Notepad did not open"
    try:
        assert _wait_foreground(qtbot, hwnd)
        if not win32.current_process_is_elevated():
            assert probe_target().status == "elevated"
            result, _ = runner.inject("x", PASTE, own_target=False)
            assert result.status is InjectStatus.FAILED_ELEVATED

        win32.SendInput([win32.key_input(win32.VK_RCONTROL, flags=win32.KEYEVENTF_EXTENDEDKEY)])
        qtbot.wait(100)
        win32.SendInput(
            [win32.key_input(win32.VK_RCONTROL, up=True, flags=win32.KEYEVENTF_EXTENDEDKEY)]
        )
        qtbot.waitUntil(lambda: not win32.is_key_down(win32.VK_RCONTROL), timeout=2000)
    finally:
        win32.PostMessageW(hwnd, win32.WM_CLOSE)  # may be refused by UIPI; best effort


# --------------------------------------------------------------------------------------
# IT-012: ABNT2 layout
# --------------------------------------------------------------------------------------


def test_it012_abnt2_layout(qtbot: QtBot, runner: Runner) -> None:
    edit = QTextEdit()
    focus(qtbot, edit)
    previous = win32.GetKeyboardLayout(0)
    loaded = set(win32.GetKeyboardLayoutList())
    abnt = win32.LoadKeyboardLayoutW("00000416", win32.KLF_ACTIVATE)
    try:
        assert win32.GetKeyboardLayout(0) == abnt
        typed, _ = runner.inject("não é já", TYPE)
        qtbot.waitUntil(lambda: edit.toPlainText() == "não é já", timeout=3000)
        pasted, _ = runner.inject("ção", PASTE)
        qtbot.waitUntil(lambda: edit.toPlainText() == "não é jáção", timeout=3000)
    finally:
        win32.ActivateKeyboardLayout(previous)
        if abnt not in loaded:
            win32.UnloadKeyboardLayout(abnt)

    assert (typed.status, pasted.status) == (InjectStatus.INSERTED, InjectStatus.INSERTED)
    assert win32.GetKeyboardLayout(0) == previous


# --------------------------------------------------------------------------------------
# IT-044: long texts
# --------------------------------------------------------------------------------------

_PT = (
    "A equipe revisou o relatório de ação às três horas, e o coração do projeto não mudou: "
    "é já a vez da integração contínua, com testes rápidos e análise de código. "
)


def _long_text(size: int) -> str:
    lines: list[str] = []
    while sum(len(line) + 1 for line in lines) < size:
        lines.append(_PT.strip())
    return "\n".join(lines)[:size].rstrip()


@pytest.mark.parametrize(("mode", "size"), [(PASTE, 5000), (TYPE, 3000)])
def test_it044_long_texts_arrive_exactly(
    qtbot: QtBot, runner: Runner, mode: InjectMode, size: int
) -> None:
    edit = QTextEdit()
    focus(qtbot, edit)
    text = _long_text(size)
    assert len(text) >= size - 1

    result, _ = runner.inject(text, mode)

    assert result.status is InjectStatus.INSERTED
    qtbot.waitUntil(lambda: edit.toPlainText() == text, timeout=20000)
