"""Real-desktop helpers for tests that drive input: focus, `SendInput` and the clipboard.

These run in the test process. Keys are injected with `SendInput`, so the app's hook only
sees them under the `TALKTYPE_TEST_ACCEPT_INJECTED=1` seam.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from PySide6.QtWidgets import QApplication

from talktype import win32

# Keys that need KEYEVENTF_EXTENDEDKEY so the target sees the right-hand variant.
_EXTENDED = frozenset({win32.VK_RCONTROL, win32.VK_RMENU, win32.VK_INSERT, win32.VK_DELETE})


def pump(ms: int) -> None:
    """Keep the Qt event loop of the test process running for `ms` milliseconds."""
    deadline = time.monotonic() + ms / 1000
    app = QApplication.instance()
    while True:
        if app is not None:
            app.processEvents()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.005))


def wait_for(predicate: Callable[[], bool], timeout_s: float, *, message: str = "") -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message or "condition not met in time")
        pump(20)


def key(vk: int, *, up: bool = False) -> None:
    flags = win32.KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED else 0
    scan = win32.MapVirtualKeyExW(vk, win32.MAPVK_VK_TO_VSC)
    sent = win32.SendInput([win32.key_input(vk, up=up, scan=scan, flags=flags)])
    assert sent == 1


def down(vk: int) -> None:
    key(vk)


def up(vk: int) -> None:
    key(vk, up=True)


def tap(vk: int, hold_ms: int = 80) -> None:
    down(vk)
    pump(hold_ms)
    up(vk)


def chord(*vks: int, gap_ms: int = 30) -> None:
    """Press `vks` in order, then release them in reverse order."""
    for vk in vks:
        down(vk)
        pump(gap_ms)
    for vk in reversed(vks):
        up(vk)
        pump(gap_ms)


def release_modifiers() -> None:
    """Never leave a modifier down after a test."""
    for vk in (
        win32.VK_RCONTROL,
        win32.VK_LCONTROL,
        win32.VK_RMENU,
        win32.VK_LMENU,
        win32.VK_RSHIFT,
        win32.VK_LSHIFT,
    ):
        if win32.GetAsyncKeyState(vk) & 0x8000:
            key(vk, up=True)


def focus(hwnd: int, timeout_s: float = 3.0) -> None:
    """Bring `hwnd` to the foreground, attaching to the current foreground thread first."""
    me = win32.GetCurrentThreadId()
    deadline = time.monotonic() + timeout_s
    while win32.GetForegroundWindow() != hwnd:
        if time.monotonic() > deadline:
            raise AssertionError(f"could not focus window {hwnd:#x}")
        foreground = win32.GetForegroundWindow()
        thread = win32.GetWindowThreadProcessId(foreground)[0] if foreground else 0
        attached = bool(thread) and thread != me and win32.AttachThreadInput(me, thread, True)
        try:
            win32.ShowWindow(hwnd, win32.SW_SHOW)
            win32.BringWindowToTop(hwnd)
            win32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                win32.AttachThreadInput(me, thread, False)
        pump(50)


def focus_desktop() -> int:
    """Give the focus to the desktop (the shell window), which has no text field."""
    shell = win32.GetShellWindow()
    with contextlib.suppress(AssertionError):
        focus(shell, timeout_s=2.0)
    return win32.GetForegroundWindow()


# -- clipboard (immediate rendering, so no process ever waits on this one) --------------


def _open_clipboard() -> None:
    for _ in range(50):
        if win32.OpenClipboard(0):
            return
        time.sleep(0.02)
    raise AssertionError("clipboard stays locked")


def set_clipboard_text(text: str) -> None:
    _open_clipboard()
    try:
        win32.EmptyClipboard()
        win32.SetClipboardData(
            win32.CF_UNICODETEXT, win32.global_from_bytes(win32.unicode_text_bytes(text))
        )
    finally:
        win32.CloseClipboard()


def clipboard_text() -> str | None:
    _open_clipboard()
    try:
        if not win32.IsClipboardFormatAvailable(win32.CF_UNICODETEXT):
            return None
        data = win32.global_to_bytes(win32.GetClipboardData(win32.CF_UNICODETEXT))
        return win32.text_from_unicode_bytes(data)
    finally:
        win32.CloseClipboard()


def clipboard_sequence() -> int:
    return win32.GetClipboardSequenceNumber()
