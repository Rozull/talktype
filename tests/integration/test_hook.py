"""The real low-level hooks with injected input (IT-013 … IT-019, IT-039).

`HookThread` runs with ``accept_injected=True`` (the `TALKTYPE_TEST_ACCEPT_INJECTED`
seam) and `SendInput` drives Right Ctrl into focused pytest-qt widgets, so these tests
take over the keyboard focus. Every test leaves the Ctrl keys up. Physical input is let
through untouched (`InjectedOnlyHook`), but a click that moves the focus away while a test
runs can still make it fail.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QTextCursor
from PySide6.QtWidgets import QPushButton, QTextEdit, QWidget
from pytestqt.qtbot import QtBot

from talktype import win32
from talktype.config import TriggerSettings
from talktype.strings import Msg
from talktype.trigger import Effect, TriggerMachine, TriggerState
from talktype.winhook import (
    HookApi,
    HookEvent,
    HookThread,
    HotkeyConflict,
    RepasteRequested,
    SessionInterrupted,
)

pytestmark = pytest.mark.integration

RCTRL = win32.VK_RCONTROL
KEY_A = 0x41
SPARE_HOTKEY = "ctrl+alt+shift+f24"  # keeps the default free for IT-019
RELEASED_AFTER_TESTS = (
    win32.VK_RCONTROL,
    win32.VK_LCONTROL,
    win32.VK_CONTROL,
    win32.VK_LSHIFT,
    win32.VK_SHIFT,
    win32.VK_LMENU,
    win32.VK_MENU,
    win32.VK_ESCAPE,
    KEY_A,
    win32.VK_V,
)


# --------------------------------------------------------------------------------------
# Input and widgets
# --------------------------------------------------------------------------------------


def rctrl(up: bool = False) -> win32.INPUT:
    return win32.key_input(RCTRL, up=up, scan=0x1D, flags=win32.KEYEVENTF_EXTENDEDKEY)


def key(vk: int, up: bool = False) -> win32.INPUT:
    return win32.key_input(vk, up=up, scan=win32.MapVirtualKeyExW(vk, win32.MAPVK_VK_TO_VSC))


class SuiteInputApi:
    """The real `talktype.win32`, except that the last-input clock counts only this suite's
    injected input: someone moving the mouse keeps ``GetLastInputInfo`` fresh, which rightly
    postpones the watchdog (it acts only on input older than 1 s)."""

    last_input = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(win32, name)

    def GetLastInputInfo(self) -> int:  # noqa: N802 (Win32 name)
        return self.last_input


SUITE_API = SuiteInputApi()


def send(*inputs: win32.INPUT) -> None:
    win32.SendInput(list(inputs))
    SUITE_API.last_input = win32.GetTickCount()


def release_everything() -> None:
    held = [vk for vk in RELEASED_AFTER_TESTS if win32.is_key_down(vk)]
    for vk in held:
        flags = win32.KEYEVENTF_EXTENDEDKEY if vk == RCTRL else 0
        win32.SendInput([win32.key_input(vk, up=True, flags=flags)])
    if win32.is_key_down(win32.VK_LBUTTON):
        win32.SendInput([win32.mouse_input(win32.MOUSEEVENTF_LEFTUP)])


class KeyLog(QTextEdit):
    """A text editor that records every key press and release it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[tuple[str, int]] = []
        self.resize(320, 120)

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802 - Qt override
        self.keys.append(("press", e.key()))
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QKeyEvent) -> None:  # noqa: N802 - Qt override
        self.keys.append(("release", e.key()))
        super().keyReleaseEvent(e)


class ClickButton(QPushButton):
    def __init__(self) -> None:
        super().__init__("clique")
        self.presses: list[Qt.KeyboardModifier] = []
        self.resize(200, 80)

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.presses.append(e.modifiers())
        super().mousePressEvent(e)


def focus(qtbot: QtBot, widget: QWidget) -> None:
    """Show `widget` and make it the foreground window with keyboard focus."""
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    hwnd = int(widget.winId())
    for _ in range(10):
        if win32.GetForegroundWindow() == hwnd:
            break
        # A synthesized Alt press lifts the foreground lock for SetForegroundWindow.
        send(key(win32.VK_MENU))
        win32.SetForegroundWindow(hwnd)
        send(key(win32.VK_MENU, up=True))
        qtbot.wait(50)
    widget.activateWindow()
    widget.setFocus()
    qtbot.waitUntil(lambda: win32.GetForegroundWindow() == hwnd and widget.hasFocus())
    qtbot.wait(100)


class Recorder:
    """Thread-safe sink for `on_effects` and `on_event` (called on the hook thread)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._effects: list[Effect] = []
        self._events: list[HookEvent] = []

    def on_effects(self, effects: tuple[Effect, ...]) -> None:
        with self._lock:
            self._effects.extend(effects)

    def on_event(self, event: HookEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def effects(self) -> list[Effect]:
        with self._lock:
            return list(self._effects)

    @property
    def events(self) -> list[HookEvent]:
        with self._lock:
            return list(self._events)


class InjectedOnlyHook(HookThread):
    """A `HookThread` that lets physical input pass untouched.

    Someone using the machine while the suite runs (typing, clicking) must not steer the
    machine: a real click during a hold is correctly a Ctrl+click, which would break the
    assertions. Physical events still stamp the watchdog clock, so they never look like a
    silently removed hook.
    """

    def _on_keyboard(self, n_code: int, wparam: int, lparam: int) -> int:
        if n_code == win32.HC_ACTION:
            info = win32.KBDLLHOOKSTRUCT.from_address(lparam)
            if not info.flags & win32.LLKHF_INJECTED:
                self._last_event_tick = info.time
                return win32.CallNextHookEx(0, n_code, wparam, lparam)
        return super()._on_keyboard(n_code, wparam, lparam)

    def _on_mouse(self, n_code: int, wparam: int, lparam: int) -> int:
        if n_code == win32.HC_ACTION:
            info = win32.MSLLHOOKSTRUCT.from_address(lparam)
            if not info.flags & win32.LLMHF_INJECTED:
                self._last_event_tick = info.time
                return win32.CallNextHookEx(0, n_code, wparam, lparam)
        return super()._on_mouse(n_code, wparam, lparam)


HookFactory = Callable[..., tuple[HookThread, Recorder]]


@pytest.fixture
def start_hook(qtbot: QtBot) -> Iterator[HookFactory]:
    hooks: list[HookThread] = []

    def factory(hotkey: str = SPARE_HOTKEY) -> tuple[HookThread, Recorder]:
        recorder = Recorder()
        hook = InjectedOnlyHook(
            trigger=TriggerSettings(),
            repaste_hotkey=hotkey,
            on_effects=recorder.on_effects,
            on_event=recorder.on_event,
            accept_injected=True,
            api=cast(HookApi, SUITE_API),
        )
        hook.start()
        hooks.append(hook)
        return hook, recorder

    try:
        yield factory
    finally:
        # Let the widgets take their last key-ups first: a release still queued when its
        # window closes leaves Qt believing the key is held, which eats the next press.
        qtbot.wait(150)
        for hook in hooks:
            hook.stop()
        release_everything()
        assert not win32.is_key_down(win32.VK_CONTROL), "a Ctrl key was left down"


@pytest.fixture
def editor(qtbot: QtBot) -> KeyLog:
    widget = KeyLog()
    focus(qtbot, widget)
    widget.keys.clear()
    return widget


# --------------------------------------------------------------------------------------
# IT-013 … IT-017: hold, tap, chord, Esc and Ctrl+click
# --------------------------------------------------------------------------------------


def test_it013_hold_commits_and_finishes_without_reaching_the_app(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory
) -> None:
    hook, recorder = start_hook()
    assert hook.running and all(hook.hooks)

    send(rctrl())
    qtbot.wait(2200)
    send(rctrl(up=True))
    qtbot.waitUntil(lambda: Effect.FINISH in recorder.effects, timeout=2000)

    assert recorder.effects == [Effect.OPEN_MIC, Effect.COMMIT_HOLD, Effect.FINISH]
    qtbot.wait(100)
    assert editor.keys == []


def test_it014_tap_reaches_the_app_as_ctrl_press_and_release(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory
) -> None:
    _, recorder = start_hook()

    send(rctrl())
    qtbot.wait(80)
    send(rctrl(up=True))
    control = int(Qt.Key.Key_Control)
    qtbot.waitUntil(lambda: ("release", control) in editor.keys, timeout=2000)

    assert editor.keys == [("press", control), ("release", control)]
    assert recorder.effects == []


def test_it015_right_ctrl_a_selects_all(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory
) -> None:
    editor.setPlainText("abc")
    editor.moveCursor(QTextCursor.MoveOperation.End)
    start_hook()

    send(rctrl())
    qtbot.wait(40)
    send(key(KEY_A))
    qtbot.wait(40)
    send(key(KEY_A, up=True))
    qtbot.wait(40)
    send(rctrl(up=True))
    qtbot.waitUntil(lambda: editor.textCursor().selectedText() == "abc", timeout=2000)

    assert editor.toPlainText() == "abc"


def test_it016_esc_cancels_and_never_reaches_the_app(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory
) -> None:
    _, recorder = start_hook()

    send(rctrl())
    qtbot.wait(2200)
    send(key(win32.VK_ESCAPE))
    qtbot.wait(40)
    send(key(win32.VK_ESCAPE, up=True))
    qtbot.wait(40)
    send(rctrl(up=True))
    qtbot.waitUntil(lambda: Effect.CANCEL in recorder.effects, timeout=2000)
    qtbot.wait(150)

    assert recorder.effects == [Effect.OPEN_MIC, Effect.COMMIT_HOLD, Effect.CANCEL]
    assert all(k != int(Qt.Key.Key_Escape) for _, k in editor.keys)
    assert editor.keys == []


def test_it017_right_ctrl_click_is_a_ctrl_click(qtbot: QtBot, start_hook: HookFactory) -> None:
    button = ClickButton()
    focus(qtbot, button)
    start_hook()
    left, top, right, bottom = win32.GetWindowRect(int(button.winId()))
    cursor = win32.GetCursorPos()
    cx, cy = (left + right) // 2, (top + bottom) // 2
    try:
        win32.SetCursorPos(cx, cy)
        qtbot.wait(50)
        send(rctrl())
        qtbot.wait(50)
        send(win32.mouse_input(win32.MOUSEEVENTF_LEFTDOWN))
        qtbot.wait(40)
        send(win32.mouse_input(win32.MOUSEEVENTF_LEFTUP))
        qtbot.wait(40)
        send(rctrl(up=True))
        qtbot.waitUntil(lambda: bool(button.presses), timeout=2000)
        # Qt keeps the Ctrl of the click as its cached modifier state until the next mouse
        # event; a move without Ctrl clears it, so the next test's Ctrl press is not eaten.
        qtbot.wait(50)
        win32.SetCursorPos(cx + 1, cy + 1)
        qtbot.wait(50)
    finally:
        win32.SetCursorPos(*cursor)

    assert button.presses[0] & Qt.KeyboardModifier.ControlModifier


# --------------------------------------------------------------------------------------
# IT-018: watchdog recovery
# --------------------------------------------------------------------------------------


def test_it018_watchdog_reinstalls_a_removed_hook(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory, logs: list[str]
) -> None:
    hook, _ = start_hook()
    old_hooks = hook.hooks
    for handle in old_hooks:
        assert win32.UnhookWindowsHookEx(handle)  # a silent removal, as Windows does

    # Keep typing for a while: the input must be more than 1 s newer than the hook's last
    # event, then more than 1 s old at a watchdog run (every 2 s).
    for _ in range(4):
        send(key(win32.VK_LSHIFT))
        send(key(win32.VK_LSHIFT, up=True))
        qtbot.wait(400)

    def reinstalled() -> bool:
        return any(line.startswith("hook_reinstalled gap_ms=") for line in logs)

    qtbot.waitUntil(reinstalled, timeout=5000)
    assert all(hook.hooks)

    editor.keys.clear()
    send(rctrl())
    qtbot.waitUntil(lambda: hook.machine.state is TriggerState.PENDING, timeout=2000)
    send(rctrl(up=True))
    control = int(Qt.Key.Key_Control)
    qtbot.waitUntil(lambda: ("release", control) in editor.keys, timeout=2000)
    assert editor.keys == [("press", control), ("release", control)]  # the replayed tap


# --------------------------------------------------------------------------------------
# IT-019: re-paste hotkey conflict
# --------------------------------------------------------------------------------------


class HotkeyHolder(threading.Thread):
    """Another thread holding Ctrl+Alt+Shift+V through ``RegisterHotKey``."""

    HOTKEY_ID = 0x5A5A

    def __init__(self) -> None:
        super().__init__(name="hotkey-holder", daemon=True)
        self.registered = threading.Event()
        self.release = threading.Event()
        self.ok = False

    def run(self) -> None:
        modifiers = win32.MOD_CONTROL | win32.MOD_ALT | win32.MOD_SHIFT
        try:
            win32.RegisterHotKey(0, self.HOTKEY_ID, modifiers, win32.VK_V)
            self.ok = True
        except OSError:
            self.ok = False
        self.registered.set()
        self.release.wait(30)
        if self.ok:
            win32.UnregisterHotKey(0, self.HOTKEY_ID)


def test_it019_hotkey_conflict_then_repaste(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory
) -> None:
    del editor  # focused, so the modifiers below land in our own window
    holder = HotkeyHolder()
    holder.start()
    holder.registered.wait(5)
    if not holder.ok:
        holder.release.set()
        pytest.skip("Ctrl+Alt+Shift+V is already registered by another program")
    try:
        hook, recorder = start_hook("ctrl+alt+shift+v")
        conflict = HotkeyConflict("ctrl+alt+shift+v")
        assert conflict in recorder.events
        assert conflict.text == Msg.HOTKEY_CONFLICT.format(hotkey="ctrl+alt+shift+v")
        assert hook.hotkey_registered is False
    finally:
        holder.release.set()
        holder.join(5)

    hook.update_settings(TriggerSettings(), "ctrl+alt+shift+v")  # the config reload
    qtbot.waitUntil(lambda: hook.hotkey_registered, timeout=2000)

    modifiers = (win32.VK_CONTROL, win32.VK_MENU, win32.VK_SHIFT)
    send(*(key(vk) for vk in modifiers))
    send(key(win32.VK_V), key(win32.VK_V, up=True))
    send(*(key(vk, up=True) for vk in reversed(modifiers)))
    qtbot.waitUntil(lambda: RepasteRequested() in recorder.events, timeout=2000)


# --------------------------------------------------------------------------------------
# IT-039: session lock
# --------------------------------------------------------------------------------------


def test_it039_session_lock_resets_the_machine(
    qtbot: QtBot, editor: KeyLog, start_hook: HookFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    del editor
    resets: list[str] = []
    original_reset = TriggerMachine.reset

    def spy_reset(machine: TriggerMachine) -> tuple[Effect, ...]:
        resets.append(threading.current_thread().name)
        return original_reset(machine)

    monkeypatch.setattr(TriggerMachine, "reset", spy_reset)
    hook, recorder = start_hook()
    locks: list[str] = []
    original_lock = hook.on_session_lock

    def spy_lock() -> None:
        locks.append(threading.current_thread().name)
        original_lock()

    monkeypatch.setattr(hook, "on_session_lock", spy_lock)

    send(rctrl())
    qtbot.wait(40)
    send(key(KEY_A))  # chord: Right Ctrl down is replayed and stays down
    qtbot.waitUntil(lambda: hook.machine.in_chord, timeout=2000)
    qtbot.waitUntil(lambda: win32.is_key_down(RCTRL), timeout=2000)

    win32.PostMessageW(hook.session_hwnd, win32.WM_WTSSESSION_CHANGE, win32.WTS_SESSION_LOCK, 0)
    qtbot.waitUntil(lambda: SessionInterrupted("lock") in recorder.events, timeout=2000)
    qtbot.waitUntil(lambda: not win32.is_key_down(RCTRL), timeout=2000)

    assert locks == ["talktype-hook"]
    assert resets == ["talktype-hook"]
    assert hook.machine.state is TriggerState.IDLE
    send(key(KEY_A, up=True))
    send(rctrl(up=True))
