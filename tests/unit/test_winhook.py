"""`HookThread` logic with a fake Win32 facade (UT-202, UT-203, UT-205, UT-206).

The thread is never started here: the hook callbacks, the watchdog and the session
handlers are called directly, and without a running loop the queue drains synchronously.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pytest

from talktype import seams, win32, winhook
from talktype.config import TriggerSettings
from talktype.strings import Msg
from talktype.trigger import REPLAY_EFFECTS, Effect, TriggerState
from talktype.winhook import (
    INJECTED_MARKER,
    HookEvent,
    HookReinstalled,
    HookThread,
    HotkeyConflict,
    RepasteRequested,
    SessionInterrupted,
    tick_delta,
)

# These cases are written for a 2000 ms hold; the shipped default is 1000 ms.
CONTRACT_TRIGGER = TriggerSettings(hold_threshold_ms=2000)

pytestmark = pytest.mark.unit

RCTRL = win32.VK_RCONTROL
KEY_A = 0x41
KEY_C = 0x43
EXT = win32.KEYEVENTF_EXTENDEDKEY
KEYUP = win32.KEYEVENTF_KEYUP
RCTRL_SCAN = 0x1D

Sent = tuple[object, ...]


def describe(inp: win32.INPUT) -> Sent:
    if inp.type == win32.INPUT_KEYBOARD:
        ki = inp.ki
        return ("key", ki.wVk, ki.wScan, ki.dwFlags, ki.dwExtraInfo)
    mi = inp.mi
    return ("mouse", mi.dwFlags, mi.mouseData, mi.dwExtraInfo)


def rctrl(up: bool) -> Sent:
    return ("key", RCTRL, RCTRL_SCAN, EXT | (KEYUP if up else 0), INJECTED_MARKER)


class FakeApi:
    """Records hooks, hotkeys and ``SendInput``; clocks are set by the test."""

    def __init__(self) -> None:
        self.sent: list[list[Sent]] = []
        self.installed: list[tuple[int, int]] = []
        self.unhooked: list[int] = []
        self.hotkeys: list[tuple[int, int]] = []
        self.hotkey_error: int | None = None
        self.next_hook_calls = 0
        self.last_input = 0
        self.tick = 0
        self.tick64 = 0
        self.down: set[int] = set()
        self._handle = 0x1000

    def SetWindowsHookExW(self, id_hook: int, proc: object, hmod: int, thread_id: int) -> int:  # noqa: N802
        del proc, hmod, thread_id
        self._handle += 1
        self.installed.append((id_hook, self._handle))
        return self._handle

    def UnhookWindowsHookEx(self, hook: int) -> bool:  # noqa: N802
        self.unhooked.append(hook)
        return True

    def CallNextHookEx(self, hook: int, code: int, wparam: int, lparam: int) -> int:  # noqa: N802
        del hook, code, wparam, lparam
        self.next_hook_calls += 1
        return 0

    def GetModuleHandleW(self, name: str | None) -> int:  # noqa: N802
        del name
        return 0x400000

    def SendInput(self, inputs: Sequence[win32.INPUT]) -> int:  # noqa: N802
        self.sent.append([describe(i) for i in inputs])
        return len(inputs)

    def MapVirtualKeyExW(self, code: int, map_type: int, hkl: int) -> int:  # noqa: N802
        del map_type, hkl
        return RCTRL_SCAN if code == RCTRL else 0

    def is_key_down(self, vk: int) -> bool:
        return vk in self.down

    def GetLastInputInfo(self) -> int:  # noqa: N802
        return self.last_input

    def GetTickCount(self) -> int:  # noqa: N802
        return self.tick

    def GetTickCount64(self) -> int:  # noqa: N802
        return self.tick64

    def RegisterHotKey(self, hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> None:  # noqa: N802
        del hwnd, hotkey_id
        if self.hotkey_error is not None:
            raise OSError(0, "hotkey already registered", None, self.hotkey_error)
        self.hotkeys.append((modifiers, vk))

    def UnregisterHotKey(self, hwnd: int, hotkey_id: int) -> bool:  # noqa: N802
        del hwnd, hotkey_id
        return True


@dataclass
class Harness:
    api: FakeApi
    hook: HookThread
    effects: list[tuple[Effect, ...]] = field(default_factory=list)
    events: list[HookEvent] = field(default_factory=list)
    order: list[object] = field(default_factory=list)

    def key(self, vk: int, *, up: bool = False, t: int = 0, flags: int = 0, extra: int = 0) -> bool:
        """Feed one keyboard event at time `t`; True when the hook swallowed it."""
        self.api.tick64 = t
        info = win32.KBDLLHOOKSTRUCT(
            vkCode=vk,
            scanCode=0x1E,
            flags=flags | (win32.LLKHF_UP if up else 0),
            time=t & 0xFFFFFFFF,
            dwExtraInfo=extra,
        )
        message = win32.WM_KEYUP if up else win32.WM_KEYDOWN
        return self.hook._on_keyboard(win32.HC_ACTION, message, ctypes.addressof(info)) == 1

    def button(self, message: int, *, t: int = 0, data: int = 0, flags: int = 0) -> bool:
        self.api.tick64 = t
        info = win32.MSLLHOOKSTRUCT(mouseData=data, flags=flags, time=t & 0xFFFFFFFF)
        return self.hook._on_mouse(win32.HC_ACTION, message, ctypes.addressof(info)) == 1

    def tick(self, t: int) -> None:
        self.api.tick64 = t
        self.hook._tick()


def build(*, accept_injected: bool = False, trigger: TriggerSettings | None = None) -> Harness:
    api = FakeApi()
    effects: list[tuple[Effect, ...]] = []
    events: list[HookEvent] = []
    order: list[object] = []

    def on_effects(batch: tuple[Effect, ...]) -> None:
        assert batch, "on_effects must never receive an empty tuple"
        assert not set(batch) & REPLAY_EFFECTS, "replays must never reach the app"
        effects.append(batch)
        order.append(batch)

    def on_event(event: HookEvent) -> None:
        events.append(event)
        order.append(event)

    hook = HookThread(
        trigger=trigger or CONTRACT_TRIGGER,
        repaste_hotkey="Ctrl+Alt+Shift+V",
        on_effects=on_effects,
        on_event=on_event,
        accept_injected=accept_injected,
        api=api,
    )
    return Harness(api, hook, effects, events, order)


@pytest.fixture
def h() -> Harness:
    return build()


# --------------------------------------------------------------------------------------
# Replay and delivery
# --------------------------------------------------------------------------------------


def test_hold_delivers_only_app_effects_in_order(h: Harness) -> None:
    assert h.key(RCTRL, t=1000) is True
    h.tick(2000)
    h.tick(2700)
    h.tick(3001)
    assert h.key(RCTRL, t=3001) is True  # auto-repeat
    assert h.key(RCTRL, up=True, t=4000) is True
    assert h.effects == [(Effect.OPEN_MIC,), (Effect.COMMIT_HOLD,), (Effect.FINISH,)]
    assert h.api.sent == []
    assert h.api.next_hook_calls == 0  # every trigger event was swallowed


def test_tap_replays_an_extended_right_ctrl_down_up(h: Harness) -> None:
    h.key(RCTRL, t=0, flags=win32.LLKHF_EXTENDED)
    assert h.key(RCTRL, up=True, t=80, flags=win32.LLKHF_EXTENDED) is True
    assert h.api.sent == [[rctrl(False), rctrl(True)]]
    assert h.effects == []


def test_chord_replays_ctrl_down_then_the_swallowed_key(h: Harness) -> None:
    h.key(RCTRL, t=0)
    assert h.key(KEY_C, t=50) is True
    assert h.api.sent == [[rctrl(False), ("key", KEY_C, 0x1E, 0, INJECTED_MARKER)]]
    assert h.key(KEY_C, up=True, t=80) is False
    assert h.key(RCTRL, up=True, t=120) is False
    assert h.api.next_hook_calls == 2


def test_ctrl_click_replays_the_mouse_button(h: Harness) -> None:
    h.key(RCTRL, t=0)
    assert h.button(win32.WM_LBUTTONDOWN, t=100) is True
    assert h.api.sent == [[rctrl(False), ("mouse", win32.MOUSEEVENTF_LEFTDOWN, 0, INJECTED_MARKER)]]
    assert h.button(win32.WM_LBUTTONUP, t=150) is False
    h.key(RCTRL, up=True, t=200)


def test_wheel_is_not_a_chord_but_keeps_its_place_behind_a_replay(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_down = (-120 & 0xFFFF) << 16
    h.key(RCTRL, t=0)
    assert h.button(win32.WM_MOUSEWHEEL, t=100, data=wheel_down) is False  # plain scroll
    assert h.hook.machine.state is TriggerState.PENDING  # the hold can still commit
    h.key(RCTRL, up=True, t=150)

    monkeypatch.setattr(winhook.win32, "PostMessageW", lambda hwnd, msg, *a: None)
    h.hook._hwnd = 0xBEEF  # pretend the loop runs, so the chord replay stays queued
    h.hook._tick_on = True
    monkeypatch.setattr(winhook.win32, "KillTimer", lambda hwnd, timer_id: True)
    h.api.sent.clear()
    h.key(RCTRL, t=1000)
    h.button(win32.WM_RBUTTONDOWN, t=1010)
    assert h.button(win32.WM_MOUSEWHEEL, t=1011, data=wheel_down) is True  # held back
    h.hook._drain()
    wheel = ("mouse", win32.MOUSEEVENTF_WHEEL, -120 & 0xFFFFFFFF, INJECTED_MARKER)
    assert h.api.sent == [
        [rctrl(False), ("mouse", win32.MOUSEEVENTF_RIGHTDOWN, 0, INJECTED_MARKER), wheel]
    ]


def test_events_behind_a_queued_replay_are_held_back_in_order(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[int] = []
    monkeypatch.setattr(winhook.win32, "PostMessageW", lambda hwnd, msg, *a: posted.append(msg))
    h.hook._hwnd = 0xBEEF  # pretend the loop runs: drains wait for the posted message
    h.hook._tick_on = True  # (the pending press would start the tick timer)
    monkeypatch.setattr(winhook.win32, "KillTimer", lambda hwnd, timer_id: True)

    h.key(RCTRL, t=0)
    assert h.key(KEY_C, t=1) is True  # chord: replay queued, not sent yet
    assert h.key(KEY_C, up=True, t=2) is True  # held back behind it
    assert h.key(RCTRL, up=True, t=3) is True
    assert h.api.sent == []
    assert posted.count(win32.WM_APP_REPLAY) >= 1

    h.hook._drain()
    c_down = ("key", KEY_C, 0x1E, 0, INJECTED_MARKER)
    c_up = ("key", KEY_C, 0x1E, KEYUP, INJECTED_MARKER)
    rctrl_up_raw = ("key", RCTRL, 0x1E, KEYUP, INJECTED_MARKER)
    assert h.api.sent == [[rctrl(False), c_down, c_up, rctrl_up_raw]]
    assert h.key(KEY_A, t=10) is False  # nothing queued any more: events pass again


# --------------------------------------------------------------------------------------
# UT-203: injected-event filter
# --------------------------------------------------------------------------------------


def test_ut203_injected_events_are_dropped_unless_the_seam_is_set() -> None:
    h = build(accept_injected=False)
    assert h.key(RCTRL, t=0, flags=win32.LLKHF_INJECTED) is False
    assert h.hook.machine.state is TriggerState.IDLE
    assert h.button(win32.WM_LBUTTONDOWN, t=10, flags=win32.LLMHF_INJECTED) is False
    assert h.api.next_hook_calls == 2

    h = build(accept_injected=True)
    assert h.key(RCTRL, t=0, flags=win32.LLKHF_INJECTED) is True
    assert h.hook.machine.state is TriggerState.PENDING


def test_ut203_own_replays_are_never_fed_back() -> None:
    h = build(accept_injected=True)
    own = win32.LLKHF_INJECTED
    assert h.key(RCTRL, t=0, flags=own, extra=INJECTED_MARKER) is False
    assert h.hook.machine.state is TriggerState.IDLE
    assert h.button(win32.WM_LBUTTONDOWN, flags=win32.LLMHF_INJECTED, t=5) is False


def test_ut203_accept_injected_defaults_to_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(seams.ACCEPT_INJECTED_ENV, "1")
    seams.current.cache_clear()
    try:
        hook = HookThread(
            trigger=CONTRACT_TRIGGER,
            repaste_hotkey="ctrl+alt+shift+v",
            on_effects=lambda effects: None,
            on_event=lambda event: None,
            api=FakeApi(),
        )
        assert hook.accept_injected is True
    finally:
        monkeypatch.delenv(seams.ACCEPT_INJECTED_ENV)
        seams.current.cache_clear()
    hook = HookThread(
        trigger=CONTRACT_TRIGGER,
        repaste_hotkey="ctrl+alt+shift+v",
        on_effects=lambda effects: None,
        on_event=lambda event: None,
        api=FakeApi(),
    )
    assert hook.accept_injected is False


# --------------------------------------------------------------------------------------
# UT-202: watchdog
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("last_event", "reinstalls"), [(8_500, True), (9_500, False)])
def test_ut202_watchdog_reinstalls_only_after_a_gap(
    h: Harness, monkeypatch: pytest.MonkeyPatch, last_event: int, reinstalls: bool
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(h.hook, "reinstall", calls.append)
    h.key(KEY_A, t=last_event)
    h.key(KEY_A, up=True, t=last_event)
    h.api.last_input = 10_000
    h.api.tick = 11_500

    assert h.hook.watchdog_check() is reinstalls
    assert calls == ([1_500] if reinstalls else [])


def test_watchdog_waits_until_the_input_is_older_than_the_gap(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(h.hook, "reinstall", calls.append)
    h.key(KEY_A, t=8_500)
    h.api.last_input = 10_000
    h.api.tick = 10_900  # the hook may simply not have been called yet
    assert h.hook.watchdog_check() is False
    assert calls == []


def test_watchdog_handles_the_tick_count_wrap(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(h.hook, "reinstall", calls.append)
    h.key(KEY_A, t=0xFFFF_FF00)
    h.api.last_input = 0x0000_0700  # 2048 ms later, after the wrap
    h.api.tick = 0x0000_1000
    assert h.hook.watchdog_check() is True
    assert calls == [0x800]
    assert tick_delta(5, 0xFFFF_FFFB) == 10
    assert tick_delta(0xFFFF_FFFB, 5) == -10


def test_reinstall_replaces_both_hooks_and_resets(h: Harness, logs: list[str]) -> None:
    h.hook._install_hooks()
    old = h.hook.hooks
    h.key(RCTRL, t=0)
    h.tick(1700)
    h.tick(2001)  # committed hold
    h.api.tick = 5_000

    h.hook.reinstall(1_500)

    assert sorted(h.api.unhooked) == sorted(old)
    new = h.hook.hooks
    assert new != old and all(new)
    assert [kind for kind, _ in h.api.installed] == [win32.WH_KEYBOARD_LL, win32.WH_MOUSE_LL] * 2
    assert h.hook.machine.state is TriggerState.IDLE
    assert h.effects[-1] == (Effect.ABORT_SILENT,)
    assert h.events == [HookReinstalled(1_500)]
    assert "hook_reinstalled gap_ms=1500" in logs


# --------------------------------------------------------------------------------------
# UT-205: hotkey conflict
# --------------------------------------------------------------------------------------


def test_ut205_hotkey_conflict_is_reported_and_dictation_still_works(logs: list[str]) -> None:
    h = build()
    h.api.hotkey_error = win32.ERROR_HOTKEY_ALREADY_REGISTERED

    h.hook._register_hotkey()

    assert h.events == [HotkeyConflict("ctrl+alt+shift+v")]
    conflict = h.events[0]
    assert isinstance(conflict, HotkeyConflict)
    assert conflict.text == Msg.HOTKEY_CONFLICT.format(hotkey="ctrl+alt+shift+v")
    assert "ctrl+alt+shift+v" in conflict.text
    assert any(line.startswith("hotkey_conflict hotkey=ctrl+alt+shift+v") for line in logs)
    assert h.hook.repaste_hotkey == "ctrl+alt+shift+v"
    assert h.hook.hotkey_registered is False

    h.key(RCTRL, t=0)
    h.tick(1700)
    h.tick(2001)
    h.key(RCTRL, up=True, t=3000)
    assert h.effects == [(Effect.OPEN_MIC,), (Effect.COMMIT_HOLD,), (Effect.FINISH,)]


def test_hotkey_registers_with_norepeat(h: Harness) -> None:
    h.hook._register_hotkey()
    modifiers = win32.MOD_CONTROL | win32.MOD_ALT | win32.MOD_SHIFT | win32.MOD_NOREPEAT
    assert h.api.hotkeys == [(modifiers, win32.VK_V)]
    assert h.hook.hotkey_registered is True
    assert h.events == []


def test_wm_hotkey_requests_a_repaste(h: Harness) -> None:
    h.hook._window_proc(0, win32.WM_HOTKEY, winhook.HOTKEY_ID, 0)
    assert h.events == [RepasteRequested()]


# --------------------------------------------------------------------------------------
# UT-206: shutdown safety
# --------------------------------------------------------------------------------------


def test_ut206_stop_in_chord_sends_one_ctrl_key_up(h: Harness, logs: list[str]) -> None:
    h.key(RCTRL, t=0)
    h.key(KEY_C, t=50)
    assert h.hook.machine.in_chord is True
    h.api.sent.clear()

    h.hook.stop()

    assert h.api.sent == [[rctrl(True)]]
    assert h.hook.machine.in_chord is False
    assert any(line.startswith("stuck_ctrl_released") for line in logs)
    h.hook.stop()  # idempotent
    assert h.api.sent == [[rctrl(True)]]


def test_release_stuck_ctrl_only_acts_in_chord(h: Harness) -> None:
    assert h.hook.release_stuck_ctrl() is False
    h.key(RCTRL, t=0)
    assert h.hook.release_stuck_ctrl() is False  # pending: the down was never replayed
    h.button(win32.WM_RBUTTONDOWN, t=10)
    h.api.sent.clear()
    assert h.hook.release_stuck_ctrl() is True
    assert h.api.sent == [[rctrl(True)]]


# --------------------------------------------------------------------------------------
# Session, commands and settings
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "wparam", "reason"),
    [
        (win32.WM_WTSSESSION_CHANGE, win32.WTS_SESSION_LOCK, "lock"),
        (win32.WM_POWERBROADCAST, win32.PBT_APMSUSPEND, "suspend"),
    ],
)
def test_session_interruption_notifies_then_resets(
    h: Harness, message: int, wparam: int, reason: Literal["lock", "suspend"]
) -> None:
    h.key(RCTRL, t=0)
    h.tick(1700)
    h.tick(2001)
    h.order.clear()

    h.hook._window_proc(0, message, wparam, 0)

    assert h.order == [SessionInterrupted(reason), (Effect.ABORT_SILENT,)]
    assert h.hook.machine.state is TriggerState.IDLE


def test_session_unlock_is_ignored(h: Harness) -> None:
    h.hook._window_proc(0, win32.WM_WTSSESSION_CHANGE, win32.WTS_SESSION_UNLOCK, 0)
    assert h.events == []


def test_commands_from_another_thread(h: Harness) -> None:
    h.key(RCTRL, t=0)
    h.button(win32.WM_LBUTTONDOWN, t=10)  # chord
    h.api.sent.clear()

    worker = threading.Thread(target=h.hook.reset)
    worker.start()
    worker.join()

    assert h.api.sent == [[rctrl(True)]]
    assert h.hook.machine.state is TriggerState.IDLE

    h.key(RCTRL, t=1000)
    h.tick(3001)
    h.hook.abort()
    assert h.effects == [(Effect.OPEN_MIC, Effect.COMMIT_HOLD)]
    assert h.key(RCTRL, up=True, t=3500) is True
    assert h.effects == [(Effect.OPEN_MIC, Effect.COMMIT_HOLD)]  # no FINISH after abort

    h.hook.set_enabled(False)
    assert h.key(RCTRL, t=5000) is False
    h.hook.set_enabled(True)


def test_update_settings_applies_timing_and_key_from_the_next_press(h: Harness) -> None:
    h.key(RCTRL, t=0)
    h.hook.update_settings(
        TriggerSettings(key="f13", hold_threshold_ms=500, mic_preopen_ms=0), "ctrl+alt+b"
    )
    h.tick(1000)
    assert h.effects == []  # the current press keeps the old timing
    h.key(RCTRL, up=True, t=1100)
    assert h.hook.repaste_hotkey == "ctrl+alt+b"

    assert h.key(RCTRL, t=2000) is False
    h.key(RCTRL, up=True, t=2050)
    assert h.key(0x7C, t=3000) is True
    h.tick(3501)
    assert h.effects == [(Effect.OPEN_MIC, Effect.COMMIT_HOLD)]


@pytest.fixture(autouse=True)
def _no_real_input(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Nothing in this module may reach the real SendInput."""

    def refuse(inputs: Sequence[win32.INPUT]) -> int:
        raise AssertionError("real SendInput called from a unit test")

    monkeypatch.setattr(win32, "SendInput", refuse)
    yield
