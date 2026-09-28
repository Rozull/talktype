"""`HookThread`: the global keyboard and mouse hooks on a dedicated thread.

The thread runs its own Win32 message loop and owns:
- ``WH_KEYBOARD_LL`` and ``WH_MOUSE_LL``. The callbacks do constant-time work: build a
  `KeyEvent`, ask `TriggerMachine`, queue its effects and return the suppress decision.
- The replay executor. Queued replays go out through ``SendInput`` after the callback has
  returned, from a posted ``WM_APP_REPLAY`` message. Every input HookThread sends carries
  `INJECTED_MARKER` in ``dwExtraInfo`` and is never fed back into the machine. While a
  replay is still queued, later key and button events are held back (swallowed and
  re-emitted right after it), so the focused app always sees them in their original order.
- A 10 ms tick timer that runs only while a press is pending.
- The watchdog: every 2 s, input newer than the hook's last event by more than 1 s (and
  older than 1 s) reinstalls both hooks, resets the machine and logs `hook_reinstalled`.
- The re-paste ``RegisterHotKey``. A conflict is reported and the hook keeps working.
- A hidden message-only window for ``WTS_SESSION_LOCK`` and ``PBT_APMSUSPEND``.

Non-replay effects reach the app through `on_effects`; everything else through `on_event`.
Both are called on the hook thread while it runs (on the caller's thread otherwise), so the
consumer marshals them to Qt itself. Injected input from other sources is ignored
unless the `TALKTYPE_TEST_ACCEPT_INJECTED` seam is set.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from talktype import seams, win32
from talktype.config import TriggerSettings, parse_hotkey
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg
from talktype.trigger import (
    REPLAY_EFFECTS,
    Decision,
    Effect,
    KeyEvent,
    TriggerMachine,
    TriggerTiming,
    trigger_vk,
)

logger = get_logger("winhook")

# dwExtraInfo tag of every input HookThread sends ("tktt"); such events always pass.
INJECTED_MARKER = 0x746B7474
TICK_MS = 10
WATCHDOG_INTERVAL_MS = 2000
WATCHDOG_GAP_MS = 1000
TICK_TIMER_ID = 1
WATCHDOG_TIMER_ID = 2
HOTKEY_ID = 0x7401
WM_APP_CALL = win32.WM_APP + 2
WM_APP_STOP = win32.WM_APP + 3

_EXTENDED_TRIGGERS = frozenset({win32.VK_RCONTROL, win32.VK_RMENU})
_NO_SCAN_TRIGGERS = frozenset({win32.VK_PAUSE})  # its scan code needs the E1 prefix

# Mouse button presses are fed to `on_mouse_down` (a button chord). Releases and
# wheels are never fed; they are only held back behind a queued replay. Each message maps
# to the SendInput flags that re-emit it.
_MOUSE_DOWN = {
    win32.WM_LBUTTONDOWN: win32.MOUSEEVENTF_LEFTDOWN,
    win32.WM_RBUTTONDOWN: win32.MOUSEEVENTF_RIGHTDOWN,
    win32.WM_MBUTTONDOWN: win32.MOUSEEVENTF_MIDDLEDOWN,
    win32.WM_XBUTTONDOWN: win32.MOUSEEVENTF_XDOWN,
}
_MOUSE_UP = {
    win32.WM_LBUTTONUP: win32.MOUSEEVENTF_LEFTUP,
    win32.WM_RBUTTONUP: win32.MOUSEEVENTF_RIGHTUP,
    win32.WM_MBUTTONUP: win32.MOUSEEVENTF_MIDDLEUP,
    win32.WM_XBUTTONUP: win32.MOUSEEVENTF_XUP,
    win32.WM_MOUSEWHEEL: win32.MOUSEEVENTF_WHEEL,
    win32.WM_MOUSEHWHEEL: win32.MOUSEEVENTF_HWHEEL,
}
_MOUSE_DATA_SIGNED = frozenset({win32.WM_MOUSEWHEEL, win32.WM_MOUSEHWHEEL})
_PASS = Decision(False)


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepasteRequested:
    """The re-paste hotkey was pressed (``WM_HOTKEY``)."""


@dataclass(frozen=True, slots=True)
class SessionInterrupted:
    """The session locked or the machine is suspending; any recording is interrupted."""

    reason: Literal["lock", "suspend"]


@dataclass(frozen=True, slots=True)
class HookReinstalled:
    """The watchdog found the hooks silently removed and reinstalled them."""

    gap_ms: int


@dataclass(frozen=True, slots=True)
class HotkeyConflict:
    """``RegisterHotKey`` failed; the hotkey setting is kept and the hook keeps working."""

    hotkey: str

    @property
    def text(self) -> str:
        return Msg.HOTKEY_CONFLICT.format(hotkey=self.hotkey)


HookEvent = RepasteRequested | SessionInterrupted | HookReinstalled | HotkeyConflict


# --------------------------------------------------------------------------------------
# Collaborators
# --------------------------------------------------------------------------------------


class HookApi(Protocol):
    """The `talktype.win32` calls whose effects unit tests fake or observe.

    The window, timer and message-loop plumbing only runs on the real thread and calls
    `talktype.win32` directly.
    """

    def SetWindowsHookExW(  # noqa: N802 (Win32 name)
        self, id_hook: int, proc: Any, hmod: int, thread_id: int, /
    ) -> int: ...

    def UnhookWindowsHookEx(self, hook: int, /) -> bool: ...  # noqa: N802

    def CallNextHookEx(  # noqa: N802
        self, hook: int, code: int, wparam: int, lparam: int, /
    ) -> int: ...

    def GetModuleHandleW(self, name: str | None, /) -> int: ...  # noqa: N802

    def SendInput(self, inputs: Sequence[win32.INPUT], /) -> int: ...  # noqa: N802

    def MapVirtualKeyExW(self, code: int, map_type: int, hkl: int, /) -> int: ...  # noqa: N802

    def is_key_down(self, vk: int, /) -> bool: ...

    def GetLastInputInfo(self) -> int: ...  # noqa: N802

    def GetTickCount(self) -> int: ...  # noqa: N802

    def GetTickCount64(self) -> int: ...  # noqa: N802

    def RegisterHotKey(  # noqa: N802
        self, hwnd: int, hotkey_id: int, modifiers: int, vk: int, /
    ) -> None: ...

    def UnregisterHotKey(self, hwnd: int, hotkey_id: int, /) -> bool: ...  # noqa: N802


def _win32_api() -> HookApi:
    return cast(HookApi, win32)


def tick_delta(a: int, b: int) -> int:
    """``a - b`` on the wrapping 32-bit ``GetTickCount`` clock, as a signed value."""
    return ((a - b + 0x80000000) & 0xFFFFFFFF) - 0x80000000


@dataclass(frozen=True, slots=True)
class _RawKey:
    vk: int
    scan: int
    extended: bool
    up: bool


@dataclass(frozen=True, slots=True)
class _RawMouse:
    flags: int
    data: int


_Raw = _RawKey | _RawMouse


@dataclass(frozen=True, slots=True)
class _Item:
    """One queued batch: machine effects (with the key they refer to) or a held-back event."""

    effects: tuple[Effect, ...]
    vk: int
    raw: _Raw | None


# --------------------------------------------------------------------------------------
# The hook thread
# --------------------------------------------------------------------------------------


class HookThread:
    def __init__(
        self,
        *,
        trigger: TriggerSettings,
        repaste_hotkey: str,
        on_effects: Callable[[tuple[Effect, ...]], None],
        on_event: Callable[[HookEvent], None],
        accept_injected: bool | None = None,
        api: HookApi | None = None,
        watchdog_ms: int = WATCHDOG_INTERVAL_MS,
    ) -> None:
        self._api: HookApi = api or _win32_api()
        self._on_effects = on_effects
        self._on_event = on_event
        self._accept_injected = (
            seams.current().accept_injected if accept_injected is None else accept_injected
        )
        self._watchdog_ms = watchdog_ms
        self._lock = threading.Lock()  # guards the machine and the queue
        self._machine = TriggerMachine(
            TriggerTiming.from_settings(trigger),
            trigger_vk(trigger.key),
            key_is_down=self._api.is_key_down,
        )
        self._hotkey = repaste_hotkey.strip().lower()
        self._hotkey_registered = False
        self._queue: deque[_Item] = deque()
        self._holding = 0  # queued items whose input has not been sent yet
        self._calls: deque[Callable[[], None]] = deque()

        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._started = threading.Event()
        self._start_error: OSError | None = None
        self._class_name = f"talktype-hook-{uuid.uuid4().hex}"
        self._hwnd = 0  # non-zero while the thread runs its loop
        self._kbd_hook = 0
        self._mouse_hook = 0
        self._power_notify = 0
        self._tick_on = False
        self._last_event_tick = 0

        # Windows calls these; they must stay referenced for the object's lifetime.
        self._kbd_cb = win32.HOOKPROC(self._on_keyboard)
        self._mouse_cb = win32.HOOKPROC(self._on_mouse)
        self._wnd_cb = win32.WNDPROC(self._window_proc)

    # -- properties --------------------------------------------------------------------

    @property
    def machine(self) -> TriggerMachine:
        """The machine, for inspection; mutate it only through this class."""
        return self._machine

    @property
    def running(self) -> bool:
        return bool(self._hwnd)

    @property
    def session_hwnd(self) -> int:
        """The hidden message-only window that receives session and power notifications."""
        return self._hwnd

    @property
    def hooks(self) -> tuple[int, int]:
        """The current ``(keyboard, mouse)`` hook handles (0 when not installed)."""
        return self._kbd_hook, self._mouse_hook

    @property
    def repaste_hotkey(self) -> str:
        return self._hotkey

    @property
    def hotkey_registered(self) -> bool:
        return self._hotkey_registered

    @property
    def accept_injected(self) -> bool:
        return self._accept_injected

    # -- lifecycle ---------------------------------------------------------------------

    def start(self, timeout_s: float = 5.0) -> None:
        """Start the thread and block until the hooks are installed.

        Raises ``OSError`` when they cannot be installed (the app shows `Msg.HOOK_BLOCKED`).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._start_error = None
        thread = threading.Thread(target=self._run, name="talktype-hook", daemon=True)
        self._thread = thread
        thread.start()
        if not self._started.wait(timeout_s):
            raise TimeoutError("the hook thread did not start")
        error = self._start_error
        if error is not None:
            thread.join(timeout_s)
            self._thread = None
            log_event(logger, "hook_install_failed", level=logging.ERROR, winerror=error.winerror)
            raise error

    def stop(self, timeout_s: float = 5.0) -> None:
        """Remove the hooks and end the thread. Idempotent; callable from any thread.

        A trigger down replayed for a chord and not yet released gets its key-up (UT-206).
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._thread = None
            self.release_stuck_ctrl()
            return
        if threading.current_thread() is thread:
            self._shutdown()
            return
        self._started.wait(timeout_s)
        hwnd = self._hwnd
        if hwnd:
            try:
                win32.PostMessageW(hwnd, WM_APP_STOP)
            except OSError:
                win32.PostThreadMessageW(self._thread_id, win32.WM_QUIT)
        thread.join(timeout_s)
        self._thread = None
        log_event(logger, "hook_stopped")

    def release_stuck_ctrl(self) -> bool:
        """Send the key-up of a replayed trigger down (atexit, ``aboutToQuit``).

        Other effects of the reset are dropped: the app is quitting. True when sent.
        """
        with self._lock:
            if not self._machine.in_chord:
                return False
            vk = self._machine.trigger_vk
            effects = self._machine.reset()
        inputs = [i for e in effects if e in REPLAY_EFFECTS for i in self._trigger_inputs(e, vk)]
        self._send(inputs)
        log_event(logger, "stuck_ctrl_released", vk=f"0x{vk:02X}")
        return True

    # -- commands (any thread) ---------------------------------------------------------

    def abort(self) -> None:
        """End the current recording silently (limit hit, pause, model not ready)."""
        with self._lock:
            vk = self._machine.trigger_vk
            self._enqueue(self._machine.abort(), vk)
        self._wake()

    def reset(self) -> None:
        """Back to idle, releasing any replayed trigger down (lock, suspend, reinstall)."""
        with self._lock:
            vk = self._machine.trigger_vk
            self._enqueue(self._machine.reset(), vk)
        self._wake()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._machine.set_enabled(enabled)
        self._wake()

    def update_settings(self, trigger: TriggerSettings, repaste_hotkey: str) -> None:
        """Apply new timing and trigger key from the next press; re-register the hotkey
        when it changed or is not registered (a conflict may have been resolved)."""
        with self._lock:
            self._machine.update_timing(
                TriggerTiming.from_settings(trigger), trigger_vk(trigger.key)
            )
        hotkey = repaste_hotkey.strip().lower()
        if hotkey != self._hotkey or not self._hotkey_registered:
            self._hotkey = hotkey
            self._call(self._register_hotkey)
        self._wake()

    # -- hook callbacks (hook thread; constant time) -----------------------------------

    def _on_keyboard(self, n_code: int, wparam: int, lparam: int) -> int:
        api = self._api
        if n_code != win32.HC_ACTION:
            return api.CallNextHookEx(0, n_code, wparam, lparam)
        info = win32.KBDLLHOOKSTRUCT.from_address(lparam)
        self._last_event_tick = info.time
        if info.flags & win32.LLKHF_INJECTED and (
            info.dwExtraInfo == INJECTED_MARKER or not self._accept_injected
        ):
            return api.CallNextHookEx(0, n_code, wparam, lparam)
        try:
            up = bool(info.flags & win32.LLKHF_UP)
            extended = bool(info.flags & win32.LLKHF_EXTENDED)
            raw = _RawKey(info.vkCode, info.scanCode, extended, up)
            event = KeyEvent(info.vkCode, not up, self._unwrap(info.time))
            with self._lock:
                vk = self._machine.trigger_vk
                suppress, wake = self._apply(self._machine.on_key(event), vk, raw)
            if wake:
                self._post_drain()
        except Exception:
            log_event(logger, "hook_error", level=logging.ERROR, exc_info=True, where="keyboard")
            suppress = False
        if suppress:
            return 1
        return api.CallNextHookEx(0, n_code, wparam, lparam)

    def _on_mouse(self, n_code: int, wparam: int, lparam: int) -> int:
        api = self._api
        if n_code != win32.HC_ACTION:
            return api.CallNextHookEx(0, n_code, wparam, lparam)
        info = win32.MSLLHOOKSTRUCT.from_address(lparam)
        self._last_event_tick = info.time
        if wparam == win32.WM_MOUSEMOVE or (
            info.flags & win32.LLMHF_INJECTED
            and (info.dwExtraInfo == INJECTED_MARKER or not self._accept_injected)
        ):
            return api.CallNextHookEx(0, n_code, wparam, lparam)
        try:
            suppress = self._feed_mouse(wparam, info)
        except Exception:
            log_event(logger, "hook_error", level=logging.ERROR, exc_info=True, where="mouse")
            suppress = False
        if suppress:
            return 1
        return api.CallNextHookEx(0, n_code, wparam, lparam)

    def _feed_mouse(self, wparam: int, info: win32.MSLLHOOKSTRUCT) -> bool:
        down = _MOUSE_DOWN.get(wparam)
        flags = down if down is not None else _MOUSE_UP.get(wparam)
        if flags is None:
            return False
        data = (info.mouseData >> 16) & 0xFFFF
        if wparam in _MOUSE_DATA_SIGNED and data >= 0x8000:
            data -= 0x10000
        raw = _RawMouse(flags, data)
        t_ms = self._unwrap(info.time)
        with self._lock:
            if down is None:  # a release or a wheel: only ever held back
                suppress, wake = self._apply(_PASS, 0, raw)
            else:
                vk = self._machine.trigger_vk
                suppress, wake = self._apply(self._machine.on_mouse_down(t_ms), vk, raw)
        if wake:
            self._post_drain()
        return suppress

    def _apply(self, decision: Decision, vk: int, raw: _Raw) -> tuple[bool, bool]:
        """Queue the decision's effects (under the lock).

        Returns ``(swallow the event, drain needed)``; the caller drains after unlocking.
        """
        effects = decision.effects
        if effects:
            swallowed = raw if Effect.REPLAY_SWALLOWED in effects else None
            self._enqueue(effects, vk, swallowed)
            return decision.suppress, True
        if not decision.suppress and self._holding:
            self._enqueue((), 0, raw)  # keep the order behind a queued replay
            return True, False
        return decision.suppress, self._machine.needs_tick != self._tick_on

    def _unwrap(self, t32: int) -> int:
        """A 32-bit hook timestamp on the non-wrapping ``GetTickCount64`` clock."""
        now = self._api.GetTickCount64()
        age = (now - t32) & 0xFFFFFFFF
        return now - (age if age < 0x80000000 else 0)

    # -- queue and replay --------------------------------------------------------------

    def _enqueue(self, effects: tuple[Effect, ...], vk: int, raw: _Raw | None = None) -> None:
        if not effects and raw is None:
            return
        self._queue.append(_Item(effects, vk, raw))
        if raw is not None or any(e in REPLAY_EFFECTS for e in effects):
            self._holding += 1

    def _post_drain(self) -> None:
        """From a hook callback: drain after it returns (right away when not running)."""
        hwnd = self._hwnd
        if not hwnd:
            self._drain()
            return
        try:
            win32.PostMessageW(hwnd, win32.WM_APP_REPLAY)
        except OSError:
            log_event(logger, "hook_post_failed", level=logging.WARNING)

    def _wake(self) -> None:
        """From a command: drain now on the hook thread or when idle, else post."""
        hwnd = self._hwnd
        if hwnd and threading.get_ident() != self._thread_ident():
            try:
                win32.PostMessageW(hwnd, win32.WM_APP_REPLAY)
                return
            except OSError:
                pass
        self._drain()

    def _thread_ident(self) -> int | None:
        thread = self._thread
        return thread.ident if thread is not None else None

    def _drain(self) -> None:
        """Send queued replays in one ``SendInput`` and deliver the other effects, in order."""
        inputs: list[win32.INPUT] = []
        delivered: list[Effect] = []
        with self._lock:
            while self._queue:
                item = self._queue.popleft()
                for effect in item.effects:
                    if effect is Effect.REPLAY_SWALLOWED:
                        if item.raw is not None:
                            inputs.append(self._raw_input(item.raw))
                    elif effect in REPLAY_EFFECTS:
                        inputs.extend(self._trigger_inputs(effect, item.vk))
                    else:
                        delivered.append(effect)
                if not item.effects and item.raw is not None:
                    inputs.append(self._raw_input(item.raw))
            self._holding = 0
            self._sync_tick_timer()
        self._send(inputs)
        if delivered:
            self._deliver(tuple(delivered))

    def _send(self, inputs: list[win32.INPUT]) -> None:
        if not inputs:
            return
        try:
            self._api.SendInput(inputs)
        except OSError as exc:
            log_event(logger, "replay_failed", level=logging.WARNING, winerror=exc.winerror)

    def _trigger_inputs(self, effect: Effect, vk: int) -> list[win32.INPUT]:
        flags = win32.KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED_TRIGGERS else 0
        scan = 0
        if vk not in _NO_SCAN_TRIGGERS:
            scan = self._api.MapVirtualKeyExW(vk, win32.MAPVK_VK_TO_VSC, 0)

        def key(up: bool) -> win32.INPUT:
            return win32.key_input(vk, up=up, scan=scan, flags=flags, extra_info=INJECTED_MARKER)

        if effect is Effect.REPLAY_CTRL_DOWN:
            return [key(False)]
        if effect is Effect.REPLAY_CTRL_UP:
            return [key(True)]
        if effect is Effect.REPLAY_CTRL_TAP:
            return [key(False), key(True)]
        return []

    @staticmethod
    def _raw_input(raw: _Raw) -> win32.INPUT:
        if isinstance(raw, _RawKey):
            flags = win32.KEYEVENTF_EXTENDEDKEY if raw.extended else 0
            return win32.key_input(
                raw.vk, up=raw.up, scan=raw.scan, flags=flags, extra_info=INJECTED_MARKER
            )
        return win32.mouse_input(raw.flags, data=raw.data, extra_info=INJECTED_MARKER)

    def _sync_tick_timer(self) -> None:
        """Run the 10 ms tick only while a press is pending (hook thread, under the lock)."""
        hwnd = self._hwnd
        if not hwnd:
            return
        need = self._machine.needs_tick
        if need and not self._tick_on:
            win32.SetTimer(hwnd, TICK_TIMER_ID, TICK_MS)
            self._tick_on = True
        elif not need and self._tick_on:
            win32.KillTimer(hwnd, TICK_TIMER_ID)
            self._tick_on = False

    def _deliver(self, effects: tuple[Effect, ...]) -> None:
        try:
            self._on_effects(effects)
        except Exception:
            log_event(logger, "hook_consumer_error", level=logging.ERROR, exc_info=True)

    def _emit(self, event: HookEvent) -> None:
        try:
            self._on_event(event)
        except Exception:
            log_event(logger, "hook_consumer_error", level=logging.ERROR, exc_info=True)

    # -- timers, watchdog, hotkey and session (hook thread) ----------------------------

    def _tick(self) -> None:
        with self._lock:
            vk = self._machine.trigger_vk
            self._enqueue(self._machine.on_tick(self._api.GetTickCount64()), vk)
        self._drain()

    def watchdog_check(self) -> bool:
        """Reinstall the hooks when input arrived more than 1 s after their last event."""
        last_input = self._api.GetLastInputInfo()
        now = self._api.GetTickCount()
        gap = tick_delta(last_input, self._last_event_tick)
        if gap > WATCHDOG_GAP_MS and tick_delta(now, last_input) > WATCHDOG_GAP_MS:
            self.reinstall(gap)
            return True
        return False

    def reinstall(self, gap_ms: int) -> None:
        """Replace both hooks, then reset the machine."""
        self._uninstall_hooks()
        try:
            self._install_hooks()
        except OSError as exc:
            log_event(logger, "hook_reinstall_failed", level=logging.ERROR, winerror=exc.winerror)
            return
        self._last_event_tick = self._api.GetTickCount()
        self.reset()
        log_event(logger, "hook_reinstalled", level=logging.WARNING, gap_ms=gap_ms)
        self._emit(HookReinstalled(gap_ms))

    def _install_hooks(self) -> None:
        api = self._api
        hmod = api.GetModuleHandleW(None)
        self._kbd_hook = api.SetWindowsHookExW(win32.WH_KEYBOARD_LL, self._kbd_cb, hmod, 0)
        try:
            self._mouse_hook = api.SetWindowsHookExW(win32.WH_MOUSE_LL, self._mouse_cb, hmod, 0)
        except OSError:
            self._uninstall_hooks()
            raise

    def _uninstall_hooks(self) -> None:
        for hook in (self._kbd_hook, self._mouse_hook):
            if hook:
                self._api.UnhookWindowsHookEx(hook)
        self._kbd_hook = self._mouse_hook = 0

    def _register_hotkey(self) -> None:
        api = self._api
        if self._hotkey_registered:
            api.UnregisterHotKey(self._hwnd, HOTKEY_ID)
            self._hotkey_registered = False
        try:
            modifiers, vk = parse_hotkey(self._hotkey)
            api.RegisterHotKey(self._hwnd, HOTKEY_ID, modifiers | win32.MOD_NOREPEAT, vk)
        except (OSError, ValueError) as exc:
            log_event(
                logger,
                "hotkey_conflict",
                level=logging.WARNING,
                hotkey=self._hotkey,
                winerror=getattr(exc, "winerror", None),
            )
            self._emit(HotkeyConflict(self._hotkey))
            return
        self._hotkey_registered = True
        log_event(logger, "hotkey_registered", hotkey=self._hotkey)

    def _call(self, fn: Callable[[], None]) -> None:
        """Run `fn` on the hook thread (it owns the window); deferred to `start` if idle."""
        hwnd = self._hwnd
        if not hwnd:
            return
        if threading.get_ident() == self._thread_ident():
            fn()
            return
        self._calls.append(fn)
        try:
            win32.PostMessageW(hwnd, WM_APP_CALL)
        except OSError:
            log_event(logger, "hook_post_failed", level=logging.WARNING)

    def on_session_lock(self) -> None:
        """``WTS_SESSION_LOCK``: notify first, then reset (releasing a replayed Ctrl)."""
        self._interrupt("lock")

    def on_suspend(self) -> None:
        """``PBT_APMSUSPEND``: notify first, then reset."""
        self._interrupt("suspend")

    def _interrupt(self, reason: Literal["lock", "suspend"]) -> None:
        log_event(logger, "session_interrupted", reason=reason)
        self._emit(SessionInterrupted(reason))
        self.reset()

    def _window_proc(self, hwnd: int | None, msg: int, wparam: int, lparam: int) -> int:
        try:
            if msg == win32.WM_APP_REPLAY:
                self._drain()
                return 0
            if msg == WM_APP_CALL:
                while self._calls:
                    self._calls.popleft()()
                return 0
            if msg == win32.WM_TIMER:
                if wparam == TICK_TIMER_ID:
                    self._tick()
                elif wparam == WATCHDOG_TIMER_ID:
                    self.watchdog_check()
                return 0
            if msg == win32.WM_HOTKEY and wparam == HOTKEY_ID:
                self._emit(RepasteRequested())
                return 0
            if msg == win32.WM_WTSSESSION_CHANGE:
                if wparam == win32.WTS_SESSION_LOCK:
                    self.on_session_lock()
                return 0
            if msg == win32.WM_POWERBROADCAST:
                if wparam == win32.PBT_APMSUSPEND:
                    self.on_suspend()
                return 1
            if msg == WM_APP_STOP:
                self._shutdown()
                return 0
        except Exception:
            log_event(logger, "hook_error", level=logging.ERROR, exc_info=True, where="window")
            return 0
        return win32.DefWindowProcW(hwnd or 0, msg, wparam, lparam)

    # -- thread body -------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._thread_id = win32.GetCurrentThreadId()
            hwnd = win32.create_message_window(self._class_name, self._wnd_cb)
            self._hwnd = hwnd
            try:
                win32.WTSRegisterSessionNotification(hwnd, win32.NOTIFY_FOR_THIS_SESSION)
            except OSError as exc:
                log_event(
                    logger, "session_notify_failed", level=logging.WARNING, winerror=exc.winerror
                )
            try:
                self._power_notify = win32.RegisterSuspendResumeNotification(hwnd)
            except OSError as exc:
                log_event(
                    logger, "power_notify_failed", level=logging.WARNING, winerror=exc.winerror
                )
            self._install_hooks()
            self._last_event_tick = self._api.GetTickCount()
            win32.SetTimer(hwnd, WATCHDOG_TIMER_ID, self._watchdog_ms)
        except OSError as exc:
            self._start_error = exc
            self._cleanup()
            self._started.set()
            return
        self._register_hotkey()
        log_event(
            logger,
            "hook_installed",
            trigger=f"0x{self._machine.trigger_vk:02X}",
            hotkey=self._hotkey,
            hotkey_registered=self._hotkey_registered,
            accept_injected=self._accept_injected,
        )
        self._started.set()
        msg = win32.MSG()
        try:
            while win32.GetMessageW(msg) > 0:
                win32.TranslateMessage(msg)
                win32.DispatchMessageW(msg)
        except OSError as exc:
            log_event(logger, "hook_loop_failed", level=logging.ERROR, winerror=exc.winerror)
        finally:
            self._cleanup()

    def _shutdown(self) -> None:
        self._cleanup()
        win32.PostQuitMessage(0)

    def _cleanup(self) -> None:
        """Unhook, release a replayed Ctrl, and tear the window down. Idempotent."""
        hwnd = self._hwnd
        if hwnd:
            win32.KillTimer(hwnd, WATCHDOG_TIMER_ID)
            if self._tick_on:
                win32.KillTimer(hwnd, TICK_TIMER_ID)
                self._tick_on = False
        self._uninstall_hooks()  # before any SendInput, so no hook of ours delays it
        with self._lock:
            if self._machine.in_chord:
                vk = self._machine.trigger_vk
                self._enqueue(self._machine.reset(), vk)
                log_event(logger, "stuck_ctrl_released", vk=f"0x{vk:02X}")
        self._hwnd = 0
        self._drain()
        if self._hotkey_registered:
            self._api.UnregisterHotKey(hwnd, HOTKEY_ID)
            self._hotkey_registered = False
        if self._power_notify:
            win32.UnregisterSuspendResumeNotification(self._power_notify)
            self._power_notify = 0
        if hwnd:
            win32.WTSUnRegisterSessionNotification(hwnd)
            win32.destroy_message_window(hwnd, self._class_name)
