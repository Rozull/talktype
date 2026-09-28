"""`probe_target()`: may text be inserted into the foreground window?

- No foreground window, the desktop shell (``Progman``, ``WorkerW``, ``Shell_TrayWnd``) or our
  own overlay means ``no_focus``.
- A target running elevated while talktype is not, or one whose token cannot be queried
  (access denied), means ``elevated``: `SendInput` into it fails silently (UIPI).
- Anything else is ``ok``.

`api` is the `win32` module, or a fake with the same functions (tests).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Literal

from talktype import win32

ProbeStatus = Literal["ok", "no_focus", "elevated"]


@dataclass(frozen=True, slots=True)
class Probe:
    status: ProbeStatus
    hwnd: int = 0
    class_name: str = ""
    pid: int = 0


def _target_elevated(pid: int, api: Any) -> bool | None:
    """``TokenElevation`` of process `pid`; None when it cannot be queried."""
    try:
        process = api.OpenProcess(win32.PROCESS_QUERY_LIMITED_INFORMATION, pid)
    except OSError:
        return None
    try:
        token = api.OpenProcessToken(process)
    except OSError:
        return None
    finally:
        api.CloseHandle(process)
    try:
        return bool(api.token_is_elevated(token))
    except OSError:
        return None
    finally:
        api.CloseHandle(token)


def _self_elevated(api: Any) -> bool:
    with contextlib.suppress(OSError):
        return bool(api.current_process_is_elevated())
    return False


def probe_target(overlay_hwnd: int = 0, *, api: Any = win32) -> Probe:
    """Classify the current foreground window as an insertion target."""
    hwnd = int(api.GetForegroundWindow())
    if not hwnd or (overlay_hwnd and hwnd == overlay_hwnd):
        return Probe("no_focus", hwnd)
    try:
        class_name = str(api.GetClassNameW(hwnd))
    except OSError:  # the window is already gone
        return Probe("no_focus", hwnd)
    if class_name in win32.SHELL_WINDOW_CLASSES:
        return Probe("no_focus", hwnd, class_name)
    _, pid = api.GetWindowThreadProcessId(hwnd)
    if pid and pid != api.GetCurrentProcessId() and not _self_elevated(api):
        elevated = _target_elevated(pid, api)
        if elevated is None or elevated:
            return Probe("elevated", hwnd, class_name, pid)
    return Probe("ok", hwnd, class_name, pid)
