"""Typed ctypes bindings for every Win32 API talktype uses.

This is the only Win32 layer in the package: no pywin32, no parallel binding modules.

Conventions:
- Each public function keeps its Win32 name and has typed parameters and results.
- Handles are plain ``int`` values, and ``0`` means NULL.
- Functions whose failure is always an error raise ``OSError(winerror)`` built from
  ``GetLastError``. Functions whose "failure" is a normal outcome (for example
  ``GetForegroundWindow`` returning NULL) return the raw value.
- Callbacks (``HOOKPROC``, ``WNDPROC``, ``TIMERPROC``) must be kept referenced by the caller
  for as long as Windows may call them.
"""

from __future__ import annotations

import ctypes
import winreg
from collections.abc import Callable, Sequence
from ctypes import wintypes
from typing import Any, NamedTuple

# --------------------------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------------------------

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
_wtsapi32 = ctypes.WinDLL("wtsapi32", use_last_error=True)
_shell32 = ctypes.WinDLL("shell32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_winmm = ctypes.WinDLL("winmm", use_last_error=True)

# --------------------------------------------------------------------------------------
# Scalar types
# --------------------------------------------------------------------------------------

LRESULT = ctypes.c_ssize_t
LONG_PTR = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t
WPARAM = wintypes.WPARAM
LPARAM = wintypes.LPARAM
HANDLE = wintypes.HANDLE
HWND = wintypes.HWND
HHOOK = HANDLE
HGLOBAL = HANDLE
HKL = HANDLE
HENHMETAFILE = HANDLE

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Hooks
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
HC_ACTION = 0
LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_UP = 0x80
LLMHF_INJECTED = 0x01

# Window messages
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_TIMER = 0x0113
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
WM_MOUSEHWHEEL = 0x020E
WM_HOTKEY = 0x0312
WM_RENDERFORMAT = 0x0305
WM_RENDERALLFORMATS = 0x0306
WM_DESTROYCLIPBOARD = 0x0307
WM_POWERBROADCAST = 0x0218
WM_WTSSESSION_CHANGE = 0x02B1
WM_APP = 0x8000
WM_APP_REPLAY = WM_APP + 1

# Power broadcast and session change codes
PBT_APMSUSPEND = 0x0004
NOTIFY_FOR_THIS_SESSION = 0
WTS_SESSION_LOCK = 0x7
WTS_SESSION_UNLOCK = 0x8
DEVICE_NOTIFY_WINDOW_HANDLE = 0

# PeekMessage / MsgWaitForMultipleObjects
PM_NOREMOVE = 0x0000
PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF

# SendInput
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000

# MapVirtualKey
MAPVK_VK_TO_VSC = 0

# Keyboard layouts
KLF_ACTIVATE = 0x00000001

# Virtual keys
VK_LBUTTON = 0x01
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_PAUSE = 0x13
VK_ESCAPE = 0x1B
VK_INSERT = 0x2D
VK_DELETE = 0x2E
VK_V = 0x56
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F1 = 0x70
VK_F13 = 0x7C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5


def VK_F(n: int) -> int:
    """Virtual key of function key F1..F24."""
    if not 1 <= n <= 24:
        raise ValueError(f"no such function key: F{n}")
    return VK_F1 + n - 1


# RegisterHotKey modifiers
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

# Window styles and positioning
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
HWND_MESSAGE = -3
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
GA_ROOT = 2
SW_SHOW = 5

# Clipboard formats
CF_BITMAP = 2
CF_METAFILEPICT = 3
CF_DIB = 8
CF_PALETTE = 9
CF_UNICODETEXT = 13
CF_ENHMETAFILE = 14
CF_HDROP = 15
CF_OWNERDISPLAY = 0x0080
CF_DSPBITMAP = 0x0082
CF_DSPMETAFILEPICT = 0x0083
CF_DSPENHMETAFILE = 0x008E
CF_PRIVATEFIRST = 0x0200
CF_PRIVATELAST = 0x02FF
CF_GDIOBJFIRST = 0x0300
CF_GDIOBJLAST = 0x03FF
CFSTR_HTML = "HTML Format"
CFSTR_EXCLUDE_MONITOR = "ExcludeClipboardContentFromMonitorProcessing"
CFSTR_CAN_INCLUDE_IN_HISTORY = "CanIncludeInClipboardHistory"
CFSTR_CAN_UPLOAD_TO_CLOUD = "CanUploadToCloudClipboard"

# Global memory
GMEM_MOVEABLE = 0x0002

# Processes and tokens
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_CLASS = 20  # TOKEN_INFORMATION_CLASS.TokenElevation

# Errors
ERROR_ACCESS_DENIED = 5
ERROR_INVALID_HANDLE = 6
ERROR_INVALID_WINDOW_HANDLE = 1400
ERROR_ALREADY_EXISTS = 183
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
ERROR_CLIPBOARD_NOT_OPEN = 1418

# MessageBox
MB_OK = 0x00000000
MB_ICONERROR = 0x00000010
MB_ICONINFORMATION = 0x00000040
MB_SETFOREGROUND = 0x00010000
MB_TOPMOST = 0x00040000

# PlaySound
SND_ASYNC = 0x0001
SND_NODEFAULT = 0x0002
SND_MEMORY = 0x0004

# Registry roots and well-known keys
HKEY_CURRENT_USER = winreg.HKEY_CURRENT_USER
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
MIC_CONSENT_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
)
MIC_CONSENT_NONPACKAGED_KEY = MIC_CONSENT_KEY + r"\NonPackaged"

# Desktop shell window classes that mean "no focused target"
SHELL_WINDOW_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd"})

# --------------------------------------------------------------------------------------
# Structures and callback types
# --------------------------------------------------------------------------------------

POINT = wintypes.POINT
RECT = wintypes.RECT
MSG = wintypes.MSG


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD))


class TOKEN_ELEVATION(ctypes.Structure):
    _fields_ = (("TokenIsElevated", wintypes.DWORD),)


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, wintypes.UINT, WPARAM, LPARAM)
TIMERPROC = ctypes.WINFUNCTYPE(None, HWND, wintypes.UINT, ULONG_PTR, wintypes.DWORD)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    )


# --------------------------------------------------------------------------------------
# Raw prototypes
# --------------------------------------------------------------------------------------


def _proto(dll: ctypes.WinDLL, name: str, restype: Any, *argtypes: Any) -> Any:
    fn = getattr(dll, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_P = ctypes.POINTER
_BOOL = wintypes.BOOL
_UINT = wintypes.UINT
_DWORD = wintypes.DWORD

# user32: hooks, messages, timers, input
_SetWindowsHookExW = _proto(
    _user32, "SetWindowsHookExW", HHOOK, ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, _DWORD
)
_UnhookWindowsHookEx = _proto(_user32, "UnhookWindowsHookEx", _BOOL, HHOOK)
_CallNextHookEx = _proto(_user32, "CallNextHookEx", LRESULT, HHOOK, ctypes.c_int, WPARAM, LPARAM)
_GetMessageW = _proto(_user32, "GetMessageW", _BOOL, _P(MSG), HWND, _UINT, _UINT)
_PeekMessageW = _proto(_user32, "PeekMessageW", _BOOL, _P(MSG), HWND, _UINT, _UINT, _UINT)
_TranslateMessage = _proto(_user32, "TranslateMessage", _BOOL, _P(MSG))
_DispatchMessageW = _proto(_user32, "DispatchMessageW", LRESULT, _P(MSG))
_PostThreadMessageW = _proto(_user32, "PostThreadMessageW", _BOOL, _DWORD, _UINT, WPARAM, LPARAM)
_PostMessageW = _proto(_user32, "PostMessageW", _BOOL, HWND, _UINT, WPARAM, LPARAM)
_PostQuitMessage = _proto(_user32, "PostQuitMessage", None, ctypes.c_int)
_MsgWaitForMultipleObjects = _proto(
    _user32, "MsgWaitForMultipleObjects", _DWORD, _DWORD, _P(HANDLE), _BOOL, _DWORD, _DWORD
)
_SetTimer = _proto(_user32, "SetTimer", ULONG_PTR, HWND, ULONG_PTR, _UINT, TIMERPROC)
_KillTimer = _proto(_user32, "KillTimer", _BOOL, HWND, ULONG_PTR)
_SendInput = _proto(_user32, "SendInput", _UINT, _UINT, _P(INPUT), ctypes.c_int)
_GetAsyncKeyState = _proto(_user32, "GetAsyncKeyState", wintypes.SHORT, ctypes.c_int)
_MapVirtualKeyExW = _proto(_user32, "MapVirtualKeyExW", _UINT, _UINT, _UINT, HKL)
_GetKeyboardLayout = _proto(_user32, "GetKeyboardLayout", HKL, _DWORD)
_LoadKeyboardLayoutW = _proto(_user32, "LoadKeyboardLayoutW", HKL, wintypes.LPCWSTR, _UINT)
_ActivateKeyboardLayout = _proto(_user32, "ActivateKeyboardLayout", HKL, HKL, _UINT)
_UnloadKeyboardLayout = _proto(_user32, "UnloadKeyboardLayout", _BOOL, HKL)
_GetKeyboardLayoutList = _proto(
    _user32, "GetKeyboardLayoutList", ctypes.c_int, ctypes.c_int, _P(HKL)
)
_AttachThreadInput = _proto(_user32, "AttachThreadInput", _BOOL, _DWORD, _DWORD, _BOOL)
_GetLastInputInfo = _proto(_user32, "GetLastInputInfo", _BOOL, _P(LASTINPUTINFO))
_RegisterHotKey = _proto(_user32, "RegisterHotKey", _BOOL, HWND, ctypes.c_int, _UINT, _UINT)
_UnregisterHotKey = _proto(_user32, "UnregisterHotKey", _BOOL, HWND, ctypes.c_int)
_RegisterSuspendResumeNotification = _proto(
    _user32, "RegisterSuspendResumeNotification", HANDLE, HANDLE, _DWORD
)
_UnregisterSuspendResumeNotification = _proto(
    _user32, "UnregisterSuspendResumeNotification", _BOOL, HANDLE
)

# user32: windows
_RegisterClassExW = _proto(_user32, "RegisterClassExW", wintypes.ATOM, _P(WNDCLASSEXW))
_UnregisterClassW = _proto(_user32, "UnregisterClassW", _BOOL, wintypes.LPCWSTR, wintypes.HINSTANCE)
_CreateWindowExW = _proto(
    _user32,
    "CreateWindowExW",
    HWND,
    _DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    _DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
)
_DestroyWindow = _proto(_user32, "DestroyWindow", _BOOL, HWND)
_DefWindowProcW = _proto(_user32, "DefWindowProcW", LRESULT, HWND, _UINT, WPARAM, LPARAM)
_IsWindow = _proto(_user32, "IsWindow", _BOOL, HWND)
_GetForegroundWindow = _proto(_user32, "GetForegroundWindow", HWND)
_SetForegroundWindow = _proto(_user32, "SetForegroundWindow", _BOOL, HWND)
_BringWindowToTop = _proto(_user32, "BringWindowToTop", _BOOL, HWND)
_ShowWindow = _proto(_user32, "ShowWindow", _BOOL, HWND, ctypes.c_int)
_GetShellWindow = _proto(_user32, "GetShellWindow", HWND)
_FindWindowExW = _proto(
    _user32, "FindWindowExW", HWND, HWND, HWND, wintypes.LPCWSTR, wintypes.LPCWSTR
)
_WindowFromPoint = _proto(_user32, "WindowFromPoint", HWND, POINT)
_GetAncestor = _proto(_user32, "GetAncestor", HWND, HWND, _UINT)
_GetCursorPos = _proto(_user32, "GetCursorPos", _BOOL, _P(POINT))
_SetCursorPos = _proto(_user32, "SetCursorPos", _BOOL, ctypes.c_int, ctypes.c_int)
_GetClassNameW = _proto(_user32, "GetClassNameW", ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_GetWindowThreadProcessId = _proto(_user32, "GetWindowThreadProcessId", _DWORD, HWND, _P(_DWORD))
_GetWindowLongPtrW = _proto(_user32, "GetWindowLongPtrW", LONG_PTR, HWND, ctypes.c_int)
_SetWindowLongPtrW = _proto(_user32, "SetWindowLongPtrW", LONG_PTR, HWND, ctypes.c_int, LONG_PTR)
_SetWindowPos = _proto(
    _user32,
    "SetWindowPos",
    _BOOL,
    HWND,
    HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    _UINT,
)
_GetWindowRect = _proto(_user32, "GetWindowRect", _BOOL, HWND, _P(RECT))
_MessageBoxW = _proto(
    _user32, "MessageBoxW", ctypes.c_int, HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, _UINT
)

# user32: clipboard
_OpenClipboard = _proto(_user32, "OpenClipboard", _BOOL, HWND)
_CloseClipboard = _proto(_user32, "CloseClipboard", _BOOL)
_EmptyClipboard = _proto(_user32, "EmptyClipboard", _BOOL)
_GetClipboardData = _proto(_user32, "GetClipboardData", HANDLE, _UINT)
_SetClipboardData = _proto(_user32, "SetClipboardData", HANDLE, _UINT, HANDLE)
_EnumClipboardFormats = _proto(_user32, "EnumClipboardFormats", _UINT, _UINT)
_IsClipboardFormatAvailable = _proto(_user32, "IsClipboardFormatAvailable", _BOOL, _UINT)
_RegisterClipboardFormatW = _proto(_user32, "RegisterClipboardFormatW", _UINT, wintypes.LPCWSTR)
_GetClipboardFormatNameW = _proto(
    _user32, "GetClipboardFormatNameW", ctypes.c_int, _UINT, wintypes.LPWSTR, ctypes.c_int
)
_GetClipboardOwner = _proto(_user32, "GetClipboardOwner", HWND)
_GetOpenClipboardWindow = _proto(_user32, "GetOpenClipboardWindow", HWND)
_GetClipboardSequenceNumber = _proto(_user32, "GetClipboardSequenceNumber", _DWORD)

# kernel32
_GetCurrentThreadId = _proto(_kernel32, "GetCurrentThreadId", _DWORD)
_GetCurrentProcessId = _proto(_kernel32, "GetCurrentProcessId", _DWORD)
_GetCurrentProcess = _proto(_kernel32, "GetCurrentProcess", HANDLE)
_GetTickCount = _proto(_kernel32, "GetTickCount", _DWORD)
_GetTickCount64 = _proto(_kernel32, "GetTickCount64", ctypes.c_ulonglong)
_GetModuleHandleW = _proto(_kernel32, "GetModuleHandleW", wintypes.HMODULE, wintypes.LPCWSTR)
_OpenProcess = _proto(_kernel32, "OpenProcess", HANDLE, _DWORD, _BOOL, _DWORD)
_TerminateProcess = _proto(_kernel32, "TerminateProcess", _BOOL, HANDLE, _UINT)
_CloseHandle = _proto(_kernel32, "CloseHandle", _BOOL, HANDLE)
_CreateMutexW = _proto(_kernel32, "CreateMutexW", HANDLE, wintypes.LPVOID, _BOOL, wintypes.LPCWSTR)
_GlobalAlloc = _proto(_kernel32, "GlobalAlloc", HGLOBAL, _UINT, ctypes.c_size_t)
_GlobalLock = _proto(_kernel32, "GlobalLock", wintypes.LPVOID, HGLOBAL)
_GlobalUnlock = _proto(_kernel32, "GlobalUnlock", _BOOL, HGLOBAL)
_GlobalSize = _proto(_kernel32, "GlobalSize", ctypes.c_size_t, HGLOBAL)
_GlobalFree = _proto(_kernel32, "GlobalFree", HGLOBAL, HGLOBAL)

# advapi32
_OpenProcessToken = _proto(_advapi32, "OpenProcessToken", _BOOL, HANDLE, _DWORD, _P(HANDLE))
_GetTokenInformation = _proto(
    _advapi32,
    "GetTokenInformation",
    _BOOL,
    HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    _DWORD,
    _P(_DWORD),
)

# wtsapi32
_WTSRegisterSessionNotification = _proto(
    _wtsapi32, "WTSRegisterSessionNotification", _BOOL, HWND, _DWORD
)
_WTSUnRegisterSessionNotification = _proto(
    _wtsapi32, "WTSUnRegisterSessionNotification", _BOOL, HWND
)

# shell32
_IsUserAnAdmin = _proto(_shell32, "IsUserAnAdmin", _BOOL)

# gdi32
_GetEnhMetaFileBits = _proto(
    _gdi32, "GetEnhMetaFileBits", _UINT, HENHMETAFILE, _UINT, wintypes.LPVOID
)
_SetEnhMetaFileBits = _proto(_gdi32, "SetEnhMetaFileBits", HENHMETAFILE, _UINT, ctypes.c_char_p)
_DeleteEnhMetaFile = _proto(_gdi32, "DeleteEnhMetaFile", _BOOL, HENHMETAFILE)

# winmm
_PlaySoundW = _proto(_winmm, "PlaySoundW", _BOOL, ctypes.c_void_p, wintypes.HMODULE, _DWORD)

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def last_error() -> int:
    """The calling thread's last Win32 error, as captured by ctypes."""
    return ctypes.get_last_error()


def win_error(code: int | None = None) -> OSError:
    """An ``OSError`` whose ``winerror`` is `code`, or the last error."""
    return ctypes.WinError(last_error() if code is None else code)


def _check(ok: object) -> None:
    if not ok:
        raise win_error()


def _handle(value: int | None) -> int:
    return value or 0


# --------------------------------------------------------------------------------------
# Hooks, messages and timers
# --------------------------------------------------------------------------------------


def SetWindowsHookExW(id_hook: int, proc: Any, hmod: int = 0, thread_id: int = 0) -> int:
    """Install a hook. `proc` must be a ``HOOKPROC`` kept alive by the caller."""
    hook = _handle(_SetWindowsHookExW(id_hook, proc, hmod or None, thread_id))
    if not hook:
        raise win_error()
    return hook


def UnhookWindowsHookEx(hook: int) -> bool:
    return bool(_UnhookWindowsHookEx(hook))


def CallNextHookEx(hook: int, code: int, wparam: int, lparam: int) -> int:
    return int(_CallNextHookEx(hook or None, code, wparam, lparam))


def GetMessageW(msg: MSG, hwnd: int = 0, msg_min: int = 0, msg_max: int = 0) -> int:
    """Returns >0 for a message, 0 for ``WM_QUIT``; raises on -1."""
    result = int(_GetMessageW(ctypes.byref(msg), hwnd or None, msg_min, msg_max))
    if result == -1:
        raise win_error()
    return result


def PeekMessageW(
    msg: MSG, hwnd: int = 0, msg_min: int = 0, msg_max: int = 0, remove: int = PM_REMOVE
) -> bool:
    return bool(_PeekMessageW(ctypes.byref(msg), hwnd or None, msg_min, msg_max, remove))


def TranslateMessage(msg: MSG) -> bool:
    return bool(_TranslateMessage(ctypes.byref(msg)))


def DispatchMessageW(msg: MSG) -> int:
    return int(_DispatchMessageW(ctypes.byref(msg)))


def PostThreadMessageW(thread_id: int, msg: int, wparam: int = 0, lparam: int = 0) -> None:
    _check(_PostThreadMessageW(thread_id, msg, wparam, lparam))


def PostMessageW(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> None:
    _check(_PostMessageW(hwnd or None, msg, wparam, lparam))


def PostQuitMessage(exit_code: int = 0) -> None:
    _PostQuitMessage(exit_code)


def MsgWaitForMultipleObjects(
    handles: Sequence[int], wait_all: bool, timeout_ms: int, wake_mask: int = QS_ALLINPUT
) -> int:
    """Wait for handles or queued input. Returns ``WAIT_OBJECT_0 + n``, or ``WAIT_TIMEOUT``."""
    array = (HANDLE * max(len(handles), 1))(*handles)
    result = int(_MsgWaitForMultipleObjects(len(handles), array, wait_all, timeout_ms, wake_mask))
    if result == WAIT_FAILED:
        raise win_error()
    return result


def pump_pending_messages() -> bool:
    """Dispatch every queued message of the calling thread. False once ``WM_QUIT`` is seen."""
    msg = MSG()
    while PeekMessageW(msg):
        if msg.message == WM_QUIT:
            return False
        TranslateMessage(msg)
        DispatchMessageW(msg)
    return True


def SetTimer(hwnd: int, timer_id: int, elapse_ms: int, proc: Any = None) -> int:
    """Create or reset a timer. With ``hwnd=0`` the returned id identifies the timer."""
    result = int(_SetTimer(hwnd or None, timer_id, elapse_ms, proc or TIMERPROC()))
    if not result:
        raise win_error()
    return result


def KillTimer(hwnd: int, timer_id: int) -> bool:
    return bool(_KillTimer(hwnd or None, timer_id))


def GetLastInputInfo() -> int:
    """Tick count (``GetTickCount`` clock) of the last input event in the session."""
    info = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
    _check(_GetLastInputInfo(ctypes.byref(info)))
    return int(info.dwTime)


def RegisterHotKey(hwnd: int, hotkey_id: int, modifiers: int, vk: int) -> None:
    """Raises ``OSError`` (typically ``ERROR_HOTKEY_ALREADY_REGISTERED``) on conflict."""
    _check(_RegisterHotKey(hwnd or None, hotkey_id, modifiers, vk))


def UnregisterHotKey(hwnd: int, hotkey_id: int) -> bool:
    return bool(_UnregisterHotKey(hwnd or None, hotkey_id))


def RegisterSuspendResumeNotification(hwnd: int, flags: int = DEVICE_NOTIFY_WINDOW_HANDLE) -> int:
    """Deliver ``WM_POWERBROADCAST`` suspend/resume to `hwnd` (works for message-only windows).

    Returns the ``HPOWERNOTIFY`` handle for `UnregisterSuspendResumeNotification`.
    """
    handle = _handle(_RegisterSuspendResumeNotification(hwnd, flags))
    if not handle:
        raise win_error()
    return handle


def UnregisterSuspendResumeNotification(handle: int) -> bool:
    return bool(_UnregisterSuspendResumeNotification(handle))


# --------------------------------------------------------------------------------------
# Keyboard input
# --------------------------------------------------------------------------------------


def key_input(
    vk: int, *, up: bool = False, scan: int = 0, flags: int = 0, extra_info: int = 0
) -> INPUT:
    """A keyboard ``INPUT`` for a virtual-key press or release.

    `extra_info` becomes ``dwExtraInfo``, which low-level hooks see (to tag own input).
    """
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = KEYBDINPUT(
        wVk=vk,
        wScan=scan,
        dwFlags=flags | (KEYEVENTF_KEYUP if up else 0),
        time=0,
        dwExtraInfo=extra_info,
    )
    return inp


def mouse_input(flags: int, *, data: int = 0, extra_info: int = 0) -> INPUT:
    """A mouse ``INPUT`` at the current cursor position (e.g. ``MOUSEEVENTF_LEFTDOWN``).

    `data` is ``mouseData`` (wheel delta or X button); `extra_info` is ``dwExtraInfo``.
    """
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(
        dx=0, dy=0, mouseData=data & 0xFFFFFFFF, dwFlags=flags, time=0, dwExtraInfo=extra_info
    )
    return inp


def unicode_inputs(text: str) -> list[INPUT]:
    """Down/up ``KEYEVENTF_UNICODE`` inputs for `text`, as UTF-16 code units.

    Characters outside the basic plane become surrogate pairs.
    """
    data = text.encode("utf-16-le")
    units = [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]
    inputs: list[INPUT] = []
    for unit in units:
        for up in (False, True):
            inp = INPUT(type=INPUT_KEYBOARD)
            inp.ki = KEYBDINPUT(
                wVk=0,
                wScan=unit,
                dwFlags=KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                time=0,
                dwExtraInfo=0,
            )
            inputs.append(inp)
    return inputs


def SendInput(inputs: Sequence[INPUT]) -> int:
    """Send `inputs`. Returns how many were inserted; raises when none were."""
    if not inputs:
        return 0
    array = (INPUT * len(inputs))(*inputs)
    sent = int(_SendInput(len(inputs), array, ctypes.sizeof(INPUT)))
    if sent == 0:
        raise win_error()
    return sent


def GetAsyncKeyState(vk: int) -> int:
    return int(_GetAsyncKeyState(vk))


def is_key_down(vk: int) -> bool:
    """True while `vk` is physically down (high bit of ``GetAsyncKeyState``)."""
    return bool(GetAsyncKeyState(vk) & 0x8000)


def MapVirtualKeyExW(code: int, map_type: int, hkl: int = 0) -> int:
    return int(_MapVirtualKeyExW(code, map_type, hkl or None))


def GetKeyboardLayout(thread_id: int = 0) -> int:
    return _handle(_GetKeyboardLayout(thread_id))


def LoadKeyboardLayoutW(klid: str, flags: int = 0) -> int:
    hkl = _handle(_LoadKeyboardLayoutW(klid, flags))
    if not hkl:
        raise win_error()
    return hkl


def ActivateKeyboardLayout(hkl: int, flags: int = 0) -> int:
    previous = _handle(_ActivateKeyboardLayout(hkl, flags))
    if not previous:
        raise win_error()
    return previous


def UnloadKeyboardLayout(hkl: int) -> bool:
    return bool(_UnloadKeyboardLayout(hkl))


def GetKeyboardLayoutList() -> list[int]:
    """Every input locale identifier (HKL) loaded for the session."""
    count = int(_GetKeyboardLayoutList(0, None))
    array = (HKL * max(count, 1))()
    count = int(_GetKeyboardLayoutList(count, array))
    return [_handle(array[i]) for i in range(count)]


def AttachThreadInput(attach_from: int, attach_to: int, attach: bool) -> bool:
    return bool(_AttachThreadInput(attach_from, attach_to, attach))


# --------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------


def GetModuleHandleW(name: str | None = None) -> int:
    return _handle(_GetModuleHandleW(name))


def DefWindowProcW(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    return int(_DefWindowProcW(hwnd or None, msg, wparam, lparam))


def RegisterClassExW(class_name: str, wndproc: Any) -> int:
    """Register a window class. `wndproc` must be a ``WNDPROC`` kept alive by the caller."""
    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = wndproc
    wc.hInstance = GetModuleHandleW() or None
    wc.lpszClassName = class_name
    atom = int(_RegisterClassExW(ctypes.byref(wc)))
    if not atom:
        raise win_error()
    return atom


def UnregisterClassW(class_name: str) -> bool:
    return bool(_UnregisterClassW(class_name, GetModuleHandleW() or None))


def CreateWindowExW(
    ex_style: int,
    class_name: str,
    window_name: str = "",
    style: int = 0,
    x: int = 0,
    y: int = 0,
    width: int = 0,
    height: int = 0,
    parent: int = 0,
) -> int:
    hwnd = _handle(
        _CreateWindowExW(
            ex_style,
            class_name,
            window_name,
            style,
            x,
            y,
            width,
            height,
            parent or None,
            None,
            GetModuleHandleW() or None,
            None,
        )
    )
    if not hwnd:
        raise win_error()
    return hwnd


def create_message_window(class_name: str, wndproc: Any) -> int:
    """Register `class_name` and create a message-only (``HWND_MESSAGE``) window.

    The window belongs to the calling thread, which must pump its messages.
    """
    RegisterClassExW(class_name, wndproc)
    try:
        return CreateWindowExW(0, class_name, class_name, parent=HWND_MESSAGE)
    except OSError:
        UnregisterClassW(class_name)
        raise


def destroy_message_window(hwnd: int, class_name: str) -> None:
    DestroyWindow(hwnd)
    UnregisterClassW(class_name)


def DestroyWindow(hwnd: int) -> bool:
    return bool(_DestroyWindow(hwnd))


def IsWindow(hwnd: int) -> bool:
    return bool(_IsWindow(hwnd or None))


def GetForegroundWindow() -> int:
    return _handle(_GetForegroundWindow())


def SetForegroundWindow(hwnd: int) -> bool:
    return bool(_SetForegroundWindow(hwnd))


def BringWindowToTop(hwnd: int) -> bool:
    return bool(_BringWindowToTop(hwnd))


def ShowWindow(hwnd: int, cmd: int) -> bool:
    """True when the window was visible before the call."""
    return bool(_ShowWindow(hwnd, cmd))


def GetShellWindow() -> int:
    """The desktop window of the shell (`Progman`), or 0."""
    return int(_GetShellWindow() or 0)


def FindWindowExW(
    parent: int = 0, after: int = 0, class_name: str | None = None, title: str | None = None
) -> int:
    """The next top-level (``parent=0``) or child window matching class and title; 0 if none."""
    return _handle(_FindWindowExW(parent or None, after or None, class_name, title))


def find_windows(class_name: str) -> list[int]:
    """Every top-level window of class `class_name`."""
    found: list[int] = []
    hwnd = FindWindowExW(0, 0, class_name)
    while hwnd:
        found.append(hwnd)
        hwnd = FindWindowExW(0, hwnd, class_name)
    return found


def WindowFromPoint(x: int, y: int) -> int:
    """The window under the screen point, honoring ``WS_EX_TRANSPARENT`` hit-testing."""
    return _handle(_WindowFromPoint(POINT(x, y)))


def GetAncestor(hwnd: int, flags: int = GA_ROOT) -> int:
    return _handle(_GetAncestor(hwnd, flags))


def GetCursorPos() -> tuple[int, int]:
    point = POINT()
    _check(_GetCursorPos(ctypes.byref(point)))
    return point.x, point.y


def SetCursorPos(x: int, y: int) -> None:
    _check(_SetCursorPos(x, y))


def GetClassNameW(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    if not _GetClassNameW(hwnd, buf, len(buf)):
        raise win_error()
    return buf.value


def GetWindowThreadProcessId(hwnd: int) -> tuple[int, int]:
    """``(thread_id, process_id)``; ``(0, 0)`` for an invalid window."""
    pid = _DWORD(0)
    tid = int(_GetWindowThreadProcessId(hwnd, ctypes.byref(pid)))
    return tid, int(pid.value)


def GetWindowLongPtrW(hwnd: int, index: int) -> int:
    ctypes.set_last_error(0)
    value = int(_GetWindowLongPtrW(hwnd, index))
    if value == 0 and last_error():
        raise win_error()
    return value


def SetWindowLongPtrW(hwnd: int, index: int, value: int) -> int:
    """Returns the previous value."""
    ctypes.set_last_error(0)
    previous = int(_SetWindowLongPtrW(hwnd, index, value))
    if previous == 0 and last_error():
        raise win_error()
    return previous


def SetWindowPos(
    hwnd: int, insert_after: int, x: int, y: int, cx: int, cy: int, flags: int
) -> None:
    _check(_SetWindowPos(hwnd, insert_after, x, y, cx, cy, flags))


def GetWindowRect(hwnd: int) -> tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` in screen coordinates."""
    rect = RECT()
    _check(_GetWindowRect(hwnd, ctypes.byref(rect)))
    return rect.left, rect.top, rect.right, rect.bottom


def MessageBoxW(text: str, caption: str = "talktype", flags: int = MB_OK, hwnd: int = 0) -> int:
    return int(_MessageBoxW(hwnd or None, text, caption, flags))


# --------------------------------------------------------------------------------------
# Clipboard and global memory
# --------------------------------------------------------------------------------------


def OpenClipboard(hwnd: int = 0) -> bool:
    """False when another window holds the clipboard open (``last_error()`` has details)."""
    return bool(_OpenClipboard(hwnd or None))


def CloseClipboard() -> bool:
    return bool(_CloseClipboard())


def EmptyClipboard() -> None:
    _check(_EmptyClipboard())


def GetClipboardData(fmt: int) -> int:
    return _handle(_GetClipboardData(fmt))


def SetClipboardData(fmt: int, handle: int = 0) -> int:
    """Put `handle` on the clipboard. ``handle=0`` requests delayed rendering.

    On success the system owns the handle. The result is not checked here, because it is
    NULL for delayed rendering.
    """
    return _handle(_SetClipboardData(fmt, handle or None))


def EnumClipboardFormats(previous: int = 0) -> int:
    return int(_EnumClipboardFormats(previous))


def clipboard_formats() -> list[int]:
    """Every format on the open clipboard, in enumeration order."""
    formats: list[int] = []
    fmt = EnumClipboardFormats(0)
    while fmt:
        formats.append(fmt)
        fmt = EnumClipboardFormats(fmt)
    return formats


def IsClipboardFormatAvailable(fmt: int) -> bool:
    return bool(_IsClipboardFormatAvailable(fmt))


def RegisterClipboardFormatW(name: str) -> int:
    fmt = int(_RegisterClipboardFormatW(name))
    if not fmt:
        raise win_error()
    return fmt


def GetClipboardFormatNameW(fmt: int) -> str | None:
    """The name of a registered format, or None for predefined formats."""
    buf = ctypes.create_unicode_buffer(256)
    length = int(_GetClipboardFormatNameW(fmt, buf, len(buf)))
    return buf.value if length else None


def GetClipboardOwner() -> int:
    return _handle(_GetClipboardOwner())


def GetOpenClipboardWindow() -> int:
    """The window that has the clipboard open (0 when none, or opened without a window)."""
    return _handle(_GetOpenClipboardWindow())


def GetClipboardSequenceNumber() -> int:
    return int(_GetClipboardSequenceNumber())


def GlobalAlloc(flags: int, size: int) -> int:
    handle = _handle(_GlobalAlloc(flags, size))
    if not handle:
        raise win_error()
    return handle


def GlobalLock(handle: int) -> int:
    ptr = _handle(_GlobalLock(handle))
    if not ptr:
        raise win_error()
    return ptr


def GlobalUnlock(handle: int) -> bool:
    return bool(_GlobalUnlock(handle))


def GlobalSize(handle: int) -> int:
    return int(_GlobalSize(handle))


def GlobalFree(handle: int) -> None:
    if _GlobalFree(handle):
        raise win_error()


def global_from_bytes(data: bytes) -> int:
    """A movable ``HGLOBAL`` holding `data`. The caller owns it until the clipboard does."""
    handle = GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
    try:
        ptr = GlobalLock(handle)
        try:
            ctypes.memmove(ptr, data, len(data))
        finally:
            GlobalUnlock(handle)
    except OSError:
        GlobalFree(handle)
        raise
    return handle


def global_to_bytes(handle: int) -> bytes:
    """A copy of the memory behind an ``HGLOBAL``."""
    size = GlobalSize(handle)
    ptr = GlobalLock(handle)
    try:
        return ctypes.string_at(ptr, size)
    finally:
        GlobalUnlock(handle)


def unicode_text_bytes(text: str) -> bytes:
    """`text` as NUL-terminated UTF-16-LE, the ``CF_UNICODETEXT`` layout."""
    return text.encode("utf-16-le") + b"\x00\x00"


def text_from_unicode_bytes(data: bytes) -> str:
    """Decode ``CF_UNICODETEXT`` bytes up to the first NUL."""
    text = data.decode("utf-16-le", errors="replace")
    return text.split("\x00", 1)[0]


def GetEnhMetaFileBits(hemf: int) -> bytes:
    size = int(_GetEnhMetaFileBits(hemf, 0, None))
    if not size:
        raise win_error()
    buf = ctypes.create_string_buffer(size)
    if not _GetEnhMetaFileBits(hemf, size, buf):
        raise win_error()
    return buf.raw


def SetEnhMetaFileBits(data: bytes) -> int:
    hemf = _handle(_SetEnhMetaFileBits(len(data), data))
    if not hemf:
        raise win_error()
    return hemf


def DeleteEnhMetaFile(hemf: int) -> bool:
    return bool(_DeleteEnhMetaFile(hemf))


# --------------------------------------------------------------------------------------
# Threads, processes and tokens
# --------------------------------------------------------------------------------------


def GetCurrentThreadId() -> int:
    return int(_GetCurrentThreadId())


def GetCurrentProcessId() -> int:
    return int(_GetCurrentProcessId())


def GetTickCount() -> int:
    """Milliseconds since boot, wrapping at 2**32 (the clock of hook event times)."""
    return int(_GetTickCount())


def GetTickCount64() -> int:
    return int(_GetTickCount64())


def OpenProcess(access: int, pid: int, inherit: bool = False) -> int:
    handle = _handle(_OpenProcess(access, inherit, pid))
    if not handle:
        raise win_error()
    return handle


def TerminateProcess(handle: int, exit_code: int = 1) -> None:
    _check(_TerminateProcess(handle, exit_code))


def CloseHandle(handle: int) -> bool:
    return bool(_CloseHandle(handle))


def GetCurrentProcess() -> int:
    """The pseudo-handle of the current process (does not need closing)."""
    return _handle(_GetCurrentProcess())


def OpenProcessToken(process: int, access: int = TOKEN_QUERY) -> int:
    token = HANDLE()
    _check(_OpenProcessToken(process, access, ctypes.byref(token)))
    return _handle(token.value)


def token_is_elevated(token: int) -> bool:
    """``GetTokenInformation(TokenElevation)``."""
    elevation = TOKEN_ELEVATION()
    returned = _DWORD(0)
    _check(
        _GetTokenInformation(
            token,
            TOKEN_ELEVATION_CLASS,
            ctypes.byref(elevation),
            ctypes.sizeof(elevation),
            ctypes.byref(returned),
        )
    )
    return bool(elevation.TokenIsElevated)


def process_is_elevated(pid: int) -> bool | None:
    """Whether process `pid` runs elevated; None when its token cannot be queried."""
    try:
        process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, pid)
    except OSError:
        return None
    try:
        token = OpenProcessToken(process)
    except OSError:
        return None
    finally:
        CloseHandle(process)
    try:
        return token_is_elevated(token)
    except OSError:
        return None
    finally:
        CloseHandle(token)


def current_process_is_elevated() -> bool:
    token = OpenProcessToken(GetCurrentProcess())
    try:
        return token_is_elevated(token)
    finally:
        CloseHandle(token)


def IsUserAnAdmin() -> bool:
    return bool(_IsUserAnAdmin())


# --------------------------------------------------------------------------------------
# Mutex
# --------------------------------------------------------------------------------------


def CreateMutexW(name: str, initial_owner: bool = False) -> tuple[int, bool]:
    """``(handle, already_existed)`` for the named mutex."""
    handle = _handle(_CreateMutexW(None, initial_owner, name))
    error = last_error()
    if not handle:
        raise win_error(error)
    return handle, error == ERROR_ALREADY_EXISTS


# --------------------------------------------------------------------------------------
# Session notifications
# --------------------------------------------------------------------------------------


def WTSRegisterSessionNotification(hwnd: int, flags: int = NOTIFY_FOR_THIS_SESSION) -> None:
    _check(_WTSRegisterSessionNotification(hwnd, flags))


def WTSUnRegisterSessionNotification(hwnd: int) -> bool:
    return bool(_WTSUnRegisterSessionNotification(hwnd))


# --------------------------------------------------------------------------------------
# Sound
# --------------------------------------------------------------------------------------


def PlaySoundW(sound: bytes | None, flags: int) -> bool:
    """Play the in-memory WAV image `sound` (with ``SND_MEMORY``), or stop playback (None).

    PlaySound plays one sound per process: a new call stops the sound still playing. With
    ``SND_ASYNC`` the caller must keep `sound` referenced until playback ends or is stopped.
    False when the image cannot be decoded or no output device can play it.
    """
    return bool(_PlaySoundW(sound, None, flags))


# --------------------------------------------------------------------------------------
# Registry (HKCU only; through the stdlib winreg, not pywin32)
# --------------------------------------------------------------------------------------


def reg_read_str(root: int, subkey: str, name: str) -> str | None:
    """A string value, or None when the key or value is absent."""
    try:
        with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ) as key:
            value, kind = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
        return None
    return str(value)


def reg_write_str(root: int, subkey: str, name: str, value: str) -> None:
    """Create the key if needed and write a ``REG_SZ``. Raises ``OSError`` (e.g. denied)."""
    with winreg.CreateKeyEx(root, subkey, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def reg_delete_value(root: int, subkey: str, name: str) -> bool:
    """Delete a value. False when it did not exist."""
    try:
        with winreg.OpenKey(root, subkey, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        return False
    return True


RegReader = Callable[[int, str, str], str | None]


def read_mic_consent(reader: RegReader = reg_read_str) -> str | None:
    """The Windows microphone privacy value for desktop apps, read-only.

    Returns ``"Deny"`` when the global ConsentStore value or its ``NonPackaged`` value is
    ``Deny``, ``"Allow"`` when either is ``Allow``, and None when neither exists.
    """
    values = [
        reader(HKEY_CURRENT_USER, key, "Value")
        for key in (MIC_CONSENT_KEY, MIC_CONSENT_NONPACKAGED_KEY)
    ]
    if "Deny" in values:
        return "Deny"
    if "Allow" in values:
        return "Allow"
    return None


# --------------------------------------------------------------------------------------
# NVML shim (nvidia-ml-py; imported lazily so the GPU stack loads only when needed)
# --------------------------------------------------------------------------------------


class GpuMemory(NamedTuple):
    free: int
    used: int
    total: int


def nvml_memory(index: int = 0, nvml: Any = None) -> GpuMemory:
    """Memory of GPU `index` via ``nvmlInit`` / ``nvmlDeviceGetMemoryInfo`` / ``nvmlShutdown``.

    `nvml` defaults to the ``pynvml`` module; tests pass a fake with the same four functions.
    Raises whatever NVML raises (no driver, no GPU); callers treat any exception as CPU.
    """
    if nvml is None:
        import pynvml

        nvml = pynvml
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuMemory(free=int(info.free), used=int(info.used), total=int(info.total))
    finally:
        nvml.nvmlShutdown()
