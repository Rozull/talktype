"""Clipboard snapshot and restore, and the delayed-render owner window.

- `ClipboardSnapshot` saves every format that can be captured: ``HGLOBAL``-based formats as
  bytes and ``CF_ENHMETAFILE`` through ``GetEnhMetaFileBits``. Handle-only formats that
  Windows synthesizes from a captured one (``CF_BITMAP``, ``CF_METAFILEPICT``,
  ``CF_PALETTE``) are skipped. Any other format that cannot be read, or that would push the
  snapshot past `MAX_SNAPSHOT_BYTES`, marks it `partial`.
- Everything talktype puts on the clipboard (the dictated text and every restore) carries
  the markers ``ExcludeClipboardContentFromMonitorProcessing``,
  ``CanIncludeInClipboardHistory = 0`` and ``CanUploadToCloudClipboard = 0``, so dictations
  stay out of Win+V history, the cloud clipboard and well-behaved clipboard managers.
- `ClipboardOwner` is a message-only window on the calling (delivery) thread. It offers
  ``CF_UNICODETEXT`` with delayed rendering and renders it on ``WM_RENDERFORMAT``: the target
  reading the clipboard is what confirms a paste. It restores the snapshot only while it
  still owns the clipboard.
- Opening the clipboard is retried 5 times at 20 ms intervals.

Every function takes `api`: the `win32` module, or a fake with the same functions (tests).
Line breaks are stored as ``\\r\\n``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from talktype import win32
from talktype.logging_setup import get_logger, log_event

logger = get_logger("clipboard")

OPEN_ATTEMPTS = 5
OPEN_RETRY_S = 0.020
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MARKER_NAMES = (
    win32.CFSTR_EXCLUDE_MONITOR,
    win32.CFSTR_CAN_INCLUDE_IN_HISTORY,
    win32.CFSTR_CAN_UPLOAD_TO_CLOUD,
)
MARKER_VALUE = (0).to_bytes(4, "little")  # DWORD 0

# GDI handles that Windows re-synthesizes from the DIB / metafile formats we do capture.
_SYNTHESIZED_HANDLES = frozenset({win32.CF_BITMAP, win32.CF_METAFILEPICT, win32.CF_PALETTE})
# Handle formats that cannot be copied at all.
_UNCAPTURABLE = frozenset({win32.CF_OWNERDISPLAY, win32.CF_DSPBITMAP, win32.CF_DSPMETAFILEPICT})
_METAFILES = frozenset({win32.CF_ENHMETAFILE, win32.CF_DSPENHMETAFILE})
_LINE_BREAK = re.compile(r"\r\n|\r|\n")

DataKind = Literal["hglobal", "emf"]


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds."""
        ...

    def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ClipboardLockedError(Exception):
    """Another window kept the clipboard open through every retry."""


def clipboard_text(text: str) -> bytes:
    """`text` as ``CF_UNICODETEXT`` bytes, with ``\\r\\n`` line breaks."""
    return win32.unicode_text_bytes(_LINE_BREAK.sub("\r\n", text))


def open_clipboard(
    hwnd: int = 0,
    *,
    api: Any = win32,
    clock: Clock | None = None,
    attempts: int = OPEN_ATTEMPTS,
    interval_s: float = OPEN_RETRY_S,
) -> bool:
    """Open the clipboard for `hwnd`, retrying while another window holds it."""
    clock = clock or MonotonicClock()
    for attempt in range(attempts):
        if api.OpenClipboard(hwnd):
            return True
        if attempt + 1 < attempts:
            clock.sleep(interval_s)
    return False


def pump_until(
    api: Any, clock: Clock, timeout_s: float, until: Callable[[], bool] | None = None
) -> bool:
    """Dispatch this thread's messages for up to `timeout_s`, or until `until()` holds.

    Returns ``until()`` (False when no condition was given).
    """
    deadline = clock.now() + timeout_s
    api.pump_pending_messages()
    while until is None or not until():
        remaining = deadline - clock.now()
        if remaining <= 0:
            return False
        # Round up to whole ms, ignoring float noise (5.000000001 ms waits 5 ms, not 6).
        api.MsgWaitForMultipleObjects([], False, max(1, math.ceil(remaining * 1000 - 1e-3)))
        api.pump_pending_messages()
    return True


def _set_data(fmt: int, data: bytes, api: Any, kind: DataKind = "hglobal") -> bool:
    """Put one format on the open clipboard. The system owns the handle on success."""
    try:
        handle = api.SetEnhMetaFileBits(data) if kind == "emf" else api.global_from_bytes(data)
    except OSError:
        return False
    if api.SetClipboardData(fmt, handle):
        return True
    with contextlib.suppress(OSError):
        if kind == "emf":
            api.DeleteEnhMetaFile(handle)
        else:
            api.GlobalFree(handle)
    return False


def marker_formats(api: Any = win32) -> tuple[int, ...]:
    return tuple(api.RegisterClipboardFormatW(name) for name in MARKER_NAMES)


def set_markers(api: Any = win32) -> bool:
    """Add the history, cloud and monitor exclusion markers to the open clipboard."""
    ok = True
    for fmt in marker_formats(api):
        ok = _set_data(fmt, MARKER_VALUE, api) and ok
    return ok


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipFormat:
    format: int
    data: bytes
    kind: DataKind = "hglobal"


@dataclass(frozen=True, slots=True)
class ClipboardSnapshot:
    formats: tuple[ClipFormat, ...] = ()
    partial: bool = False  # some format could not be captured

    @property
    def empty(self) -> bool:
        return not self.formats

    def get(self, fmt: int) -> bytes | None:
        return next((f.data for f in self.formats if f.format == fmt), None)

    @classmethod
    def read(cls, api: Any = win32) -> ClipboardSnapshot:
        """Capture the clipboard, which the caller has open."""
        items: list[ClipFormat] = []
        partial = False
        total = 0
        for fmt in api.clipboard_formats():
            if fmt in _SYNTHESIZED_HANDLES:
                continue
            if (
                fmt in _UNCAPTURABLE
                or win32.CF_PRIVATEFIRST <= fmt <= win32.CF_PRIVATELAST
                or win32.CF_GDIOBJFIRST <= fmt <= win32.CF_GDIOBJLAST
            ):
                partial = True
                continue
            handle = api.GetClipboardData(fmt)
            if not handle:
                partial = True
                continue
            kind: DataKind = "emf" if fmt in _METAFILES else "hglobal"
            try:
                data = (
                    api.GetEnhMetaFileBits(handle) if kind == "emf" else api.global_to_bytes(handle)
                )
            except OSError:
                partial = True
                continue
            if total + len(data) > MAX_SNAPSHOT_BYTES:
                partial = True
                continue
            total += len(data)
            items.append(ClipFormat(fmt, data, kind))
        return cls(tuple(items), partial)

    def write(self, api: Any = win32) -> bool:
        """Replace the open clipboard's content with this snapshot.

        An empty snapshot leaves the clipboard empty. Otherwise the exclusion markers are
        added. Returns False when some format could not be set.
        """
        api.EmptyClipboard()
        if not self.formats:
            return True
        markers = set(marker_formats(api))
        ok = True
        for item in self.formats:
            if item.format not in markers:
                ok = _set_data(item.format, item.data, api, item.kind) and ok
        return set_markers(api) and ok

    @classmethod
    def capture(
        cls, hwnd: int = 0, *, api: Any = win32, clock: Clock | None = None
    ) -> ClipboardSnapshot:
        """Open the clipboard and capture it. Raises `ClipboardLockedError`."""
        if not open_clipboard(hwnd, api=api, clock=clock):
            raise ClipboardLockedError
        try:
            return cls.read(api)
        finally:
            api.CloseClipboard()

    def restore(self, hwnd: int = 0, *, api: Any = win32, clock: Clock | None = None) -> bool:
        """Open the clipboard and write this snapshot. Raises `ClipboardLockedError`.

        Returns False when some format could not be set.
        """
        if not open_clipboard(hwnd, api=api, clock=clock):
            raise ClipboardLockedError
        try:
            return self.write(api)
        finally:
            api.CloseClipboard()


def put_text(text: str, hwnd: int = 0, *, api: Any = win32, clock: Clock | None = None) -> bool:
    """Replace the clipboard with `text` (and the markers). False when it stays locked."""
    if not open_clipboard(hwnd, api=api, clock=clock):
        return False
    try:
        api.EmptyClipboard()
        ok = _set_data(win32.CF_UNICODETEXT, clipboard_text(text), api)
        set_markers(api)
        return ok
    except OSError:
        return False
    finally:
        api.CloseClipboard()


def read_text(hwnd: int = 0, *, api: Any = win32, clock: Clock | None = None) -> str | None:
    """The clipboard's ``CF_UNICODETEXT``, or None when absent. Raises `ClipboardLockedError`."""
    if not open_clipboard(hwnd, api=api, clock=clock):
        raise ClipboardLockedError
    try:
        handle = api.GetClipboardData(win32.CF_UNICODETEXT)
        return win32.text_from_unicode_bytes(api.global_to_bytes(handle)) if handle else None
    finally:
        api.CloseClipboard()


# --------------------------------------------------------------------------------------
# Delayed-render owner
# --------------------------------------------------------------------------------------


class ClipboardOwner:
    """The message-only window that owns the clipboard during a paste.

    Created lazily on the first call, on the calling thread, which must be the one that
    pumps its messages (the delivery thread).
    """

    def __init__(self, *, api: Any = win32, clock: Clock | None = None) -> None:
        self._api = api
        self._clock: Clock = clock or MonotonicClock()
        self._proc = api.WNDPROC(self._wndproc)  # kept referenced for Windows
        self._class_name = f"talktype-clipboard-{uuid.uuid4().hex}"
        self._hwnd = 0
        self._pending: bytes | None = None  # delayed CF_UNICODETEXT not yet rendered
        self.rendered = False  # a WM_RENDERFORMAT arrived for the current offer

    @property
    def hwnd(self) -> int:
        if not self._hwnd:
            self._hwnd = int(self._api.create_message_window(self._class_name, self._proc))
        return self._hwnd

    def _open(self) -> bool:
        return open_clipboard(self.hwnd, api=self._api, clock=self._clock)

    # -- window procedure --------------------------------------------------------------

    def _wndproc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        try:
            if msg == win32.WM_RENDERFORMAT:
                if wparam == win32.CF_UNICODETEXT and self._render():
                    self.rendered = True
                    self._log_requester()
                return 0
            if msg == win32.WM_RENDERALLFORMATS:
                self._render_all(hwnd)
                return 0
        except Exception as exc:  # never let an exception unwind into Windows
            log_event(
                logger,
                "worker_error",
                level=logging.ERROR,
                thread="clipboard",
                exc=type(exc).__name__,
                exc_info=True,
            )
            return 0
        return int(self._api.DefWindowProcW(hwnd, msg, wparam, lparam))

    def _log_requester(self) -> None:
        """Log who read the text: a clipboard monitor here would be a false confirmation."""
        requester = int(self._api.GetOpenClipboardWindow())
        class_name = ""
        pid = 0
        if requester:
            with contextlib.suppress(OSError):
                class_name = str(self._api.GetClassNameW(requester))
            _, pid = self._api.GetWindowThreadProcessId(requester)
        log_event(logger, "paste_rendered", requester_class=class_name, requester_pid=pid)

    def _render(self) -> bool:
        """Answer ``WM_RENDERFORMAT``: the requesting window holds the clipboard open."""
        if self._pending is None:
            return False
        return _set_data(win32.CF_UNICODETEXT, self._pending, self._api)

    def _render_all(self, hwnd: int) -> None:
        """Our window is going away while it still owes the text: render it for good."""
        if self._pending is None or not self._api.OpenClipboard(hwnd):
            return
        try:
            if self._api.GetClipboardOwner() == hwnd:
                self._render()
        finally:
            self._api.CloseClipboard()

    # -- operations --------------------------------------------------------------------

    def offer(self, text: str) -> ClipboardSnapshot | None:
        """Snapshot the clipboard, then own it with `text` rendered on request.

        Returns None when the clipboard stays locked.
        """
        if not self._open():
            return None
        try:
            snapshot = ClipboardSnapshot.read(self._api)
            self._api.EmptyClipboard()
            self._pending = clipboard_text(text)
            self.rendered = False
            self._api.SetClipboardData(win32.CF_UNICODETEXT, 0)  # delayed rendering
            set_markers(self._api)
        finally:
            self._api.CloseClipboard()
        return snapshot

    def wait_render(self, timeout_s: float) -> bool:
        """Pump messages until the offered text is requested, or `timeout_s` passes."""
        return pump_until(self._api, self._clock, timeout_s, lambda: self.rendered)

    def settle(self, seconds: float) -> None:
        """Keep pumping messages for `seconds` (the restore delay)."""
        pump_until(self._api, self._clock, seconds)

    def owns(self) -> bool:
        return bool(self._hwnd) and self._api.GetClipboardOwner() == self._hwnd

    def restore(self, snapshot: ClipboardSnapshot) -> tuple[bool, bool]:
        """Put `snapshot` back while our window still owns the clipboard.

        Returns ``(restored, partial)``. Nothing is restored when another window copied in the
        meantime or the clipboard stays locked.
        """
        if not self._open():
            return False, False
        try:
            if self._api.GetClipboardOwner() != self._hwnd:
                return False, False
            self._pending = None
            complete = snapshot.write(self._api)
            return True, snapshot.partial or not complete
        finally:
            self._api.CloseClipboard()

    def put_text(self, text: str) -> bool:
        """Leave `text` on the clipboard as normal content (the failure path)."""
        ok = put_text(text, self.hwnd, api=self._api, clock=self._clock)
        if ok:
            self._pending = None
        return ok

    def close(self) -> None:
        """Destroy the window (on its own thread). Owed text is rendered first."""
        if self._hwnd:
            with contextlib.suppress(OSError):
                self._api.destroy_message_window(self._hwnd, self._class_name)
            self._hwnd = 0
            self._pending = None
