"""`Injector.inject(text, mode)`: paste and type modes.

Both modes start with the same steps:
- **Pre-checks** through `probe_target()`: no focus or an elevated target fails at once.
- **Trigger release.** Before the first ``SendInput`` the injector polls the trigger key and
  every modifier (Ctrl, Alt, Shift, Win) every 10 ms for up to 500 ms, so injected text never
  combines with a held modifier (the re-paste hotkey is still held when it fires). On timeout
  it proceeds and logs ``trigger_still_down``.

**Paste** snapshots the clipboard, owns it with the text in delayed rendering, sends
Ctrl+V (virtual keys, layout independent), then pumps messages for up to
`paste_confirm_timeout_ms` waiting for the target to request the text. When it does, the
result is ``INSERTED`` and, after `restore_delay_ms`, the snapshot is restored while our window
still owns the clipboard. When it does not, the result is ``FAILED_UNCONFIRMED``. The text
then stays on the clipboard as normal content, and nothing is restored.

**Type** sends ``KEYEVENTF_UNICODE`` input in chunks of `type_chunk_chars` characters, 5 ms
apart, and re-checks the foreground window between chunks. Line breaks become
Shift+Enter; characters outside the basic plane become surrogate pairs. A foreground change
stops typing: the full text goes to the clipboard and the result is ``PARTIAL``. A successful
type never touches the clipboard.

No sequence ever contains a ``VK_RETURN`` without ``VK_SHIFT`` held. Every failure
leaves the text on the clipboard when the clipboard can be written. `inject()` must run on
the delivery thread, which owns the clipboard window.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from talktype import win32
from talktype.clipboard import ClipboardOwner, Clock, MonotonicClock, pump_until
from talktype.config import InjectionSettings
from talktype.foreground import probe_target
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg

logger = get_logger("injector")

TRIGGER_WAIT_S = 0.500
TRIGGER_POLL_S = 0.010
TYPE_CHUNK_GAP_S = 0.005
MODIFIER_VKS = (
    win32.VK_LCONTROL,
    win32.VK_RCONTROL,
    win32.VK_LMENU,
    win32.VK_RMENU,
    win32.VK_LSHIFT,
    win32.VK_RSHIFT,
    win32.VK_LWIN,
    win32.VK_RWIN,
)

_LINE_BREAK = re.compile(r"\r\n|\r")


class InjectMode(StrEnum):
    PASTE = "paste"
    TYPE = "type"


class InjectStatus(StrEnum):
    INSERTED = "inserted"
    FAILED_NO_FOCUS = "failed_no_focus"
    FAILED_ELEVATED = "failed_elevated"
    FAILED_UNCONFIRMED = "failed_unconfirmed"
    FAILED_CLIPBOARD_LOCKED = "failed_clipboard_locked"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class InjectResult:
    status: InjectStatus
    text_on_clipboard: bool = False  # True on every failure where the clipboard write succeeded
    clipboard_restored: bool = False
    restore_partial: bool = False  # the snapshot had formats that could not be captured

    @property
    def inserted(self) -> bool:
        return self.status is InjectStatus.INSERTED


_FAILURE_MESSAGES = {
    InjectStatus.FAILED_NO_FOCUS: Msg.INSERT_FAILED_NO_FOCUS,
    InjectStatus.FAILED_ELEVATED: Msg.INSERT_FAILED_ELEVATED,
    InjectStatus.FAILED_UNCONFIRMED: Msg.INSERT_FAILED_CLIPBOARD,
    InjectStatus.PARTIAL: Msg.INSERT_PARTIAL,
    InjectStatus.FAILED_CLIPBOARD_LOCKED: Msg.INSERT_FAILED_OPEN_HISTORY,
}


def message_for(result: InjectResult) -> Msg | None:
    """The overlay message for `result`; None for a clean insert.

    A failure whose text could not be put on the clipboard points to the history instead.
    """
    if result.status is InjectStatus.INSERTED:
        return Msg.CLIPBOARD_PARTIAL_RESTORE if result.restore_partial else None
    if not result.text_on_clipboard:
        return Msg.INSERT_FAILED_OPEN_HISTORY
    return _FAILURE_MESSAGES[result.status]


# --------------------------------------------------------------------------------------
# Key sequences
# --------------------------------------------------------------------------------------


def paste_inputs() -> list[win32.INPUT]:
    """Ctrl+V with virtual-key codes: the same on every keyboard layout."""
    return [
        win32.key_input(win32.VK_CONTROL),
        win32.key_input(win32.VK_V),
        win32.key_input(win32.VK_V, up=True),
        win32.key_input(win32.VK_CONTROL, up=True),
    ]


def newline_inputs() -> list[win32.INPUT]:
    """Shift+Enter: a line break that never submits."""
    return [
        win32.key_input(win32.VK_SHIFT),
        win32.key_input(win32.VK_RETURN),
        win32.key_input(win32.VK_RETURN, up=True),
        win32.key_input(win32.VK_SHIFT, up=True),
    ]


def type_chunks(text: str, chunk_chars: int) -> list[list[win32.INPUT]]:
    """The ``SendInput`` batches for typing `text`, `chunk_chars` characters each."""
    chars = list(_LINE_BREAK.sub("\n", text))
    size = max(1, chunk_chars)
    batches: list[list[win32.INPUT]] = []
    for start in range(0, len(chars), size):
        batch: list[win32.INPUT] = []
        for char in chars[start : start + size]:
            batch.extend(newline_inputs() if char == "\n" else win32.unicode_inputs(char))
        batches.append(batch)
    return batches


# --------------------------------------------------------------------------------------
# Injector
# --------------------------------------------------------------------------------------


class Injector:
    """Inserts text into the foreground window (delivery thread only).

    `overlay_hwnd` (or a callable returning it) is excluded as a target. `trigger_vk` may be
    reassigned when the trigger key changes. `api` is the `win32` module or a fake.
    """

    def __init__(
        self,
        *,
        overlay_hwnd: int | Callable[[], int] = 0,
        trigger_vk: int = win32.VK_RCONTROL,
        api: Any = win32,
        clock: Clock | None = None,
    ) -> None:
        self.trigger_vk = trigger_vk
        self._overlay_hwnd = overlay_hwnd
        self._api = api
        self._clock: Clock = clock or MonotonicClock()
        self._owner: ClipboardOwner | None = None

    @property
    def clipboard(self) -> ClipboardOwner:
        if self._owner is None:
            self._owner = ClipboardOwner(api=self._api, clock=self._clock)
        return self._owner

    def close(self) -> None:
        """Destroy the clipboard window (on the thread that injected)."""
        if self._owner is not None:
            self._owner.close()
            self._owner = None

    def put_on_clipboard(self, text: str) -> bool:
        """Leave `text` on the clipboard as normal content. False when it stays locked."""
        return self.clipboard.put_text(text)

    # -- inject ------------------------------------------------------------------------

    def inject(
        self, text: str, mode: InjectMode | str, settings: InjectionSettings | None = None
    ) -> InjectResult:
        s = settings or InjectionSettings()
        mode = InjectMode(mode)
        overlay = self._overlay_hwnd() if callable(self._overlay_hwnd) else self._overlay_hwnd
        target = probe_target(overlay, api=self._api)
        if target.status == "no_focus":
            return self._fail(text, InjectStatus.FAILED_NO_FOCUS, target_class=target.class_name)
        if target.status == "elevated":
            return self._fail(text, InjectStatus.FAILED_ELEVATED, target_class=target.class_name)
        self._wait_trigger_release()
        if mode is InjectMode.PASTE:
            return self._paste(text, s)
        return self._type(text, s, target.hwnd)

    def _fail(self, text: str, status: InjectStatus, **fields: Any) -> InjectResult:
        on_clipboard = self.put_on_clipboard(text)
        log_event(logger, "inject_failed", status=status, text_on_clipboard=on_clipboard, **fields)
        return InjectResult(status, text_on_clipboard=on_clipboard)

    def _held_keys(self) -> list[int]:
        keys = dict.fromkeys((self.trigger_vk, *MODIFIER_VKS))  # ordered, deduplicated
        return [vk for vk in keys if vk and self._api.is_key_down(vk)]

    def _wait_trigger_release(self) -> None:
        """Wait for the trigger and every modifier to be up (the re-paste hotkey is still
        held when ``WM_HOTKEY`` arrives: Ctrl+V with Alt+Shift down would not paste)."""
        if not self._held_keys():
            return
        started = self._clock.now()
        while held := self._held_keys():
            if self._clock.now() - started >= TRIGGER_WAIT_S:
                log_event(
                    logger,
                    "trigger_still_down",
                    level=logging.WARNING,
                    vk=self.trigger_vk,
                    held=",".join(f"0x{vk:02X}" for vk in held),
                )
                return
            pump_until(self._api, self._clock, TRIGGER_POLL_S)

    def _send(self, inputs: Sequence[win32.INPUT]) -> bool:
        """Send one batch. On a short count, release the modifiers we may have pressed."""
        sent = int(self._api.SendInput(inputs))
        if sent >= len(inputs):
            return True
        log_event(logger, "send_input_short", level=logging.WARNING, sent=sent, total=len(inputs))
        release = [win32.key_input(vk, up=True) for vk in (win32.VK_CONTROL, win32.VK_SHIFT)]
        with contextlib.suppress(OSError):
            self._api.SendInput(release)
        return False

    @staticmethod
    def _send_failure(exc: OSError | None) -> InjectStatus:
        denied = exc is not None and getattr(exc, "winerror", None) == win32.ERROR_ACCESS_DENIED
        return InjectStatus.FAILED_ELEVATED if denied else InjectStatus.FAILED_UNCONFIRMED

    # -- paste -------------------------------------------------------------------------

    def _paste(self, text: str, s: InjectionSettings) -> InjectResult:
        owner = self.clipboard
        try:
            snapshot = owner.offer(text)
        except OSError:
            snapshot = None
        if snapshot is None:
            log_event(logger, "inject_failed", status=InjectStatus.FAILED_CLIPBOARD_LOCKED)
            return InjectResult(InjectStatus.FAILED_CLIPBOARD_LOCKED)

        error: OSError | None = None
        try:
            sent = self._send(paste_inputs())
        except OSError as exc:
            sent, error = False, exc
        if not sent:
            return self._fail(text, self._send_failure(error), step="send_input")

        if not owner.wait_render(s.paste_confirm_timeout_ms / 1000):
            return self._fail(
                text, InjectStatus.FAILED_UNCONFIRMED, timeout_ms=s.paste_confirm_timeout_ms
            )

        owner.settle(s.restore_delay_ms / 1000)
        restored, partial = owner.restore(snapshot)
        if not restored:
            log_event(logger, "clipboard_restore_skipped", owner_changed=not owner.owns())
        return InjectResult(
            InjectStatus.INSERTED,
            text_on_clipboard=not restored and owner.owns(),
            clipboard_restored=restored,
            restore_partial=restored and partial,
        )

    # -- type --------------------------------------------------------------------------

    def _type(self, text: str, s: InjectionSettings, target_hwnd: int) -> InjectResult:
        batches = type_chunks(text, s.type_chunk_chars)
        for index, batch in enumerate(batches):
            if index:
                pump_until(self._api, self._clock, TYPE_CHUNK_GAP_S)
                if self._api.GetForegroundWindow() != target_hwnd:
                    return self._fail(
                        text, InjectStatus.PARTIAL, chunks=index, total_chunks=len(batches)
                    )
            error: OSError | None = None
            try:
                sent = self._send(batch)
            except OSError as exc:
                sent, error = False, exc
            if not sent:
                status = InjectStatus.PARTIAL if index else self._send_failure(error)
                return self._fail(text, status, chunks=index, total_chunks=len(batches))
        return InjectResult(InjectStatus.INSERTED)
