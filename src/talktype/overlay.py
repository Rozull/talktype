"""`Overlay`: the non-activating, click-through recording pill.

The pill appears at the bottom center of the monitor that holds the foreground window and
stays where it opened until the dictation ends. It shows the state, a live
level meter, the remaining time during the last 30 s, the newest live preview and the
outcome. It never takes focus and never receives mouse input: besides the Qt window flags,
the native window gets ``WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED`` and is
kept topmost. The injector excludes `Overlay.hwnd` from its focus probe. Qt GUI thread only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol

from PySide6.QtCore import QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from talktype import win32
from talktype.strings import Msg

Rect = tuple[int, int, int, int]  # (left, top, right, bottom) in native pixels

WIDTH = 440
HEIGHT = 76
BOTTOM_MARGIN = 48
PAD_X = 20
PAD_Y = 10
PREVIEW_WIDTH = WIDTH - 2 * PAD_X
COUNTDOWN_FROM_S = 30
INSERTING_AFTER_MS = 500
OUTCOME_MS = 1800
_POLL_MS = 50

NATIVE_EX_STYLE = win32.WS_EX_NOACTIVATE | win32.WS_EX_TRANSPARENT | win32.WS_EX_LAYERED


class OverlayState(StrEnum):
    HIDDEN = "hidden"
    RECORDING = "recording"
    HANDSFREE = "handsfree"
    TRANSCRIBING = "transcribing"
    INSERTING = "inserting"
    OUTCOME = "outcome"


_STATE_MESSAGES = {
    OverlayState.RECORDING: Msg.STATE_RECORDING,
    OverlayState.HANDSFREE: Msg.STATE_HANDSFREE,
    OverlayState.TRANSCRIBING: Msg.STATE_TRANSCRIBING,
    OverlayState.INSERTING: Msg.STATE_INSERTING,
}

# Outcome statuses (`DictationState`, `InjectStatus`, `AsrOutcome.kind`, rewrite) -> text.
OUTCOME_MESSAGES: dict[str, Msg] = {
    "inserted": Msg.INSERTED,
    "no_speech": Msg.NOTHING_DETECTED,
    "cancelled": Msg.CANCELLED,
    "interrupted": Msg.INTERRUPTED,
    "error": Msg.ASR_ERROR,
    "failed_insert": Msg.INSERT_FAILED_CLIPBOARD,
    "failed_unconfirmed": Msg.INSERT_FAILED_CLIPBOARD,
    "failed_no_focus": Msg.INSERT_FAILED_NO_FOCUS,
    "failed_elevated": Msg.INSERT_FAILED_ELEVATED,
    "failed_clipboard_locked": Msg.INSERT_FAILED_OPEN_HISTORY,
    "partial": Msg.INSERT_PARTIAL,
    "rewrite_skipped": Msg.REWRITE_SKIPPED,
}


class MsClock(Protocol):
    def now_ms(self) -> int: ...


class MonotonicClock:
    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


def _overlap(a: Rect, b: Rect) -> int:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0) * max(height, 0)


def pick_screen(fg_rect: Rect | None, screens: Sequence[Rect], primary: int = 0) -> int:
    """Index of the screen holding the foreground window's center (else the most overlap)."""
    if fg_rect is None or not screens:
        return primary
    cx, cy = (fg_rect[0] + fg_rect[2]) // 2, (fg_rect[1] + fg_rect[3]) // 2
    for index, (left, top, right, bottom) in enumerate(screens):
        if left <= cx < right and top <= cy < bottom:
            return index
    overlaps = [_overlap(fg_rect, screen) for screen in screens]
    best = max(range(len(screens)), key=overlaps.__getitem__)
    return best if overlaps[best] > 0 else primary


def elide_words(text: str, metrics: QFontMetrics, width: int) -> str:
    """The latest words of `text` that fit in `width` pixels, prefixed with "…"."""
    words = text.split()
    full = " ".join(words)
    if metrics.horizontalAdvance(full) <= width:
        return full
    kept: list[str] = []
    for word in reversed(words):
        candidate = "… " + " ".join([word, *kept])
        if metrics.horizontalAdvance(candidate) > width:
            break
        kept.insert(0, word)
    if not kept:
        return metrics.elidedText(words[-1], Qt.TextElideMode.ElideLeft, width)
    return "… " + " ".join(kept)


def countdown_text(remaining_s: float | None) -> str:
    """ "m:ss" during the last 30 s of the recording limit, otherwise empty."""
    if remaining_s is None or remaining_s > COUNTDOWN_FROM_S:
        return ""
    seconds = max(math.ceil(remaining_s), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def foreground_rect(exclude: int = 0) -> Rect | None:
    """Native rectangle of the foreground window, or None (no window, or the overlay)."""
    try:
        hwnd = win32.GetForegroundWindow()
        if not hwnd or hwnd == exclude:
            return None
        return win32.GetWindowRect(hwnd)
    except OSError:
        return None


# --------------------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------------------


class LevelMeter(QWidget):
    """Five bars that light up with the input level (0..1)."""

    BARS = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.level = 0.0
        self.setFixedSize(34, 22)

    def set_level(self, level: float) -> None:
        level = min(max(level, 0.0), 1.0)
        if level != self.level:
            self.level = level
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_w = self.width() / (self.BARS * 2 - 1)
        lit = round(self.level * self.BARS)
        for index in range(self.BARS):
            height = self.height() * (0.35 + 0.65 * (index + 1) / self.BARS)
            rect = QRectF(index * 2 * bar_w, self.height() - height, bar_w, height)
            color = QColor("#ff5252") if index < lit else QColor(255, 255, 255, 70)
            painter.fillRect(rect, color)
        painter.end()


class Overlay(QWidget):
    def __init__(
        self,
        *,
        clock: MsClock | None = None,
        foreground: Callable[[], Rect | None] | None = None,
        outcome_ms: int = OUTCOME_MS,
    ) -> None:
        super().__init__(None)
        self._clock: MsClock = MonotonicClock() if clock is None else clock
        self._foreground = foreground or (lambda: foreground_rect(exclude=self.hwnd))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(WIDTH, HEIGHT)

        self.state = OverlayState.HIDDEN
        self.dictation_id: int | None = None
        self._last_seq = -1
        self._inject_started_ms: int | None = None

        self._meter = LevelMeter(self)
        self._state_label = QLabel(self)
        self._countdown_label = QLabel(self)
        self._preview_label = QLabel(self)
        state_font = QFont(self.font())
        state_font.setPointSizeF(10.5)
        state_font.setBold(True)
        self._state_label.setFont(state_font)
        self._countdown_label.setFont(state_font)
        preview_font = QFont(self.font())
        preview_font.setPointSizeF(10.0)
        self._preview_label.setFont(preview_font)
        self._preview_label.setFixedWidth(PREVIEW_WIDTH)
        for label in (self._state_label, self._countdown_label, self._preview_label):
            label.setStyleSheet("color: #ffffff; background: transparent;")
            label.setTextFormat(Qt.TextFormat.PlainText)
        self._countdown_label.setStyleSheet("color: #ffd54f; background: transparent;")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)
        top.addWidget(self._meter)
        top.addWidget(self._state_label, 1)
        top.addWidget(self._countdown_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PAD_X, PAD_Y, PAD_X, PAD_Y)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._preview_label)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(outcome_ms)
        self._hide_timer.timeout.connect(self.hide_now)
        self._insert_timer = QTimer(self)
        self._insert_timer.setInterval(_POLL_MS)
        self._insert_timer.timeout.connect(self.poll_inserting)

    # -- read-only views (tests and logs) ------------------------------------------------

    @property
    def hwnd(self) -> int:
        """The native window handle, which the injector excludes from its focus probe."""
        return int(self.winId())

    @property
    def state_text(self) -> str:
        return self._state_label.text()

    @property
    def preview_text(self) -> str:
        return self._preview_label.text()

    @property
    def countdown_text(self) -> str:
        return self._countdown_label.text()

    @property
    def level(self) -> float:
        return self._meter.level

    # -- dictation lifecycle -------------------------------------------------------------

    def show_recording(self, dictation_id: int, *, handsfree: bool = False) -> None:
        """A fresh overlay for a new dictation, placed on the foreground window's screen."""
        self._hide_timer.stop()
        self._insert_timer.stop()
        self._inject_started_ms = None
        self.dictation_id = dictation_id
        self._last_seq = -1
        self._preview_label.clear()
        self._countdown_label.clear()
        self._meter.set_level(0.0)
        self._set_state(OverlayState.HANDSFREE if handsfree else OverlayState.RECORDING)
        self._present(reposition=True)

    def set_level(self, level: float) -> None:
        self._meter.set_level(level)

    def set_remaining(self, remaining_s: float | None) -> None:
        """Seconds left before the recording limit; shown only in the last 30 s."""
        self._countdown_label.setText(countdown_text(remaining_s))

    def show_preview(self, dictation_id: int, seq: int, text: str) -> bool:
        """Show a live preview. Stale, out-of-order or late results are dropped."""
        recording = self.state in (OverlayState.RECORDING, OverlayState.HANDSFREE)
        if not recording or dictation_id != self.dictation_id or seq <= self._last_seq:
            return False
        self._last_seq = seq
        self._preview_label.setText(
            elide_words(text, self._preview_label.fontMetrics(), PREVIEW_WIDTH)
        )
        return True

    def show_transcribing(self) -> None:
        self._countdown_label.clear()
        self._meter.set_level(0.0)
        self._set_state(OverlayState.TRANSCRIBING)
        self._present()

    def inject_started(self) -> None:
        """Injection began: "inserindo" appears if it is still running after 500 ms."""
        self._inject_started_ms = self._clock.now_ms()
        self._insert_timer.start()

    def poll_inserting(self) -> None:
        if self._inject_started_ms is None:
            self._insert_timer.stop()
            return
        if self._clock.now_ms() - self._inject_started_ms >= INSERTING_AFTER_MS:
            self._insert_timer.stop()
            self._inject_started_ms = None
            self.show_inserting()

    def show_inserting(self) -> None:
        self._set_state(OverlayState.INSERTING)
        self._present()

    def show_outcome(self, status: str | Msg) -> None:
        """Show the dictation's outcome briefly, then hide."""
        message = status if isinstance(status, Msg) else OUTCOME_MESSAGES[str(status)]
        self.show_message(message.text)

    def show_message(self, text: str) -> None:
        """Any short notice (an outcome or an error), shown briefly, then hidden."""
        self._insert_timer.stop()
        self._inject_started_ms = None
        self.dictation_id = None
        self._countdown_label.clear()
        self._preview_label.clear()
        self._meter.set_level(0.0)
        self.state = OverlayState.OUTCOME
        self._state_label.setText(text)
        self._present(reposition=not self.isVisible())
        self._hide_timer.start()

    def hide_now(self) -> None:
        self._hide_timer.stop()
        self._insert_timer.stop()
        self._inject_started_ms = None
        self.state = OverlayState.HIDDEN
        self.hide()

    # -- internals -----------------------------------------------------------------------

    def _set_state(self, state: OverlayState) -> None:
        self.state = state
        self._state_label.setText(_STATE_MESSAGES[state].text)

    def _present(self, *, reposition: bool = False) -> None:
        if reposition or not self.isVisible():
            self._place()
        if not self.isVisible():
            self._apply_native_styles()
            self.show()
        self._apply_native_styles()

    def _place(self) -> None:
        screens = QGuiApplication.screens()
        if not screens:
            return
        primary = QGuiApplication.primaryScreen()
        primary_index = screens.index(primary) if primary in screens else 0
        index = pick_screen(
            self._foreground(),
            [_native_rect(s.geometry(), s.devicePixelRatio()) for s in screens],
            primary_index,
        )
        area = screens[index].availableGeometry()
        x = area.x() + (area.width() - self.width()) // 2
        y = area.y() + area.height() - self.height() - BOTTOM_MARGIN
        self.move(max(x, area.x()), max(y, area.y()))

    def _apply_native_styles(self) -> None:
        if QGuiApplication.platformName() != "windows":
            return
        hwnd = self.hwnd
        try:
            ex_style = win32.GetWindowLongPtrW(hwnd, win32.GWL_EXSTYLE)
            wanted = ex_style | NATIVE_EX_STYLE
            if wanted != ex_style:
                win32.SetWindowLongPtrW(hwnd, win32.GWL_EXSTYLE, wanted)
            win32.SetWindowPos(
                hwnd,
                win32.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32.SWP_NOMOVE | win32.SWP_NOSIZE | win32.SWP_NOACTIVATE,
            )
        except OSError:
            pass  # the Qt flags still keep the window non-activating and click-through

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 24, 28, 232))
        radius = min(self.height() / 2, 26)
        painter.drawRoundedRect(QRectF(self.rect()), radius, radius)
        painter.end()


def _native_rect(geometry: QRect, ratio: float) -> Rect:
    """A screen's native rectangle: Qt keeps the native top-left and scales the size."""
    left, top = geometry.x(), geometry.y()
    return (
        left,
        top,
        left + round(geometry.width() * ratio),
        top + round(geometry.height() * ratio),
    )
