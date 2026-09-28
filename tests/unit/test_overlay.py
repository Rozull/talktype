"""Overlay states, preview filtering and placement (UT-217, UT-218, UT-219, UT-222)."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from talktype.overlay import (
    COUNTDOWN_FROM_S,
    PREVIEW_WIDTH,
    Overlay,
    OverlayState,
    Rect,
    countdown_text,
    pick_screen,
)
from talktype.strings import Msg
from tests.conftest import FakeClock

pytestmark = pytest.mark.unit


class Foreground:
    """A settable foreground-window rectangle."""

    def __init__(self, rect: Rect | None = None) -> None:
        self.rect = rect

    def __call__(self) -> Rect | None:
        return self.rect


@pytest.fixture
def overlay(qtbot: QtBot) -> Overlay:
    widget = Overlay(foreground=Foreground(), clock=FakeClock(1000.0))
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("inserted", "inserido"),
        ("no_speech", "nada detectado"),
        ("cancelled", "cancelado"),
        ("failed_insert", "não consegui colar — o texto está no clipboard (Ctrl+V)"),
        ("failed_elevated", Msg.INSERT_FAILED_ELEVATED.text),
        ("rewrite_skipped", "revisão por IA ignorada"),
    ],
)
def test_ut217_outcome_texts(overlay: Overlay, status: str, expected: str) -> None:
    overlay.show_recording(1)
    overlay.show_outcome(status)
    assert overlay.state is OverlayState.OUTCOME
    assert overlay.state_text == expected
    if status == "failed_elevated":
        assert "janelas de administrador" in overlay.state_text


def test_ut217_outcome_hides_after_a_moment(qtbot: QtBot) -> None:
    widget = Overlay(foreground=Foreground(), outcome_ms=50)
    qtbot.addWidget(widget)
    widget.show_recording(1)
    widget.show_outcome(Msg.CANCELLED)
    assert widget.isVisible()
    qtbot.waitUntil(lambda: not widget.isVisible(), timeout=2000)
    assert widget.state is OverlayState.HIDDEN


def test_ut218_no_placeholder_and_fresh_overlay_per_dictation(overlay: Overlay) -> None:
    overlay.show_recording(1)
    assert overlay.state_text == "gravando"
    assert overlay.preview_text == ""  # nothing before the first PreviewResult

    assert overlay.show_preview(1, 0, "primeira frase")
    assert overlay.preview_text == "primeira frase"
    overlay.show_outcome("inserted")

    overlay.show_recording(2, handsfree=True)
    assert overlay.state_text == "gravando (mãos livres)"
    assert overlay.preview_text == ""
    assert overlay.level == 0.0


def test_ut218_stale_and_out_of_order_previews_are_dropped(overlay: Overlay) -> None:
    overlay.show_recording(7)
    assert overlay.show_preview(7, 2, "texto dois")
    assert not overlay.show_preview(7, 1, "texto um")  # out of order
    assert not overlay.show_preview(7, 2, "texto dois de novo")  # same seq
    assert not overlay.show_preview(6, 9, "ditado anterior")  # stale dictation
    assert overlay.preview_text == "texto dois"

    overlay.show_transcribing()
    assert not overlay.show_preview(7, 3, "tarde demais")  # the final text wins
    assert overlay.preview_text == "texto dois"


def test_ut219_pick_screen_uses_the_foreground_window_monitor() -> None:
    screens = [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]
    assert pick_screen((2000, 100, 2500, 600), screens) == 1
    assert pick_screen((100, 100, 600, 600), screens) == 0
    assert pick_screen((1800, 100, 2100, 600), screens) == 1  # center on screen 2
    assert pick_screen((-500, -500, -400, -400), screens) == 0  # off-screen: primary
    assert pick_screen(None, screens, primary=1) == 1


def test_ut219_position_stays_fixed_while_shown(qtbot: QtBot) -> None:
    foreground = Foreground((100, 100, 600, 600))
    widget = Overlay(foreground=foreground)
    qtbot.addWidget(widget)
    widget.show_recording(1)
    opened_at = widget.pos()
    screen = widget.screen().availableGeometry()
    assert screen.contains(widget.frameGeometry())
    assert abs(widget.geometry().center().x() - screen.center().x()) <= 1

    foreground.rect = (5000, 3000, 5400, 3400)  # the window moved mid-recording
    widget.set_level(0.8)
    widget.show_preview(1, 0, "olá")
    widget.set_remaining(12)
    widget.show_transcribing()
    widget.show_inserting()
    widget.show_outcome("inserted")
    assert widget.pos() == opened_at


def test_ut222_inserting_shown_after_500_ms(qtbot: QtBot) -> None:
    clock = FakeClock(1000.0)
    widget = Overlay(foreground=Foreground(), clock=clock)
    qtbot.addWidget(widget)
    widget.show_recording(1)
    widget.show_transcribing()

    widget.inject_started()
    clock.advance(0.3)
    widget.poll_inserting()
    assert widget.state is OverlayState.TRANSCRIBING

    clock.advance(0.2)
    widget.poll_inserting()
    assert widget.state is OverlayState.INSERTING
    assert widget.state_text == "inserindo"

    clock.advance(0.1)  # the injection ends at 600 ms
    widget.show_outcome("inserted")
    assert widget.state_text == "inserido"


def test_ut222_quick_injection_never_shows_inserting(overlay: Overlay) -> None:
    clock = FakeClock(0.0)
    widget = overlay
    widget._clock = clock  # pyright: ignore[reportPrivateUsage]
    widget.show_recording(1)
    widget.show_transcribing()
    widget.inject_started()
    clock.advance(0.4)
    widget.show_outcome("inserted")
    clock.advance(1.0)
    widget.poll_inserting()
    assert widget.state is OverlayState.OUTCOME


def test_level_and_countdown(overlay: Overlay) -> None:
    overlay.show_recording(1)
    overlay.set_level(1.7)
    assert overlay.level == 1.0
    overlay.set_remaining(COUNTDOWN_FROM_S + 5)
    assert overlay.countdown_text == ""
    overlay.set_remaining(29.2)
    assert overlay.countdown_text == "0:30"
    overlay.set_remaining(4)
    assert overlay.countdown_text == "0:04"
    assert countdown_text(None) == ""
    assert countdown_text(-1) == "0:00"


def test_long_preview_keeps_the_latest_words(overlay: Overlay) -> None:
    overlay.show_recording(1)
    words = [f"palavra{i}" for i in range(80)]
    overlay.show_preview(1, 0, " ".join(words))
    shown = overlay.preview_text
    assert shown.startswith("… ")
    assert shown.endswith("palavra79")
    assert shown.split()[1] in words
    metrics = overlay._preview_label.fontMetrics()  # pyright: ignore[reportPrivateUsage]
    assert metrics.horizontalAdvance(shown) <= PREVIEW_WIDTH
