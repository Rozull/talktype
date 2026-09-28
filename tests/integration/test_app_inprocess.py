"""In-process App with `FakeEngine` and the real injector (IT-043)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QTextEdit

from talktype import seams
from talktype.app import App, AppState, build_app
from talktype.audio import Recorder
from talktype.strings import Msg
from talktype.trigger import Effect
from tests import desktop
from tests.conftest import FIXTURES_AUDIO
from tests.fakes import FakeEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def app(qtbot: Any, home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[App]:
    del qtbot
    monkeypatch.setenv("TALKTYPE_TEST_NO_MSGBOX", "1")
    seams.current.cache_clear()
    engine = FakeEngine("large-v3-turbo", "cpu", text="teste")
    recorder = Recorder(audio_source=FIXTURES_AUDIO / "deploy_pr.wav")
    built = build_app(home, engine=engine, recorder=recorder, show_tray=False)
    built.start()
    try:
        desktop.wait_for(lambda: built.state is AppState.READY, 20, message="not READY")
        built.config.set_value("cleanup.final_period", False)  # exact texts below
        yield built
    finally:
        built.shutdown()
        desktop.release_modifiers()
        seams.current.cache_clear()


def dictate(app: App, editor: QTextEdit, expected: str) -> None:
    app.handle_effects((Effect.OPEN_MIC, Effect.COMMIT_HOLD))
    desktop.pump(1200)  # speech from the WAV seam, so the VAD finds some
    app.handle_effect(Effect.FINISH)
    try:
        desktop.wait_for(lambda: editor.toPlainText() == expected, 15)
    except AssertionError:
        raise AssertionError(f"editor text is {editor.toPlainText()!r}") from None


def test_it043_tray_mode_switch(app: App, qtbot: Any) -> None:
    editor = QTextEdit()
    qtbot.addWidget(editor)
    editor.show()
    desktop.focus(int(editor.winId()))
    editor.setFocus()
    desktop.pump(150)

    typed = app.tray.find_action(Msg.MENU_MODE_TYPE.text)
    assert typed is not None
    typed.trigger()
    assert app.config.settings.injection.mode == "type"
    desktop.set_clipboard_text("sentinela")
    sequence = desktop.clipboard_sequence()
    dictate(app, editor, "teste")
    assert desktop.clipboard_sequence() == sequence  # typed: the clipboard was untouched
    assert app.history.entries()[0].status == "inserted"

    pasted = app.tray.find_action(Msg.MENU_MODE_PASTE.text)
    assert pasted is not None
    pasted.trigger()
    assert app.config.settings.injection.mode == "paste"
    dictate(app, editor, "testeteste")
    assert desktop.clipboard_sequence() != sequence  # pasted through the clipboard
    desktop.pump(500)
    assert desktop.clipboard_text() == "sentinela"  # and restored afterwards
