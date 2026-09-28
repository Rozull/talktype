"""Tray status, icon states and submenus (UT-231, UT-235, UT-244, UT-245)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu
from pytestqt.qtbot import QtBot

from talktype.asr_device import Device
from talktype.config import ConfigManager
from talktype.history import HistoryEntry, HistoryStatus, HistoryStore
from talktype.strings import Msg
from talktype.tray import IconShape, Tray, TrayState, render_icon

pytestmark = pytest.mark.unit


class RecordingInjector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def inject(self, text: str, mode: str) -> None:
        self.calls.append((text, mode))


class Env:
    def __init__(self, home: Path) -> None:
        self.config = ConfigManager(home)
        self.history = HistoryStore(home)
        self.devices = ["Microfone (Realtek)", "Headset Jabra"]
        self.autostart = False
        self.clipboard: list[str] = []
        self.save_path: Path | None = None
        self.open_path: Path | None = None
        self.tray = Tray(
            self.config,
            history=self.history.entries,
            devices=lambda: self.devices,
            autostart_enabled=lambda: self.autostart,
            copy_text=self.clipboard.append,
            save_dialog=lambda: self.save_path,
            open_dialog=lambda: self.open_path,
        )


@pytest.fixture
def env(qtbot: QtBot, home: Path) -> Env:
    del qtbot  # only needed for the QApplication
    return Env(home)


def _texts(menu: QMenu) -> list[str]:
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _action(menu: QMenu, text: str) -> QAction:
    return next(a for a in menu.actions() if a.text() == text)


def _submenu(menu: QMenu, text: str) -> QMenu:
    sub = _action(menu, text).menu()
    assert isinstance(sub, QMenu)
    return sub


def test_ut231_status_label_names_model_and_device(env: Env) -> None:
    env.tray.set_status("large-v3-turbo", Device("cuda", "int8_float16"))
    assert env.tray.status_text == "Modelo: large-v3-turbo · GPU"
    assert env.tray.menu.actions()[0].text() == "Modelo: large-v3-turbo · GPU"

    env.tray.set_status("large-v3-turbo", Device("cpu", "int8"))
    assert env.tray.status_text == "Modelo: large-v3-turbo · CPU"
    env.tray.set_status("parakeet-v3", "cpu")
    assert env.tray.status_text.endswith("· CPU")


def test_ut235_each_state_has_tooltip_and_distinct_icon_shape(env: Env) -> None:
    cases = [
        (TrayState.LOADING, {}, "Carregando modelo…"),
        (TrayState.READY, {}, "Pronto"),
        (TrayState.PAUSED, {}, "Pausado"),
        (TrayState.ERROR, {"reason": "modelo corrompido"}, "Erro: modelo corrompido"),
    ]
    shapes: list[IconShape] = []
    for state, kwargs, text in cases:
        env.tray.set_state(state, **kwargs)
        assert text in env.tray.toolTip()
        assert env.tray.state_action.text() == text
        shapes.append(env.tray.icon_shape)
    assert len(set(shapes)) == 4

    images = [render_icon(shape).toImage() for shape in shapes]
    for i, image in enumerate(images):
        for other in images[i + 1 :]:
            assert image != other

    env.tray.set_state(TrayState.DOWNLOADING, percent=42)
    assert "Baixando modelo… 42%" in env.tray.toolTip()
    assert env.tray.icon_shape is IconShape.RING


def test_ut235_pause_entry_follows_state(env: Env) -> None:
    requests: list[bool] = []
    env.tray.pause_requested.connect(requests.append)
    env.tray.set_state(TrayState.READY)
    env.tray.pause_action.trigger()
    env.tray.set_state(TrayState.PAUSED)
    assert env.tray.pause_action.text() == Msg.MENU_RESUME.text
    env.tray.pause_action.trigger()
    assert requests == [True, False]


def test_ut244_history_entry_offers_copy_and_insert(env: Env) -> None:
    injector = RecordingInjector()
    env.tray.history_insert_requested.connect(lambda text: injector.inject(text, "paste"))
    env.history.add_pending("texto um")

    env.tray.history_menu.aboutToShow.emit()
    entry_menu = _submenu(env.tray.history_menu, "texto um")
    assert _texts(entry_menu) == ["Copiar", "Inserir"]

    _action(entry_menu, "Copiar").trigger()
    assert env.clipboard == ["texto um"]
    _action(entry_menu, "Inserir").trigger()
    assert injector.calls == [("texto um", "paste")]


def test_ut244_failed_entry_is_marked_and_clear_is_offered(env: Env) -> None:
    entry_id = env.history.add_pending("não entrou")
    env.history.mark(entry_id, HistoryStatus.FAILED_INSERT)
    cleared: list[bool] = []
    env.tray.history_clear_requested.connect(lambda: cleared.append(True))

    env.tray.rebuild_history()
    texts = _texts(env.tray.history_menu)
    assert texts == ["não entrou (não inserido)", "Limpar histórico"]
    _action(env.tray.history_menu, "Limpar histórico").trigger()
    assert cleared == [True]


def test_ut245_empty_history_and_refresh_on_open(env: Env) -> None:
    env.tray.history_menu.aboutToShow.emit()
    [empty] = env.tray.history_menu.actions()
    assert empty.text() == "Nenhum ditado ainda"
    assert not empty.isEnabled()

    env.history.add_pending("ditado novo")  # added while the menu is closed
    assert _texts(env.tray.history_menu) == ["Nenhum ditado ainda"]
    env.tray.history_menu.aboutToShow.emit()
    assert _texts(env.tray.history_menu)[0] == "ditado novo"


def test_history_labels_are_single_line_and_escape_ampersands(env: Env) -> None:
    env.history.add_pending("P&D\nsegunda linha " + "x" * 80)
    env.tray.rebuild_history()
    title = _texts(env.tray.history_menu)[0]
    assert title.startswith("P&&D segunda linha")
    assert title.endswith("…")


def test_microphone_submenu_lists_devices_and_persists_choice(env: Env) -> None:
    changes: list[tuple[str, Any, bool]] = []
    env.tray.setting_changed.connect(lambda k, v, s: changes.append((k, v, s)))

    env.tray.microphone_menu.aboutToShow.emit()
    menu = env.tray.microphone_menu
    assert _texts(menu) == ["Padrão do Windows", "Microfone (Realtek)", "Headset Jabra"]
    assert _action(menu, "Padrão do Windows").isChecked()

    _action(menu, "Headset Jabra").trigger()
    assert env.config.settings.recording.microphone == "Headset Jabra"
    assert changes == [("recording.microphone", "Headset Jabra", True)]
    assert 'microphone = "Headset Jabra"' in env.config.path.read_text(encoding="utf-8")

    env.devices = ["Microfone (Realtek)"]  # the pinned headset was unplugged
    env.tray.microphone_menu.aboutToShow.emit()
    assert _texts(menu) == ["Padrão do Windows", "Microfone (Realtek)", "Headset Jabra"]
    assert _action(menu, "Headset Jabra").isChecked()


def test_toggle_guard_can_veto_a_change(qtbot: QtBot, home: Path) -> None:
    del qtbot
    config = ConfigManager(home)
    tray = Tray(
        config,
        history=list[HistoryEntry],
        devices=list[str],
        autostart_enabled=lambda: False,
        toggle_guard=lambda key, value: key != "llm.enabled",
    )
    tray.rewrite_action.trigger()
    assert config.settings.llm.enabled is False
    assert not tray.rewrite_action.isChecked()


def test_autostart_entry_reflects_the_real_state(env: Env) -> None:
    requests: list[bool] = []

    def handle(enable: bool) -> None:
        requests.append(enable)  # e.g. blocked by policy: the state stays off

    env.tray.autostart_requested.connect(handle)
    env.tray.autostart_action.trigger()
    assert requests == [True]
    assert not env.tray.autostart_action.isChecked()

    env.autostart = True
    env.tray.menu.aboutToShow.emit()
    assert env.tray.autostart_action.isChecked()


def test_export_and_import_use_the_file_dialogs(env: Env, tmp_path: Path) -> None:
    exported: list[Path] = []
    imported: list[Path] = []
    env.tray.export_requested.connect(exported.append)
    env.tray.import_requested.connect(imported.append)

    _action(env.tray.menu, "Exportar config").trigger()  # dialog cancelled
    _action(env.tray.menu, "Importar config").trigger()
    assert exported == imported == []

    env.save_path = tmp_path / "out.toml"
    env.open_path = tmp_path / "in.toml"
    _action(env.tray.menu, "Exportar config").trigger()
    _action(env.tray.menu, "Importar config").trigger()
    assert exported == [env.save_path]
    assert imported == [env.open_path]


def test_simple_actions_emit_signals(env: Env) -> None:
    fired: list[str] = []
    for text, signal in (
        ("Abrir config", env.tray.open_config_requested),
        ("Recarregar config", env.tray.reload_config_requested),
        ("Abrir pasta de config", env.tray.open_config_folder_requested),
        ("Tentar baixar novamente", env.tray.retry_download_requested),
        ("Sair", env.tray.quit_requested),
    ):
        signal.connect(lambda text=text: fired.append(text))
        _action(env.tray.menu, text).trigger()
    assert fired == [
        "Abrir config",
        "Recarregar config",
        "Abrir pasta de config",
        "Tentar baixar novamente",
        "Sair",
    ]
    assert not env.tray.retry_action.isVisible()
    env.tray.set_retry_visible(True)
    assert env.tray.retry_action.isVisible()
