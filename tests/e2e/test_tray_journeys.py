"""Journeys that need tray actions, with `build_app()` in-process (E2E-011, E2E-012,
E2E-014, E2E-015, E2E-018, E2E-021)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from talktype import win32
from talktype.config import set_value
from talktype.paths import Paths
from talktype.strings import Msg
from tests import desktop
from tests.e2e.conftest import (
    RCTRL,
    AppProcess,
    InProcess,
    TextHarness,
    configure,
    contains_terms,
    duration_ms,
    fixture_wav,
    history,
    hold_trigger,
    key_terms,
    normalize,
    wait_text,
)

pytestmark = pytest.mark.e2e

Build = Callable[..., InProcess]
Launch = Callable[..., AppProcess]
PREOPEN_MS = 300


def speak_ms(name: str) -> int:
    return duration_ms(name) - PREOPEN_MS + 400


def has_hum(text: str) -> bool:
    return "hum" in normalize(text).split()


# --------------------------------------------------------------------------------------
# E2E-011: cleanup toggle from the tray
# --------------------------------------------------------------------------------------


def test_e2e011_cleanup_toggle(home: Path, inprocess: Build, harness: TextHarness) -> None:
    configure(home, cleanup__enabled=True)
    ui = inprocess(home, fixture_wav("hesitacao"))
    harness.open()
    hold_trigger(ui.log, speak_ms("hesitacao"))
    cleaned = wait_text(harness, lambda t: contains_terms(t, key_terms("hesitacao")))
    ui.log.wait("dictation_finished", status="inserted")
    assert not has_hum(cleaned), cleaned

    ui.action(Msg.MENU_CLEANUP.text).trigger()
    assert ui.app.config.settings.cleanup.enabled is False
    harness.clear()
    harness.open()
    hold_trigger(ui.log, speak_ms("hesitacao"))
    verbatim = wait_text(harness, lambda t: contains_terms(t, key_terms("hesitacao")))
    ui.log.wait("dictation_finished", status="inserted", id="2")
    assert has_hum(verbatim), verbatim


# --------------------------------------------------------------------------------------
# E2E-012: AI rewrite (needs Ollama)
# --------------------------------------------------------------------------------------


def ollama_ready() -> bool:
    try:
        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2, trust_env=False)
    except httpx.HTTPError:
        return False
    return any(m.get("name") == "qwen2.5:7b-instruct" for m in tags.json().get("models", []))


def test_e2e012_ai_rewrite_and_its_fallback(
    home: Path, tmp_path: Path, launch: Launch, harness: TextHarness
) -> None:
    if not ollama_ready():
        pytest.fail("E2E-012 needs Ollama with qwen2.5:7b-instruct on 127.0.0.1:11434")
    # Cleanup off, so removing "hum" is the rewrite's job.
    configure(home, llm__enabled=True, cleanup__enabled=False)
    app = launch(home, fixture_wav("hesitacao"))
    harness.open()
    hold_trigger(app.log, speak_ms("hesitacao"))
    finished = app.log.wait("dictation_finished", timeout_s=40, status="inserted")
    text = wait_text(harness, lambda t: bool(t.strip()))
    assert int(finished["llm_ms"]) > 0, finished
    assert "rewrite_skipped" not in finished, finished
    assert not has_hum(text), text
    assert contains_terms(text, ["revisar", "amanhã"]), text
    app.stop()

    closed = tmp_path / "home-closed-port"
    configure(closed, llm__enabled=True, llm__endpoint="http://127.0.0.1:9")
    app = launch(closed, fixture_wav("hesitacao"))
    harness.clear()
    harness.open()
    hold_trigger(app.log, speak_ms("hesitacao"))
    app.log.wait("rewrite_skipped", timeout_s=30, reason="unreachable")
    app.log.wait("dictation_finished", timeout_s=30, status="inserted")
    text = wait_text(harness, lambda t: contains_terms(t, key_terms("hesitacao")))
    assert contains_terms(text, key_terms("hesitacao"))


# --------------------------------------------------------------------------------------
# E2E-014: history insert and the re-paste hotkey
# --------------------------------------------------------------------------------------


def history_insert_action(ui: InProcess, index: int = 0) -> Any:
    tray = ui.app.tray
    tray.rebuild_history()
    submenus = [a.menu() for a in tray.history_menu.actions() if isinstance(a.menu(), QMenu)]
    menu = submenus[index]
    assert menu is not None
    for action in menu.actions():
        if action.text() == Msg.MENU_HISTORY_INSERT.text:
            return action
    raise AssertionError("no Inserir action")


def test_e2e014_history_insert_then_repaste(home: Path, inprocess: Build, qtbot: Any) -> None:
    ui = inprocess(home, fixture_wav("curta_sim"))
    first = TextHarness()
    qtbot.addWidget(first)
    first.open()
    desktop.down(RCTRL)
    try:
        ui.log.wait("dictation_committed", timeout_s=4)
        desktop.pump(speak_ms("curta_sim"))
        first.close()
        desktop.focus_desktop()
        desktop.pump(200)
    finally:
        desktop.up(RCTRL)
    ui.log.wait("dictation_finished", status="failed_insert")
    transcript = history(home)[0]["text"]
    assert "sim" in normalize(transcript)

    target = TextHarness("talktype e2e history")
    qtbot.addWidget(target)
    target.open()
    history_insert_action(ui).trigger()
    wait_text(target, lambda t: t == transcript)

    desktop.chord(win32.VK_LCONTROL, win32.VK_LMENU, win32.VK_LSHIFT, ord("V"))
    wait_text(target, lambda t: t == transcript * 2)
    desktop.pump(500)
    assert target.text() == transcript * 2


# --------------------------------------------------------------------------------------
# E2E-015: pause and resume from the tray
# --------------------------------------------------------------------------------------


def test_e2e015_pause_makes_right_ctrl_normal(
    home: Path, inprocess: Build, harness: TextHarness
) -> None:
    ui = inprocess(home, fixture_wav("curta_sim"))
    ui.action(Msg.MENU_PAUSE.text).trigger()
    assert ui.app.tray.state.value == "paused"
    harness.open()
    desktop.down(RCTRL)
    desktop.pump(2500)
    desktop.up(RCTRL)
    desktop.pump(300)
    assert ui.log.events("dictation_committed") == []
    assert harness.keys.pressed(Qt.Key.Key_Control)
    assert harness.keys.released(Qt.Key.Key_Control)

    ui.action(Msg.MENU_RESUME.text).trigger()
    assert ui.app.tray.state.value == "ready"
    harness.open()
    hold_trigger(ui.log, speak_ms("curta_sim"))
    text = wait_text(harness, lambda t: "sim" in normalize(t))
    assert "sim" in normalize(text)


# --------------------------------------------------------------------------------------
# E2E-018: export and import
# --------------------------------------------------------------------------------------


def test_e2e018_export_then_import(tmp_path: Path, inprocess: Build) -> None:
    home_a = tmp_path / "home-a"
    configure(home_a)
    assert set_value(Paths(home_a).config, "asr.vocabulary", ["Kubernetes"]).saved
    ui_a = inprocess(home_a, ready=False)
    ui_a.action(Msg.MENU_RELOAD_CONFIG.text).trigger()
    exported = tmp_path / "x.toml"
    ui_a.app.tray.export_requested.emit(exported)  # what "Exportar config" sends
    assert "Kubernetes" in exported.read_text(encoding="utf-8")
    ui_a.app.shutdown()

    home_b = tmp_path / "home-b"
    configure(home_b)
    ui_b = inprocess(home_b, ready=False)
    ui_b.app.tray.import_requested.emit(exported)  # what "Importar config" sends
    assert "Kubernetes" in Paths(home_b).config.read_text(encoding="utf-8")
    assert list(home_b.glob("config.backup-*.toml")), "no backup of the previous config"
    assert ui_b.app.config.settings.asr.vocabulary == ["Kubernetes"]


# --------------------------------------------------------------------------------------
# E2E-021: model switch through "Recarregar config"
# --------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_e2e021_switch_to_small_and_dictate(
    home: Path, inprocess: Build, harness: TextHarness
) -> None:
    ui = inprocess(home, fixture_wav("deploy_pr"))
    assert set_value(Paths(home).config, "asr.model", "small").saved
    ui.action(Msg.MENU_RELOAD_CONFIG.text).trigger()
    ui.log.wait("model_loaded", timeout_s=600, fail_on="model_load_failed", model="small")
    harness.open()
    hold_trigger(ui.log, speak_ms("deploy_pr"))
    text = wait_text(harness, lambda t: "deploy" in normalize(t), timeout_s=30)
    assert "deploy" in normalize(text)
