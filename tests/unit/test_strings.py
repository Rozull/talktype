"""Portuguese string catalog completeness (UT-207)."""

from __future__ import annotations

import pytest

from talktype.strings import Msg, placeholders

pytestmark = pytest.mark.unit

# Every `Msg` member the app relies on.
NAMED_IN_SPECS = [
    "APP_ERROR",
    "AUTOSTART_BLOCKED",
    "CANCELLED",
    "CLIPBOARD_PARTIAL_RESTORE",
    "CONFIG_UNREADABLE",
    "DEVICE_FALLBACK_CPU",
    "DEVICE_FORCED_FALLBACK",
    "DOWNLOAD_BLOCKED",
    "DOWNLOAD_NO_SPACE",
    "DOWNLOAD_OFFLINE",
    "EXPORT_FAILED",
    "GPU_FALLBACK_CPU",
    "HISTORY_CORRUPT",
    "HOOK_BLOCKED",
    "HOTKEY_CONFLICT",
    "INSERT_FAILED_OPEN_HISTORY",
    "INTERRUPTED",
    "LLM_ENDPOINT_NOT_LOCAL",
    "MIC_LOST",
    "MODEL_CACHE_NOT_WRITABLE",
    "MODEL_DOWNLOADING",
    "MODEL_LOADING",
    "MODEL_SWITCH_FAILED",
    "NOTHING_DETECTED",
    "NOTHING_DETECTED_MUTED_HINT",
    "NOTHING_TO_REPASTE",
    "ONBOARDING",
    "SOUND_MISSING",
    "VOCAB_IGNORED",
]

# Literal Portuguese texts that must not change.
PINNED_TEXTS = {
    Msg.INSERTED: "inserido",
    Msg.NOTHING_DETECTED: "nada detectado",
    Msg.CANCELLED: "cancelado",
    Msg.INTERRUPTED: "interrompido",
    Msg.STATE_TRANSCRIBING: "transcrevendo",
    Msg.STATE_INSERTING: "inserindo",
    Msg.INSERT_FAILED_CLIPBOARD: "não consegui colar — o texto está no clipboard (Ctrl+V)",
    Msg.REWRITE_SKIPPED: "revisão por IA ignorada",
    Msg.ALREADY_RUNNING: "talktype já está em execução",
    Msg.AUTOSTART_BLOCKED: "não foi possível ativar a inicialização automática",
    Msg.HISTORY_EMPTY: "Nenhum ditado ainda",
    Msg.TRAY_LOADING: "Carregando modelo…",
    Msg.TRAY_READY: "Pronto",
    Msg.TRAY_PAUSED: "Pausado",
    Msg.MENU_MODE_PASTE: "Modo: colar",
    Msg.MENU_MODE_TYPE: "Modo: digitar",
    Msg.MENU_PREVIEW: "Preview ao vivo",
    Msg.MENU_CLEANUP: "Limpeza",
    Msg.MENU_COMMANDS: "Comandos falados",
    Msg.MENU_REWRITE: "Revisão por IA",
    Msg.MENU_PAUSE: "Pausar",
    Msg.MENU_RESUME: "Retomar",
    Msg.MENU_AUTOSTART: "Iniciar com o Windows",
    Msg.MENU_MICROPHONE: "Microfone",
    Msg.MENU_HISTORY: "Histórico",
    Msg.MENU_HISTORY_COPY: "Copiar",
    Msg.MENU_HISTORY_INSERT: "Inserir",
    Msg.MENU_OPEN_CONFIG: "Abrir config",
    Msg.MENU_RELOAD_CONFIG: "Recarregar config",
    Msg.MENU_OPEN_CONFIG_FOLDER: "Abrir pasta de config",
    Msg.MENU_EXPORT_CONFIG: "Exportar config",
    Msg.MENU_IMPORT_CONFIG: "Importar config",
    Msg.MENU_RETRY_DOWNLOAD: "Tentar baixar novamente",
    Msg.MENU_QUIT: "Sair",
}


def test_ut207_every_member_has_a_portuguese_template_matching_its_kwargs() -> None:
    for msg in Msg:
        assert msg.text.strip(), msg.name
        assert msg.text == msg.text.strip(), msg.name
        assert placeholders(msg.text) == set(msg.params), msg.name
        rendered = msg.format(**dict.fromkeys(msg.params, "X"))
        assert "{" not in rendered, msg.name


def test_ut207_format_rejects_wrong_kwargs() -> None:
    assert Msg.HOTKEY_CONFLICT.format(hotkey="ctrl+alt+shift+v") == (
        "o atalho ctrl+alt+shift+v já está em uso por outro programa"
    )
    with pytest.raises(TypeError):
        Msg.HOTKEY_CONFLICT.format()
    with pytest.raises(TypeError):
        Msg.INSERTED.format(extra=1)


def test_ut207_catalog_covers_every_name_in_the_specs() -> None:
    missing = [name for name in NAMED_IN_SPECS if name not in Msg.__members__]
    assert missing == []


def test_ut207_pinned_texts() -> None:
    for msg, text in PINNED_TEXTS.items():
        assert msg.text == text, msg.name
    assert "janelas de administrador" in Msg.INSERT_FAILED_ELEVATED.text
    assert Msg.TRAY_STATUS.format(model="large-v3-turbo", device="GPU") == (
        "Modelo: large-v3-turbo · GPU"
    )
    assert Msg.TRAY_ERROR.format(reason="x") == "Erro: x"
