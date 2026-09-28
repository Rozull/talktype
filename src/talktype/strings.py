"""Every Portuguese user-facing string, keyed by `Msg`.

Each member is declared as ``(text, *params)``: ``text`` is a `str.format` template and
``params`` lists the keyword arguments it takes. Use `Msg.format()` to render a member.
"""

from __future__ import annotations

from enum import Enum, unique
from string import Formatter
from typing import Self


@unique
class Msg(Enum):
    text: str
    params: tuple[str, ...]

    def __new__(cls, text: str, *params: str) -> Self:
        obj = object.__new__(cls)
        obj._value_ = text
        obj.text = text
        obj.params = params
        return obj

    # --- Configuration: validation and reload -------------------------------------------
    CONFIG_SYNTAX = ("linha {line}: {detail}", "line", "detail")
    CONFIG_UNKNOWN_KEY = ("chave desconhecida: {key}", "key")
    CONFIG_CAPPED = ("{key} limitado a {limit}", "key", "limit")
    CONFIG_INVALID_CHOICE = ("valor inválido; use um destes: {options}", "options")
    CONFIG_WRONG_TYPE = ("tipo inválido; esperado {expected}", "expected")
    CONFIG_OUT_OF_RANGE = (
        "fora do intervalo permitido ({minimum} a {maximum})",
        "minimum",
        "maximum",
    )
    CONFIG_INVALID_VALUE = ("valor inválido: {detail}", "detail")
    CONFIG_INVALID_LANGUAGE = ("idioma não suportado pelo Whisper: {value}", "value")
    CONFIG_INVALID_HOTKEY = ("atalho inválido: {value}", "value")
    CONFIG_EMPTY_SUBSTITUTION = ('substituição com "from" vazio',)
    CONFIG_EMPTY_COMMAND = ("comando falado com frase vazia",)
    CONFIG_NOT_A_TABLE = ("deveria ser uma seção [tabela]",)
    CONFIG_UNREADABLE = (
        "não foi possível ler a configuração ({path}); a última configuração válida continua ativa",
        "path",
    )
    CONFIG_RELOADED = ("configuração recarregada",)
    CONFIG_RELOADED_WITH_ERRORS = (
        "configuração recarregada; valores anteriores mantidos para: {details}",
        "details",
    )
    CONFIG_RELOAD_REJECTED = (
        "configuração não recarregada: {details}",
        "details",
    )
    CONFIG_SAVE_FAILED = (
        "não foi possível salvar a configuração; a alteração vale só nesta sessão",
    )
    CONFIG_EXPORTED = ("configuração exportada para {path}", "path")
    EXPORT_FAILED = ("não foi possível exportar a configuração para {path}", "path")
    CONFIG_IMPORTED = ("configuração importada; a anterior foi salva em {backup}", "backup")
    IMPORT_REJECTED = ("importação recusada: {reason}", "reason")
    IMPORT_EMPTY = ("o arquivo está vazio",)
    IMPORT_NOT_TALKTYPE = ("o arquivo não é uma configuração do talktype",)
    IMPORT_UNREADABLE = ("não foi possível ler o arquivo {path}", "path")
    LLM_ENDPOINT_NOT_LOCAL = (
        "o endereço da revisão por IA precisa ser local (127.0.0.1, localhost ou ::1); "
        "a revisão por IA foi desativada",
    )

    # --- Models, device and downloads ----------------------------------------------------
    MODEL_DOWNLOADING = ("baixando o modelo…",)
    DOWNLOAD_OFFLINE = ("sem conexão para baixar o modelo; conecte-se e tente novamente",)
    DOWNLOAD_BLOCKED = (
        "o download do modelo foi bloqueado ao acessar {host} (proxy ou certificado)",
        "host",
    )
    DOWNLOAD_NO_SPACE = (
        "espaço em disco insuficiente para baixar o modelo (são necessários {required})",
        "required",
    )
    MODEL_CACHE_NOT_WRITABLE = ("a pasta de modelos não permite gravação: {path}", "path")
    MODEL_LOADING = ("carregando modelo…",)
    MODEL_SWITCH_FAILED = (
        "não foi possível carregar {model}; o modelo anterior continua ativo",
        "model",
    )
    DEVICE_FALLBACK_CPU = ("memória de GPU insuficiente para {model}; usando a CPU", "model")
    DEVICE_FORCED_FALLBACK = (
        '{model} roda apenas na CPU; o dispositivo "cuda" foi ignorado',
        "model",
    )
    GPU_FALLBACK_CPU = ("a GPU ficou sem memória; este ditado foi transcrito na CPU",)
    GPU_RESET = ("a GPU foi reiniciada; recarregando o modelo",)
    VOCAB_IGNORED = ("o vocabulário personalizado é ignorado pelo modelo {model}", "model")
    VOCAB_TRUNCATED = (
        "vocabulário longo demais; {dropped} termos foram ignorados",
        "dropped",
    )

    # --- Microphone ----------------------------------------------------------------------
    MIC_BLOCKED = (
        "o Windows está bloqueando o microfone; libere em Configurações > Privacidade e "
        "segurança > Microfone",
    )
    MIC_NO_DEVICE = ("nenhum microfone encontrado",)
    MIC_BUSY = ("o microfone está em uso por outro programa",)
    MIC_LOST = ("o microfone foi desconectado; o áudio já capturado foi transcrito",)
    MIC_PINNED_MISSING = (
        'microfone "{device}" não encontrado; usando o padrão do Windows',
        "device",
    )
    MIC_DEFAULT = ("Padrão do Windows",)

    # --- Dictation states and outcomes (overlay) ----------------------------------------
    STATE_RECORDING = ("gravando",)
    STATE_HANDSFREE = ("gravando (mãos livres)",)
    STATE_TRANSCRIBING = ("transcrevendo",)
    STATE_INSERTING = ("inserindo",)
    INSERTED = ("inserido",)
    NOTHING_DETECTED = ("nada detectado",)
    NOTHING_DETECTED_MUTED_HINT = ("nada detectado — o microfone está mudo?",)
    CANCELLED = ("cancelado",)
    INTERRUPTED = ("interrompido",)
    ASR_ERROR = ("erro na transcrição; tente novamente",)
    INSERT_FAILED_CLIPBOARD = ("não consegui colar — o texto está no clipboard (Ctrl+V)",)
    INSERT_FAILED_NO_FOCUS = ("nenhuma janela com foco — o texto está no clipboard (Ctrl+V)",)
    INSERT_FAILED_ELEVATED = (
        "não é possível inserir em janelas de administrador — o texto está no clipboard (Ctrl+V)",
    )
    INSERT_PARTIAL = ("inserção interrompida — o texto completo está no clipboard (Ctrl+V)",)
    INSERT_FAILED_OPEN_HISTORY = ("não consegui inserir — abra o histórico na bandeja",)
    CLIPBOARD_PARTIAL_RESTORE = ("parte do conteúdo anterior do clipboard não pôde ser restaurada",)
    REWRITE_SKIPPED = ("revisão por IA ignorada",)
    REWRITE_SKIPPED_TOO_LONG = ("revisão por IA ignorada: texto longo demais",)
    REWRITE_SKIPPED_AUTH = ("revisão por IA ignorada: o servidor recusou as credenciais",)
    REWRITE_SKIPPED_UNREACHABLE = ("revisão por IA ignorada: servidor local indisponível",)
    NOTHING_TO_REPASTE = ("nenhum ditado para reenviar",)

    # --- App, startup and Windows integration -------------------------------------------
    ONBOARDING = (
        "Segure o Ctrl direito por 2 segundos, fale depois do sinal e solte para inserir o texto.",
    )
    ALREADY_RUNNING = ("talktype já está em execução",)
    STARTUP_FAILED = ("o talktype não conseguiu iniciar: {reason}", "reason")
    APP_ERROR = ("o talktype encontrou um erro: {reason}", "reason")
    HOOK_BLOCKED = ("não foi possível instalar o atalho de teclado global",)
    HOTKEY_CONFLICT = ("o atalho {hotkey} já está em uso por outro programa", "hotkey")
    AUTOSTART_BLOCKED = ("não foi possível ativar a inicialização automática",)
    SOUND_MISSING = ("som não encontrado: {path}; usando o som padrão", "path")

    # --- History -------------------------------------------------------------------------
    HISTORY_CORRUPT = (
        "o histórico estava corrompido e foi reiniciado; uma cópia do arquivo foi guardada",
    )
    HISTORY_SAVE_FAILED = ("não foi possível salvar o histórico; os ditados ficam só nesta sessão",)
    HISTORY_EMPTY = ("Nenhum ditado ainda",)
    HISTORY_NOT_INSERTED = ("não inserido",)

    # --- Tray ----------------------------------------------------------------------------
    TRAY_STATUS = ("Modelo: {model} · {device}", "model", "device")
    TRAY_LOADING = ("Carregando modelo…",)
    TRAY_DOWNLOADING = ("Baixando modelo… {percent}%", "percent")
    TRAY_READY = ("Pronto",)
    TRAY_PAUSED = ("Pausado",)
    TRAY_ERROR = ("Erro: {reason}", "reason")
    MENU_MODE_PASTE = ("Modo: colar",)
    MENU_MODE_TYPE = ("Modo: digitar",)
    MENU_PREVIEW = ("Preview ao vivo",)
    MENU_CLEANUP = ("Limpeza",)
    MENU_COMMANDS = ("Comandos falados",)
    MENU_REWRITE = ("Revisão por IA",)
    MENU_PAUSE = ("Pausar",)
    MENU_RESUME = ("Retomar",)
    MENU_AUTOSTART = ("Iniciar com o Windows",)
    MENU_MICROPHONE = ("Microfone",)
    MENU_HISTORY = ("Histórico",)
    MENU_HISTORY_COPY = ("Copiar",)
    MENU_HISTORY_INSERT = ("Inserir",)
    MENU_HISTORY_CLEAR = ("Limpar histórico",)
    MENU_OPEN_CONFIG = ("Abrir config",)
    MENU_RELOAD_CONFIG = ("Recarregar config",)
    MENU_OPEN_CONFIG_FOLDER = ("Abrir pasta de config",)
    MENU_EXPORT_CONFIG = ("Exportar config",)
    MENU_IMPORT_CONFIG = ("Importar config",)
    MENU_RETRY_DOWNLOAD = ("Tentar baixar novamente",)
    MENU_QUIT = ("Sair",)
    DIALOG_EXPORT_TITLE = ("Exportar configuração",)
    DIALOG_IMPORT_TITLE = ("Importar configuração",)
    DIALOG_TOML_FILTER = ("Configuração do talktype (*.toml)",)

    def format(self, **kwargs: object) -> str:
        """Render the message. The keyword arguments must match `params` exactly."""
        if set(kwargs) != set(self.params):
            raise TypeError(
                f"{self.name} expects {sorted(self.params)}, got {sorted(kwargs)}",
            )
        return self.text.format(**kwargs)

    def __str__(self) -> str:
        return self.text


def placeholders(template: str) -> set[str]:
    """Return the named `str.format` fields used by `template`."""
    return {field for _, field, _, _ in Formatter().parse(template) if field}
