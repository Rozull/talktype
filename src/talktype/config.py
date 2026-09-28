"""Settings models, per-field validation with fallback, the tomlkit editor, export and import.

In short:
- `config.toml` is created from the packaged `config_template.toml` on first run.
- It is read with `tomllib` and edited with `tomlkit`, which keeps the user's comments.
- A syntax error rejects the whole reload and reports the line.
- A field that fails validation keeps its previous valid value; each is reported once.
- Unknown keys are reported as warnings.
- A non-loopback `llm.endpoint` is refused, and the AI rewrite is switched off.
- A file modified within the last 200 ms is re-read after 250 ms (half-written saves).
- Every write goes to ``<name>.tmp`` and then `os.replace()`.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import re
import time
import tomllib
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

import tomlkit
import tomlkit.items
from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import ErrorDetails

from talktype.paths import Paths
from talktype.strings import Msg
from talktype.win32 import MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, VK_F

TEMPLATE_NAME = "config_template.toml"

# --------------------------------------------------------------------------------------
# Static data
# --------------------------------------------------------------------------------------

TriggerKey = Literal[
    "right_ctrl",
    "right_alt",
    "right_shift",
    "caps_lock",
    "scroll_lock",
    "pause",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
]
ModelName = Literal["large-v3-turbo", "large-v3", "medium", "small", "parakeet-v3"]
DeviceName = Literal["auto", "cuda", "cpu"]
PreviewMode = Literal["auto", "on", "off"]

# Whisper's language codes (faster_whisper.tokenizer._LANGUAGE_CODES), kept here so that
# validating the config never imports ctranslate2.
WHISPER_LANGUAGES = frozenset(
    [
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "yue",
        "zh",
    ]
)

DEFAULT_HESITATIONS = ("hum", "hmm", "humm", "ahn", "ãh", "hã", "éé", "eh", "uh", "um")

DEFAULT_LLM_INSTRUCTION = (
    "Você revisa textos ditados em português do Brasil. Corrija a pontuação, as maiúsculas e "
    "erros evidentes de transcrição, e remova hesitações e repetições acidentais. Não mude o "
    "sentido, não traduza, não resuma e não acrescente informações. Mantenha os termos "
    "técnicos e as palavras em inglês como foram ditos. Responda apenas com o texto revisado, "
    "sem comentários."
)

DEFAULT_REPASTE_HOTKEY = "ctrl+alt+shift+v"

_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT}
_MODIFIERS |= {"win": MOD_WIN}
_NAMED_KEYS = {
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "pause": 0x13,
    "tab": 0x09,
}


def parse_hotkey(text: str) -> tuple[int, int]:
    """Parse ``"ctrl+alt+shift+v"`` into ``(MOD_* flags, virtual key)`` for RegisterHotKey.

    At least one modifier and exactly one key are required. Raises ValueError.
    """
    parts = [p.strip().lower() for p in text.split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(Msg.CONFIG_INVALID_HOTKEY.format(value=text))
    *mod_names, key = parts
    modifiers = 0
    for name in mod_names:
        flag = _MODIFIERS.get(name)
        if flag is None or modifiers & flag:
            raise ValueError(Msg.CONFIG_INVALID_HOTKEY.format(value=text))
        modifiers |= flag
    if not modifiers or key in _MODIFIERS:
        raise ValueError(Msg.CONFIG_INVALID_HOTKEY.format(value=text))
    if len(key) == 1 and ("a" <= key <= "z" or "0" <= key <= "9"):
        vk = ord(key.upper())
    elif key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    elif re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", key):
        vk = VK_F(int(key[1:]))
    else:
        raise ValueError(Msg.CONFIG_INVALID_HOTKEY.format(value=text))
    return modifiers, vk


def is_loopback_endpoint(url: str) -> bool:
    """True for an ``http(s)`` URL whose host is ``localhost`` or a loopback address.

    Loopback means 127.0.0.0/8 or ::1. No DNS lookup is made, so every other
    host name is refused.
    """
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_preview(preview: str, device: str) -> bool:
    """Whether the live preview runs: ``auto`` means only on CUDA."""
    if preview == "on":
        return True
    if preview == "off":
        return False
    return device == "cuda"


def _normalize_words(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        word = value.strip()
        if word and word not in seen:
            seen.add(word)
            result.append(word)
    return result


# --------------------------------------------------------------------------------------
# Settings models
# --------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="ignore", strict=True, validate_by_name=True, validate_by_alias=True
    )


class TriggerSettings(_Model):
    key: TriggerKey = "right_ctrl"
    hold_threshold_ms: Annotated[int, Field(ge=500, le=5000)] = 1000
    tap_threshold_ms: Annotated[int, Field(ge=50, le=500)] = 200
    double_tap_window_ms: Annotated[int, Field(ge=150, le=1000)] = 400
    mic_preopen_ms: Annotated[int, Field(ge=0, le=1000)] = 300


class RecordingSettings(_Model):
    max_seconds: Annotated[int, Field(ge=1, le=1800)] = 300
    min_speech_seconds: Annotated[float, Field(ge=0.1, le=2.0)] = 0.3
    sounds: bool = True
    start_sound: str = ""
    end_sound: str = ""
    microphone: str = ""


class OverlaySettings(_Model):
    preview: PreviewMode = "auto"
    preview_interval_ms: Annotated[int, Field(ge=300, le=2000)] = 700
    preview_window_seconds: Annotated[int, Field(ge=5, le=30)] = 15


class InjectionSettings(_Model):
    mode: Literal["paste", "type"] = "paste"
    paste_confirm_timeout_ms: Annotated[int, Field(ge=100, le=3000)] = 500
    restore_delay_ms: Annotated[int, Field(ge=0, le=2000)] = 150
    type_chunk_chars: Annotated[int, Field(ge=1, le=128)] = 16


class AsrSettings(_Model):
    model: ModelName = "large-v3-turbo"
    device: DeviceName = "auto"
    language: str = "pt"
    vocabulary: list[str] = Field(default_factory=list)

    @field_validator("language")
    @classmethod
    def _known_language(cls, value: str) -> str:
        code = value.strip().lower()
        if code not in WHISPER_LANGUAGES:
            raise ValueError(Msg.CONFIG_INVALID_LANGUAGE.format(value=value))
        return code

    @field_validator("vocabulary")
    @classmethod
    def _clean_vocabulary(cls, value: list[str]) -> list[str]:
        return _normalize_words(value)


class Substitution(_Model):
    from_: str = Field(alias="from")
    to: str = ""

    @field_validator("from_")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(Msg.CONFIG_EMPTY_SUBSTITUTION.text)
        return value


class CleanupSettings(_Model):
    enabled: bool = True
    remove_hesitations: bool = True
    normalize_whitespace: bool = True
    final_period: bool = True
    hesitations: list[str] = Field(default_factory=lambda: list(DEFAULT_HESITATIONS))
    substitutions: list[Substitution] = Field(default_factory=list)

    @field_validator("hesitations")
    @classmethod
    def _clean_hesitations(cls, value: list[str]) -> list[str]:
        return _normalize_words(value)


class CommandItem(_Model):
    phrase: str
    text: str

    @field_validator("phrase")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(Msg.CONFIG_EMPTY_COMMAND.text)
        return value.strip()


def _default_commands() -> list[CommandItem]:
    return [
        CommandItem(phrase="nova linha", text="\n"),
        CommandItem(phrase="novo parágrafo", text="\n\n"),
    ]


class CommandsSettings(_Model):
    enabled: bool = True
    items: list[CommandItem] = Field(default_factory=_default_commands)


class LlmSettings(_Model):
    enabled: bool = False
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_s: Annotated[float, Field(ge=1, le=30)] = 5.0
    keep_alive: str = "30m"
    api_key: str = ""
    instruction: str = DEFAULT_LLM_INSTRUCTION

    @field_validator("endpoint")
    @classmethod
    def _http_url(cls, value: str) -> str:
        url = value.strip()
        try:
            parts = urlsplit(url)
            valid = parts.scheme in ("http", "https") and bool(parts.hostname)
            _ = parts.port  # raises ValueError on a malformed port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("use um endereço http://host:porta")
        return url.rstrip("/")

    @field_validator("instruction")
    @classmethod
    def _default_instruction(cls, value: str) -> str:
        return value.strip() or DEFAULT_LLM_INSTRUCTION


class HistorySettings(_Model):
    size: Annotated[int, Field(ge=1, le=200)] = 20
    repaste_hotkey: str = DEFAULT_REPASTE_HOTKEY

    @field_validator("repaste_hotkey")
    @classmethod
    def _valid_hotkey(cls, value: str) -> str:
        parse_hotkey(value)
        return value.strip().lower()


class DebugSettings(_Model):
    log_text: bool = False


class Settings(_Model):
    trigger: TriggerSettings = Field(default_factory=TriggerSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    injection: InjectionSettings = Field(default_factory=InjectionSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    commands: CommandsSettings = Field(default_factory=CommandsSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    debug: DebugSettings = Field(default_factory=DebugSettings)


# Values above these limits are capped with a warning instead of rejected.
_CAPS: dict[tuple[str, str], int] = {("recording", "max_seconds"): 1800}

# Fields whose items are validated one by one; a bad item is dropped, the rest apply.
_ITEM_MODELS: dict[tuple[str, str], type[_Model]] = {
    ("cleanup", "substitutions"): Substitution,
    ("commands", "items"): CommandItem,
}


def _tables() -> dict[str, type[_Model]]:
    tables: dict[str, type[_Model]] = {}
    for name, info in Settings.model_fields.items():
        annotation = info.annotation
        assert isinstance(annotation, type) and issubclass(annotation, _Model)
        tables[name] = annotation
    return tables


def _field_names(model: type[BaseModel]) -> dict[str, str]:
    """TOML key (alias or name) -> attribute name."""
    return {(info.alias or name): name for name, info in model.model_fields.items()}


def leaf_keys() -> list[str]:
    """Every ``table.key`` of `Settings`, in declaration order."""
    return [f"{table}.{key}" for table, model in _tables().items() for key in _field_names(model)]


# --------------------------------------------------------------------------------------
# Reports and results
# --------------------------------------------------------------------------------------


class ConfigError(Exception):
    """One rejected key (or the whole file, keyed by its path) and why, in Portuguese."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(key, reason)
        self.key = key
        self.reason = reason

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfigError):
            return NotImplemented
        return (self.key, self.reason) == (other.key, other.reason)

    def __hash__(self) -> int:
        return hash((self.key, self.reason))

    def __str__(self) -> str:
        return f"{self.key}: {self.reason}"

    def __repr__(self) -> str:
        return f"ConfigError({self.key!r}, {self.reason!r})"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: list[ConfigError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fatal: bool = False  # the file was not applied at all
    changed: bool = False  # the effective settings differ from before
    model_changed: bool = False  # asr.model or asr.device changed

    @property
    def ok(self) -> bool:
        return not self.errors and not self.fatal

    def notice(self) -> str:
        """One Portuguese notice listing every rejected key and its reason."""
        details = "; ".join(str(e) for e in self.errors)
        if self.fatal:
            return Msg.CONFIG_RELOAD_REJECTED.format(details=details)
        if self.errors:
            return Msg.CONFIG_RELOADED_WITH_ERRORS.format(details=details)
        return Msg.CONFIG_RELOADED.text


SaveReason = Literal["read_only", "io_error"]


@dataclass(frozen=True, slots=True)
class SaveResult:
    saved: bool
    reason: SaveReason | None = None
    path: Path | None = None


ImportReason = Literal["unreadable", "empty", "syntax", "not_talktype", "invalid", "write_failed"]


@dataclass(frozen=True, slots=True)
class ImportResult:
    applied: bool
    reason: ImportReason | None = None
    message: str = ""
    backup: Path | None = None
    report: ValidationReport | None = None


# --------------------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------------------

_SYNTAX_POSITION = re.compile(r"\s*\(at line (\d+), column \d+\)\s*$")
_SYNTAX_DETAILS = {
    "Unterminated string": "texto sem aspas de fechamento",
    "Invalid value": "valor inválido",
    "Invalid statement": "linha inválida",
    "Expected '=' after a key in a key/value pair": "falta '=' depois da chave",
    "Expected newline or end of document after a statement": "sobra texto depois do valor",
    "Unclosed array": "lista sem ']' de fechamento",
    "Unclosed inline table": "tabela sem '}' de fechamento",
    "Invalid initial character for a key part": "nome de chave inválido",
    "Cannot overwrite a value": "chave repetida",
    "Illegal character": "caractere inválido",
}


class ConfigSyntaxError(ValueError):
    def __init__(self, line: int, detail: str) -> None:
        super().__init__(line, detail)
        self.line = line
        self.detail = detail

    @property
    def reason(self) -> str:
        return Msg.CONFIG_SYNTAX.format(line=self.line, detail=self.detail)


def parse_toml(content: bytes) -> dict[str, Any]:
    """Parse TOML bytes. Raises `ConfigSyntaxError` with the 1-based line."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        line = content[: exc.start].count(b"\n") + 1
        raise ConfigSyntaxError(line, "o arquivo não está em UTF-8") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        message = str(exc)
        match = _SYNTAX_POSITION.search(message)
        if match:
            line = int(match.group(1))
            message = message[: match.start()]
        else:
            message = message.replace("(at end of document)", "").strip()
            line = text.count("\n") + 1
        detail = next(
            (pt for en, pt in _SYNTAX_DETAILS.items() if message.startswith(en)),
            f"erro de sintaxe ({message})",
        )
        raise ConfigSyntaxError(line, detail) from exc


def _bounds(model: type[BaseModel], attr: str) -> tuple[object, object]:
    low: object = "?"
    high: object = "?"
    for meta in model.model_fields[attr].metadata:
        if isinstance(meta, Ge):
            low = meta.ge
        elif isinstance(meta, Le):
            high = meta.le
    return low, high


_TYPE_NAMES = {
    "int_type": "número inteiro",
    "float_type": "número",
    "bool_type": "true ou false",
    "string_type": "texto entre aspas",
    "list_type": "lista [ ... ]",
    "dict_type": "tabela { ... }",
    "model_type": "tabela { ... }",
}


def _reason(model: type[BaseModel], attr: str, error: ErrorDetails) -> str:
    kind = error["type"]
    ctx = error.get("ctx") or {}
    if kind == "literal_error":
        options = typing.get_args(model.model_fields[attr].annotation)
        return Msg.CONFIG_INVALID_CHOICE.format(options=", ".join(f'"{o}"' for o in options))
    if kind in _TYPE_NAMES:
        return Msg.CONFIG_WRONG_TYPE.format(expected=_TYPE_NAMES[kind])
    if kind in ("greater_than_equal", "less_than_equal", "greater_than", "less_than"):
        low, high = _bounds(model, attr)
        return Msg.CONFIG_OUT_OF_RANGE.format(minimum=low, maximum=high)
    if kind == "value_error" and "error" in ctx:
        return str(ctx["error"])
    return Msg.CONFIG_INVALID_VALUE.format(detail=error["msg"])


def _validate_items(
    item_model: type[_Model],
    key: str,
    raw: list[Any],
    errors: list[ConfigError],
    warnings: list[str],
) -> list[Any]:
    names = _field_names(item_model)
    kept: list[Any] = []
    for index, item in enumerate(raw):
        item_key = f"{key}[{index}]"
        if not isinstance(item, dict):
            errors.append(ConfigError(item_key, Msg.CONFIG_WRONG_TYPE.format(expected="tabela")))
            continue
        item = typing.cast(dict[str, Any], item)
        warnings.extend(
            Msg.CONFIG_UNKNOWN_KEY.format(key=f"{item_key}.{k}") for k in item if k not in names
        )
        try:
            kept.append(item_model.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0]
            attr = names.get(str(first["loc"][0]), str(first["loc"][0]))
            errors.append(ConfigError(item_key, _reason(item_model, attr, first)))
    return kept


def _validate_table(
    table: str,
    model: type[_Model],
    raw: Mapping[str, Any],
    previous: _Model,
    errors: list[ConfigError],
    warnings: list[str],
) -> _Model:
    names = _field_names(model)
    data: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in names:
            warnings.append(Msg.CONFIG_UNKNOWN_KEY.format(key=f"{table}.{key}"))
            continue
        attr = names[key]
        full_key = f"{table}.{key}"
        limit = _CAPS.get((table, attr))
        if limit is not None and type(value) is int and value > limit:
            warnings.append(Msg.CONFIG_CAPPED.format(key=full_key, limit=limit))
            value = limit
        item_model = _ITEM_MODELS.get((table, attr))
        if item_model is not None and isinstance(value, list):
            value = _validate_items(
                item_model, full_key, typing.cast(list[Any], value), errors, warnings
            )
        data[attr] = value

    rejected: set[str] = set()
    for _ in range(len(names) + 1):
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            for error in exc.errors():
                attr = names.get(str(error["loc"][0]), str(error["loc"][0]))
                if attr in rejected:
                    continue
                rejected.add(attr)
                alias = model.model_fields[attr].alias or attr
                errors.append(ConfigError(f"{table}.{alias}", _reason(model, attr, error)))
                data[attr] = getattr(previous, attr)
    return previous


def validate_data(
    data: Mapping[str, Any], previous: Settings | None = None
) -> tuple[Settings, list[ConfigError], list[str]]:
    """Validate parsed TOML field by field.

    A missing key takes its default. A rejected key keeps its value from `previous`.
    """
    previous = Settings() if previous is None else previous
    errors: list[ConfigError] = []
    warnings: list[str] = []
    tables = _tables()
    values: dict[str, _Model] = {}
    for key in data:
        if key not in tables:
            warnings.append(Msg.CONFIG_UNKNOWN_KEY.format(key=key))
    for table, model in tables.items():
        raw = data.get(table, {})
        prev_table = getattr(previous, table)
        if not isinstance(raw, dict):
            errors.append(ConfigError(table, Msg.CONFIG_NOT_A_TABLE.text))
            values[table] = prev_table
            continue
        raw = typing.cast(dict[str, Any], raw)
        values[table] = _validate_table(table, model, raw, prev_table, errors, warnings)
    values["llm"] = _local_llm_only(typing.cast(LlmSettings, values["llm"]), previous.llm, errors)
    return Settings.model_validate(values), errors, warnings


def _local_llm_only(
    llm: LlmSettings, previous: LlmSettings, errors: list[ConfigError]
) -> LlmSettings:
    """Refuse a non-loopback `llm.endpoint`: keep a local one and switch the rewrite off."""
    if is_loopback_endpoint(llm.endpoint):
        return llm
    errors.append(ConfigError("llm.endpoint", Msg.LLM_ENDPOINT_NOT_LOCAL.text))
    endpoint = previous.endpoint
    if not is_loopback_endpoint(endpoint):
        endpoint = LlmSettings().endpoint
    return llm.model_copy(update={"enabled": False, "endpoint": endpoint})


def known_keys(data: Mapping[str, Any]) -> list[str]:
    """The ``table.key`` entries of `data` that `Settings` understands."""
    found: list[str] = []
    for table, model in _tables().items():
        raw = data.get(table)
        if isinstance(raw, dict):
            names = _field_names(model)
            found.extend(f"{table}.{k}" for k in typing.cast(dict[str, Any], raw) if k in names)
    return found


def template_bytes() -> bytes:
    """The packaged first-run configuration."""
    return resources.files("talktype").joinpath(TEMPLATE_NAME).read_bytes()


# --------------------------------------------------------------------------------------
# File writing and editing
# --------------------------------------------------------------------------------------


def atomic_write(path: Path, data: bytes) -> None:
    """Write `<path>.tmp` and `os.replace()` it over `path`. Removes the tmp on failure."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _save_error(exc: OSError) -> SaveReason:
    return "read_only" if isinstance(exc, PermissionError) else "io_error"


def _split_key(key: str) -> tuple[str, str]:
    table, _, name = key.partition(".")
    model = _tables().get(table)
    if model is None or name not in _field_names(model):
        raise KeyError(key)
    return table, name


def _to_toml_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, list):
        return [_to_toml_value(v) for v in typing.cast(list[Any], value)]
    return value


def _edit_document(content: bytes, key: str, value: Any) -> bytes:
    table, name = _split_key(key)
    doc = tomlkit.parse(content.decode("utf-8-sig"))
    if table not in doc:
        doc.add(table, tomlkit.table())
    container = doc[table]
    assert isinstance(container, tomlkit.items.Table | tomlkit.items.InlineTable)
    new_item = tomlkit.item(_to_toml_value(value))
    old_item = container.get(name)
    if isinstance(old_item, tomlkit.items.Item):
        new_item.trivia.indent = old_item.trivia.indent
        new_item.trivia.comment_ws = old_item.trivia.comment_ws
        new_item.trivia.comment = old_item.trivia.comment
        new_item.trivia.trail = old_item.trivia.trail
    container[name] = new_item
    return tomlkit.dumps(doc).encode("utf-8")


def set_value(path: Path, key: str, value: Any) -> SaveResult:
    """Rewrite only `key` in the TOML file at `path`, keeping every comment.

    A missing file is recreated from the template first.
    """
    try:
        content = path.read_bytes() if path.exists() else template_bytes()
        updated = _edit_document(content, key, value)
        if path.exists() and not os.access(path, os.W_OK):
            return SaveResult(saved=False, reason="read_only", path=path)
        atomic_write(path, updated)
    except OSError as exc:
        return SaveResult(saved=False, reason=_save_error(exc), path=path)
    return SaveResult(saved=True, path=path)


# --------------------------------------------------------------------------------------
# The live configuration
# --------------------------------------------------------------------------------------


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock seconds since the epoch (comparable with file mtimes)."""
        ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


FRESH_WRITE_S = 0.200
RETRY_DELAY_S = 0.250
MAX_READ_ATTEMPTS = 8


class ConfigManager:
    """Owns `config.toml` and the last valid `Settings` (Qt GUI thread)."""

    def __init__(self, home: Path, *, clock: Clock | None = None) -> None:
        self.paths = Paths(home)
        self._clock: Clock = SystemClock() if clock is None else clock
        self._settings = Settings()
        self.last_report = self.reload()

    @property
    def path(self) -> Path:
        return self.paths.config

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- reading -----------------------------------------------------------------------

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self.path, template_bytes())

    def _read_stable(self) -> bytes:
        """Read the file, waiting out a save that is still in progress."""
        data = b""
        for _ in range(MAX_READ_ATTEMPTS):
            before = self.path.stat()
            if abs(self._clock.now() - before.st_mtime) < FRESH_WRITE_S:
                self._clock.sleep(RETRY_DELAY_S)
                continue
            data = self.path.read_bytes()
            after = self.path.stat()
            if (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size):
                return data
            self._clock.sleep(RETRY_DELAY_S)
        return self.path.read_bytes()

    def reload(self) -> ValidationReport:
        """Re-read `config.toml` and apply every valid field."""
        previous = self._settings
        try:
            self._ensure_file()
            content = self._read_stable()
        except OSError:
            report = ValidationReport(
                errors=[ConfigError(str(self.path), Msg.CONFIG_UNREADABLE.format(path=self.path))],
                fatal=True,
            )
            self.last_report = report
            return report
        try:
            data = parse_toml(content)
        except ConfigSyntaxError as exc:
            report = ValidationReport(errors=[ConfigError(str(self.path), exc.reason)], fatal=True)
            self.last_report = report
            return report
        settings, errors, warnings = validate_data(data, previous)
        self._settings = settings
        report = ValidationReport(
            errors=errors,
            warnings=warnings,
            changed=settings != previous,
            model_changed=(settings.asr.model, settings.asr.device)
            != (previous.asr.model, previous.asr.device),
        )
        self.last_report = report
        return report

    # -- editing -----------------------------------------------------------------------

    def set_value(self, key: str, value: Any) -> SaveResult:
        """Apply `key` in memory and persist it. The change holds even when saving fails."""
        table, name = _split_key(key)
        model = _tables()[table]
        current = getattr(self._settings, table)
        data = current.model_dump(by_alias=True)
        data[name] = _to_toml_value(value)
        updated = model.model_validate(data)
        self._settings = self._settings.model_copy(update={table: updated})
        return set_value(self.path, key, value)

    # -- export and import -------------------------------------------------------------

    def export_config(self, dest: Path) -> SaveResult:
        """Write a byte-identical copy of `config.toml` to `dest`."""
        try:
            self._ensure_file()
            atomic_write(dest, self.path.read_bytes())
        except OSError as exc:
            return SaveResult(saved=False, reason=_save_error(exc), path=dest)
        return SaveResult(saved=True, path=dest)

    def _backup_path(self) -> Path:
        when = datetime.fromtimestamp(self._clock.now())
        candidate = self.paths.config_backup(when)
        counter = 1
        while candidate.exists():
            candidate = candidate.with_name(f"{candidate.stem}-{counter}.toml")
            counter += 1
        return candidate

    def import_config(self, src: Path) -> ImportResult:
        """Replace `config.toml` with `src` when it is a valid configuration.

        The previous file is kept as ``config.backup-<timestamp>.toml``. Anything invalid
        leaves `config.toml` untouched and creates no backup.
        """

        def reject(reason: ImportReason, detail: str) -> ImportResult:
            return ImportResult(
                applied=False, reason=reason, message=Msg.IMPORT_REJECTED.format(reason=detail)
            )

        try:
            content = src.read_bytes()
        except OSError:
            return reject("unreadable", Msg.IMPORT_UNREADABLE.format(path=src))
        if not content.strip():
            return reject("empty", Msg.IMPORT_EMPTY.text)
        try:
            data = parse_toml(content)
        except ConfigSyntaxError as exc:
            return reject("syntax", exc.reason)
        if not known_keys(data):
            return reject("not_talktype", Msg.IMPORT_NOT_TALKTYPE.text)
        _, errors, _ = validate_data(data, self._settings)
        if errors:
            return reject("invalid", "; ".join(str(e) for e in errors))

        backup: Path | None = None
        try:
            if self.path.exists():
                backup = self._backup_path()
                atomic_write(backup, self.path.read_bytes())
            atomic_write(self.path, content)
        except OSError:
            return reject("write_failed", Msg.CONFIG_SAVE_FAILED.text)
        report = self.reload()
        return ImportResult(
            applied=True,
            message=Msg.CONFIG_IMPORTED.format(backup=backup or "-"),
            backup=backup,
            report=report,
        )


def load_or_create(home: Path, *, clock: Clock | None = None) -> Settings:
    """Load `home/config.toml`, creating it from the template when missing."""
    return ConfigManager(home, clock=clock).settings
