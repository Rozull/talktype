"""Rotating key=value log.

Records are written as ``ts level event key=value ...``. The ``text`` field is replaced by
``text_len=N`` unless `debug.log_text` is enabled with `set_log_text(True)`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "talktype"
MAX_BYTES = 1024 * 1024
BACKUP_COUNT = 5
REDACTED_KEYS = frozenset({"text"})
_FIELDS_ATTR = "talktype_fields"

_log_text = False


def set_log_text(enabled: bool) -> None:
    """Allow transcript text in the log (`debug.log_text`)."""
    global _log_text
    _log_text = enabled


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Log `event` with key=value fields."""
    logger.log(level, event, extra={_FIELDS_ATTR: fields}, exc_info=exc_info)


def _format_value(value: object) -> str:
    text = str(value)
    if text == "" or any(c.isspace() or c in '="' for c in text):
        escaped = (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    return text


def render_fields(fields: dict[str, Any], *, log_text: bool) -> str:
    items: dict[str, Any] = {}
    for key, value in fields.items():
        if key in REDACTED_KEYS and not log_text:
            items.setdefault(f"{key}_len", len(str(value)))
            continue
        items[key] = value
    return " ".join(f"{key}={_format_value(value)}" for key, value in items.items())


class KeyValueFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        ts = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        line = f"{ts} {record.levelname} {record.getMessage()}"
        fields: dict[str, Any] | None = getattr(record, _FIELDS_ATTR, None)
        if fields:
            line = f"{line} {render_fields(fields, log_text=_log_text)}"
        if record.exc_info:
            line = f"{line} exc={_format_value(self.formatException(record.exc_info))}"
        return line


def setup_logging(
    log_file: Path,
    *,
    log_text: bool = False,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure the `talktype` logger with a 1 MiB x 5 rotating file handler."""
    set_log_text(log_text)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    close_logging()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = KeyValueFormatter()
    file_handler = RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console and sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def close_logging() -> None:
    """Detach and close every handler of the `talktype` logger."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
