r"""Start with Windows through the HKCU Run key.

The value ``talktype`` under ``HKCU\Software\Microsoft\Windows\CurrentVersion\Run`` holds
``"<venv>\Scripts\pythonw.exe" -m talktype``. Autostart counts as enabled only when that
value exists and the ``pythonw.exe`` it names still exists, so a moved repository shows
as off. The state lives in the registry, never in the configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Protocol

from talktype import win32
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg

VALUE_NAME = "talktype"

_log = get_logger("autostart")


class Registry(Protocol):
    """The HKCU Run-key operations autostart needs (a seam for `FakeRegistry`)."""

    def read(self, root: int, subkey: str, name: str) -> str | None: ...
    def write(self, root: int, subkey: str, name: str, value: str) -> None: ...
    def delete(self, root: int, subkey: str, name: str) -> bool: ...


class WinRegistry:
    def read(self, root: int, subkey: str, name: str) -> str | None:
        return win32.reg_read_str(root, subkey, name)

    def write(self, root: int, subkey: str, name: str, value: str) -> None:
        win32.reg_write_str(root, subkey, name, value)

    def delete(self, root: int, subkey: str, name: str) -> bool:
        return win32.reg_delete_value(root, subkey, name)


class AutostartError(Exception):
    """The Run key could not be written, or the launcher it would name does not exist."""

    def __init__(self, reason: Literal["blocked", "target_missing"]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.message = Msg.AUTOSTART_BLOCKED.text


def default_venv() -> Path:
    """The virtual environment running this interpreter (``.venv`` under uv)."""
    return Path(sys.prefix)


def launcher(venv: Path) -> Path:
    return venv / "Scripts" / "pythonw.exe"


def command_for(venv: Path) -> str:
    return f'"{launcher(venv)}" -m talktype'


def target_of(command: str) -> Path | None:
    """The executable named by a Run-key command line, or None when it cannot be parsed."""
    command = command.strip()
    if command.startswith('"'):
        end = command.find('"', 1)
        return Path(command[1:end]) if end > 1 else None
    first = command.split(" ", 1)[0]
    return Path(first) if first else None


def _registry(registry: Registry | None) -> Registry:
    return WinRegistry() if registry is None else registry


def is_enabled(*, name: str = VALUE_NAME, registry: Registry | None = None) -> bool:
    """True when the Run value exists and its target executable exists."""
    try:
        command = _registry(registry).read(win32.HKEY_CURRENT_USER, win32.RUN_KEY, name)
    except OSError:
        return False
    if not command:
        return False
    target = target_of(command)
    return target is not None and target.is_file()


def enable(
    venv: Path | None = None, *, name: str = VALUE_NAME, registry: Registry | None = None
) -> None:
    """Write the Run value (idempotent: one value, overwritten). Raises `AutostartError`."""
    reg = _registry(registry)
    venv = default_venv() if venv is None else venv
    if not launcher(venv).is_file():
        log_event(_log, "autostart_set", enabled=True, result="target_missing")
        raise AutostartError("target_missing")
    try:
        reg.write(win32.HKEY_CURRENT_USER, win32.RUN_KEY, name, command_for(venv))
    except OSError as exc:
        log_event(_log, "autostart_set", enabled=True, result="blocked")
        raise AutostartError("blocked") from exc
    if not is_enabled(name=name, registry=reg):
        log_event(_log, "autostart_set", enabled=True, result="blocked")
        raise AutostartError("blocked")
    log_event(_log, "autostart_set", enabled=True, result="ok")


def disable(*, name: str = VALUE_NAME, registry: Registry | None = None) -> None:
    """Remove the Run value; a missing value is fine. Raises `AutostartError`."""
    try:
        _registry(registry).delete(win32.HKEY_CURRENT_USER, win32.RUN_KEY, name)
    except OSError as exc:
        log_event(_log, "autostart_set", enabled=False, result="blocked")
        raise AutostartError("blocked") from exc
    log_event(_log, "autostart_set", enabled=False, result="ok")
