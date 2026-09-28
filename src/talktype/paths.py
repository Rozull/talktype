"""Resolves the per-machine data directory and the files inside it."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HOME_ENV = "TALKTYPE_HOME"


def resolve_home(env: Mapping[str, str] | None = None) -> Path:
    r"""`TALKTYPE_HOME` when set, otherwise `%APPDATA%\talktype`."""
    env = os.environ if env is None else env
    override = env.get(HOME_ENV, "").strip()
    if override:
        return Path(override)
    appdata = env.get("APPDATA", "").strip()
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "talktype"


@dataclass(frozen=True, slots=True)
class Paths:
    home: Path

    @classmethod
    def resolve(cls, env: Mapping[str, str] | None = None) -> Paths:
        return cls(resolve_home(env))

    @property
    def config(self) -> Path:
        return self.home / "config.toml"

    @property
    def history(self) -> Path:
        return self.home / "history.json"

    @property
    def state(self) -> Path:
        return self.home / "state.json"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "talktype.log"

    def config_backup(self, when: datetime) -> Path:
        return self.home / f"config.backup-{when:%Y%m%d-%H%M%S}.toml"

    def history_corrupt(self, when: datetime) -> Path:
        return self.home / f"history.corrupt-{when:%Y%m%d-%H%M%S}.json"

    def ensure(self) -> Paths:
        """Create the data and log directories if they are missing."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self
