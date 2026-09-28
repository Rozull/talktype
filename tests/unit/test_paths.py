"""Data directory resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from talktype.paths import Paths, resolve_home

pytestmark = pytest.mark.unit


def test_talktype_home_overrides_appdata() -> None:
    env = {"TALKTYPE_HOME": r"D:\tmp\tt", "APPDATA": r"C:\Users\u\AppData\Roaming"}

    assert resolve_home(env) == Path(r"D:\tmp\tt")


def test_default_is_appdata_talktype() -> None:
    env = {"APPDATA": r"C:\Users\u\AppData\Roaming", "TALKTYPE_HOME": "  "}

    assert resolve_home(env) == Path(r"C:\Users\u\AppData\Roaming\talktype")


def test_files_inside_home() -> None:
    paths = Paths(Path(r"C:\h"))
    when = datetime(2026, 9, 28, 14, 5, 9)

    assert paths.config == Path(r"C:\h\config.toml")
    assert paths.history == Path(r"C:\h\history.json")
    assert paths.state == Path(r"C:\h\state.json")
    assert paths.log_file == Path(r"C:\h\logs\talktype.log")
    assert paths.config_backup(when).name == "config.backup-20260928-140509.toml"
    assert paths.history_corrupt(when).name == "history.corrupt-20260928-140509.json"
