"""The online installer: scripts/setup.ps1 and scripts/uninstall.ps1, end to end.

A zip of the working tree stands in for the GitHub release (``-Source``), and APPDATA and
LOCALAPPDATA point into `tmp_path`, so nothing touches the real install. The HKCU
"Uninstall\\talktype" entry is real: it is saved before and restored afterwards.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import winreg
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.e2e

UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\talktype"


def _read_key() -> dict[str, tuple[Any, int]] | None:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except FileNotFoundError:
        return None
    values: dict[str, tuple[Any, int]] = {}
    with key:
        index = 0
        while True:
            try:
                name, data, kind = winreg.EnumValue(key, index)
            except OSError:
                return values
            values[name] = (data, kind)
            index += 1


def _write_key(values: dict[str, tuple[Any, int]] | None) -> None:
    with contextlib.suppress(FileNotFoundError):
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    if values is None:
        return
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
        for name, (data, kind) in values.items():
            winreg.SetValueEx(key, name, 0, kind, data)


@pytest.fixture
def keep_uninstall_entry() -> Iterator[None]:
    saved = _read_key()
    yield
    _write_key(saved)


def _source_zip(target: Path) -> Path:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in listed:
            path = REPO_ROOT / name
            if path.is_file():
                archive.write(path, f"talktype-test/{name}")
    return target


def _run(script: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )


@pytest.mark.usefixtures("keep_uninstall_entry")
def test_setup_installs_updates_and_uninstalls(tmp_path: Path) -> None:
    zip_path = _source_zip(tmp_path / "talktype.zip")
    env = dict(os.environ)
    env["APPDATA"] = str(tmp_path / "appdata")
    env["LOCALAPPDATA"] = str(tmp_path / "local")
    env.pop("TALKTYPE_HOME", None)
    app = tmp_path / "local" / "talktype" / "app"
    link = tmp_path / "appdata" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    setup = REPO_ROOT / "scripts" / "setup.ps1"

    first = _run(setup, "-Source", str(zip_path), "-Device", "cpu", "-NoLaunch", env=env)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert (app / ".venv" / "Scripts" / "pythonw.exe").is_file()
    assert (app / ".talktype-install").is_file()
    assert (link / "talktype.lnk").is_file()
    assert not (app / ".venv" / "Lib" / "site-packages" / "nvidia").exists()  # CPU: no CUDA
    config = (tmp_path / "appdata" / "talktype" / "config.toml").read_text(encoding="utf-8")
    assert 'model = "parakeet-v3"' in config  # a new config on a CPU machine
    entry = _read_key()
    assert entry is not None and entry["InstallLocation"][0] == str(app)

    # Running it again is an update: the config the user has is left alone.
    (tmp_path / "appdata" / "talktype" / "config.toml").write_text(
        config.replace('model = "parakeet-v3"', 'model = "small"'), encoding="utf-8"
    )
    again = _run(setup, "-Source", str(zip_path), "-Device", "cpu", "-NoLaunch", env=env)
    assert again.returncode == 0, again.stdout[-3000:] + again.stderr[-3000:]
    kept = (tmp_path / "appdata" / "talktype" / "config.toml").read_text(encoding="utf-8")
    assert 'model = "small"' in kept

    removed = _run(app / "scripts" / "uninstall.ps1", "-Quiet", env=env)
    assert removed.returncode == 0, removed.stdout[-3000:] + removed.stderr[-3000:]
    assert not app.exists()
    assert not (link / "talktype.lnk").exists()
    assert (tmp_path / "appdata" / "talktype" / "config.toml").is_file()  # kept without -Purge
    assert _read_key() is None


def test_uninstall_never_deletes_a_folder_the_setup_did_not_create(tmp_path: Path) -> None:
    """Run from a clone (no marker), the uninstaller keeps the folder."""
    clone = tmp_path / "clone"
    (clone / "scripts").mkdir(parents=True)
    script = clone / "scripts" / "uninstall.ps1"
    script.write_bytes((REPO_ROOT / "scripts" / "uninstall.ps1").read_bytes())
    (clone / "pyproject.toml").write_text("[project]\nname = 'talktype'\n", encoding="utf-8")
    env = dict(os.environ)
    env["APPDATA"] = str(tmp_path / "appdata")

    result = _run(script, "-Quiet", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (clone / "pyproject.toml").is_file()
    assert "não foi criada pelo instalador" in result.stdout
