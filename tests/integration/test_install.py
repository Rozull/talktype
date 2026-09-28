"""`scripts/install.ps1`: shortcut, idempotency and prerequisites (IT-032, IT-033)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from talktype import win32
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

SCRIPT = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / (
    r"System32\WindowsPowerShell\v1.0\powershell.exe"
)


def run_install(
    script: Path, appdata: Path, *args: str, path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, APPDATA=str(appdata))
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


def read_shortcut(link: Path) -> tuple[str, str, str]:
    command = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:TT_LINK); "
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        "Write-Output $s.TargetPath; Write-Output $s.Arguments; Write-Output $s.WorkingDirectory"
    )
    out = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        env=dict(os.environ, TT_LINK=str(link)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    ).stdout.splitlines()
    return out[0], out[1], out[2]


def fake_repo(root: Path, *, venv: bool) -> Path:
    """A repository layout with a copy of the script and, optionally, a `.venv`."""
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts" / "install.ps1"
    shutil.copyfile(SCRIPT, script)
    if venv:
        pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
        pythonw.parent.mkdir(parents=True)
        pythonw.write_bytes(b"")
    return script


def run_value() -> str | None:
    return win32.reg_read_str(win32.HKEY_CURRENT_USER, win32.RUN_KEY, "talktype")


def programs(appdata: Path) -> Path:
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def test_it032_creates_the_shortcut_idempotently(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    run_before = run_value()

    first = run_install(SCRIPT, appdata)
    assert first.returncode == 0, first.stdout + first.stderr
    link = programs(appdata) / "talktype.lnk"
    assert link.is_file()
    target, arguments, workdir = read_shortcut(link)
    assert Path(target) == REPO_ROOT / ".venv" / "Scripts" / "pythonw.exe"
    assert arguments == "-m talktype"
    assert Path(workdir) == REPO_ROOT

    second = run_install(SCRIPT, appdata)
    assert second.returncode == 0, second.stdout + second.stderr
    assert [p.name for p in programs(appdata).iterdir()] == ["talktype.lnk"]
    assert run_value() == run_before  # the Run key is never written


def test_it032_refuses_to_run_without_the_venv(tmp_path: Path) -> None:
    script = fake_repo(tmp_path / "repo", venv=False)
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    result = run_install(script, appdata)
    assert result.returncode == 1
    assert "Execute `uv sync` primeiro" in result.stdout
    assert not programs(appdata).exists() or not any(programs(appdata).iterdir())


def test_it032_a_copied_repository_points_to_its_own_venv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    script = fake_repo(repo, venv=True)
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    result = run_install(script, appdata)
    assert result.returncode == 0, result.stdout + result.stderr
    target, arguments, _ = read_shortcut(programs(appdata) / "talktype.lnk")
    assert Path(target) == repo / ".venv" / "Scripts" / "pythonw.exe"
    assert arguments == "-m talktype"


def test_it033_missing_uv(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH for this test"
    uv_dirs = {str(Path(uv).parent).lower()}
    kept = [
        entry
        for entry in os.environ["PATH"].split(os.pathsep)
        if entry
        and entry.rstrip("\\").lower() not in uv_dirs
        and not (Path(entry) / "uv.exe").exists()
    ]
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    result = run_install(SCRIPT, appdata, path=os.pathsep.join(kept))
    assert result.returncode == 1
    assert "uv não encontrado" in result.stdout


def test_it033_old_windows(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    result = run_install(SCRIPT, appdata, "-FakeOsBuild", "9600")
    assert result.returncode == 1
    assert "Windows 10 ou 11 é necessário" in result.stdout
    assert not programs(appdata).exists()
