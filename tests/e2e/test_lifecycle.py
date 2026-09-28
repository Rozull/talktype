"""First run, single instance and a fresh install (E2E-016, E2E-017, E2E-019)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from talktype.history import HistoryStore
from talktype.paths import Paths
from tests import desktop
from tests.conftest import REPO_ROOT
from tests.e2e.conftest import (
    AppProcess,
    LogView,
    TextHarness,
    duration_ms,
    fixture_wav,
    hold_trigger,
    normalize,
    seam_env,
    wait_text,
)
from tests.integration.test_install import POWERSHELL, read_shortcut

pytestmark = pytest.mark.e2e

Launch = Callable[..., AppProcess]


# --------------------------------------------------------------------------------------
# E2E-016: first run
# --------------------------------------------------------------------------------------


def test_e2e016_first_run_creates_config_and_onboards_once(home: Path, launch: Launch) -> None:
    assert not any(home.iterdir())
    started = time.monotonic()
    app = launch(home, ready=False)
    app.wait_ready(timeout_s=15)
    assert time.monotonic() - started <= 15
    assert Paths(home).config.is_file()
    assert app.log.count("notice", key="ONBOARDING") == 1
    app.stop()

    again = launch(home)
    desktop.pump(1000)
    assert again.log.count("notice", key="ONBOARDING") == 1  # still only the first run's


# --------------------------------------------------------------------------------------
# E2E-017: single instance
# --------------------------------------------------------------------------------------


def test_e2e017_second_instance_exits_and_the_first_keeps_working(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    first = launch(home, fixture_wav("curta_sim"))
    started = time.monotonic()
    second = launch(home, fixture_wav("curta_sim"), ready=False)
    assert second.proc.wait(timeout=3) == 0
    assert time.monotonic() - started <= 3.5
    assert first.log.count("already_running") == 1

    harness.open()
    hold_trigger(first.log, duration_ms("curta_sim"))
    text = wait_text(harness, lambda t: "sim" in normalize(t))
    assert "sim" in normalize(text)


# --------------------------------------------------------------------------------------
# E2E-019: clone, uv sync, install, launch, update
# --------------------------------------------------------------------------------------


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=300
    )
    return result.stdout


def snapshot_repository(dest: Path) -> Path:
    """A git repository holding the working tree as it would be pushed (ignored files out).

    The work under test is not committed yet, so cloning `REPO_ROOT` itself would miss it.
    """
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", cwd=REPO_ROOT)
    dest.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=dest)
    for name in filter(None, listed.split("\0")):
        source = REPO_ROOT / name
        if not source.is_file():
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    git("add", "-A", cwd=dest)
    git("-c", "user.name=e2e", "-c", "user.email=e2e@localhost", "commit", "-q", "-m", "snapshot",
        cwd=dest)  # fmt: skip
    return dest


def uv(*args: str, cwd: Path, timeout: float = 1800) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(
        ["uv", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def test_e2e019_fresh_install_flow(tmp_path: Path) -> None:
    origin = snapshot_repository(tmp_path / "origin")
    clone = tmp_path / "clone"
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)

    synced = uv("sync", "--frozen", cwd=clone)
    assert synced.returncode == 0, synced.stderr[-2000:]

    appdata = tmp_path / "appdata"
    appdata.mkdir()
    install = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(clone / "scripts" / "install.ps1")],
        env=dict(os.environ, APPDATA=str(appdata)),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        check=False,
    )  # fmt: skip
    assert install.returncode == 0, install.stdout + install.stderr
    link = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "talktype.lnk"
    assert link.is_file()

    # The shortcut's target reaches READY.
    target, arguments, workdir = read_shortcut(link)
    home = tmp_path / "home"
    proc = subprocess.Popen([target, *arguments.split()], cwd=workdir, env=seam_env(home, None))
    app = AppProcess(home, proc, LogView(home))
    try:
        app.wait_ready()
    finally:
        app.stop()
    config_before = Paths(home).config.read_bytes()
    HistoryStore(home).add_pending("texto guardado")
    history_before = Paths(home).history.read_bytes()

    # An interrupted uv sync recovers on the next run.
    shutil.rmtree(clone / ".venv")
    interrupted = subprocess.Popen(["uv", "sync", "--frozen"], cwd=clone)
    time.sleep(3)
    interrupted.kill()  # TerminateProcess
    interrupted.wait(30)
    resumed = uv("sync", "--frozen", cwd=clone)
    assert resumed.returncode == 0, resumed.stderr[-2000:]

    # An update keeps the per-machine data.
    (origin / "NOTAS.md").write_text("atualização de teste\n", encoding="utf-8")
    git("add", "NOTAS.md", cwd=origin)
    git("-c", "user.name=e2e", "-c", "user.email=e2e@localhost", "commit", "-q", "-m", "update",
        cwd=origin)  # fmt: skip
    git("pull", "-q", cwd=clone)
    assert (clone / "NOTAS.md").is_file()
    updated = uv("sync", "--frozen", cwd=clone)
    assert updated.returncode == 0, updated.stderr[-2000:]
    assert Paths(home).config.read_bytes() == config_before
    assert Paths(home).history.read_bytes() == history_before
