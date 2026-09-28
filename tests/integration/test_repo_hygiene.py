"""Repository hygiene and template/example parity (IT-034)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import PurePosixPath

import pytest

from talktype.config import template_bytes
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


def repo_files() -> list[PurePosixPath]:
    """Tracked files plus untracked files that are not ignored (what a commit could add)."""
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        [git, "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return [PurePosixPath(line) for line in out.splitlines() if line]


def test_it034_no_personal_data_in_the_repository() -> None:
    files = repo_files()
    assert files

    configs = [f for f in files if f.name == "config.toml"]
    histories = [f for f in files if f.name in ("history.json", "state.json")]
    logs = [f for f in files if f.suffix == ".log" or "logs" in f.parts]
    models = [f for f in files if any(p.startswith("models--") or p == "models" for p in f.parts)]
    backups = [f for f in files if f.name.startswith(("config.backup-", "history.corrupt-"))]

    assert configs == []
    assert histories == []
    assert logs == []
    assert models == []
    assert backups == []


def test_it034_example_config_is_byte_identical_to_template() -> None:
    template = REPO_ROOT / "src" / "talktype" / "config_template.toml"

    assert (REPO_ROOT / "config.example.toml").read_bytes() == template.read_bytes()
    assert template.read_bytes() == template_bytes()
