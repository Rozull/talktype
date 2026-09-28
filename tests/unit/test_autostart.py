"""Autostart through the HKCU Run key with `FakeRegistry` (UT-197, UT-198)."""

from __future__ import annotations

from pathlib import Path

import pytest

from talktype import autostart, win32
from talktype.strings import Msg
from tests.fakes import FakeRegistry

pytestmark = pytest.mark.unit

KEY = (win32.RUN_KEY, autostart.VALUE_NAME)


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / ".venv"
    (root / "Scripts").mkdir(parents=True)
    (root / "Scripts" / "pythonw.exe").write_bytes(b"")
    return root


def test_ut197_enable_writes_run_value_once_and_disable_removes_it(venv: Path) -> None:
    registry = FakeRegistry()
    assert autostart.is_enabled(registry=registry) is False  # default after install

    autostart.enable(venv, registry=registry)
    expected = f'"{venv}\\Scripts\\pythonw.exe" -m talktype'
    assert registry.values == {KEY: expected}
    assert autostart.is_enabled(registry=registry) is True

    autostart.enable(venv, registry=registry)
    assert registry.values == {KEY: expected}  # still a single startup entry

    autostart.disable(registry=registry)
    assert registry.values == {}
    assert autostart.is_enabled(registry=registry) is False
    autostart.disable(registry=registry)  # disabling twice is fine


def test_ut197_policy_block_raises_autostart_error(venv: Path, logs: list[str]) -> None:
    registry = FakeRegistry()
    registry.fail_writes = PermissionError(5, "Acesso negado")

    with pytest.raises(autostart.AutostartError) as info:
        autostart.enable(venv, registry=registry)

    assert info.value.reason == "blocked"
    assert info.value.message == Msg.AUTOSTART_BLOCKED.text
    assert autostart.is_enabled(registry=registry) is False
    assert "autostart_set enabled=True result=blocked" in logs

    with pytest.raises(autostart.AutostartError):
        autostart.disable(registry=registry)


def test_ut197_missing_launcher_is_not_written(tmp_path: Path) -> None:
    registry = FakeRegistry()
    with pytest.raises(autostart.AutostartError) as info:
        autostart.enable(tmp_path / "no-venv", registry=registry)
    assert info.value.reason == "target_missing"
    assert registry.values == {}


def test_ut198_run_value_with_missing_pythonw_reads_as_disabled(venv: Path) -> None:
    registry = FakeRegistry()
    autostart.enable(venv, registry=registry)
    (venv / "Scripts" / "pythonw.exe").unlink()  # the repository was moved

    assert autostart.is_enabled(registry=registry) is False

    registry.values[KEY] = "garbage"
    assert autostart.is_enabled(registry=registry) is False


def test_target_of_parses_quoted_and_bare_commands() -> None:
    assert autostart.target_of('"C:\\a b\\pythonw.exe" -m talktype') == Path("C:/a b/pythonw.exe")
    assert autostart.target_of("C:\\x\\pythonw.exe -m talktype") == Path("C:/x/pythonw.exe")
    assert autostart.target_of('"') is None
