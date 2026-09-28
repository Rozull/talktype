"""Config reload against real files and real timing (IT-020, IT-021)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from talktype.config import ConfigManager, Settings, template_bytes

pytestmark = pytest.mark.integration


def age(path: Path, seconds: float = 10.0) -> None:
    """Backdate `path` so the half-written-file guard does not wait for it."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_it020_real_file_reload_and_comment_preserving_edit(home: Path) -> None:
    path = home / "config.toml"
    user_text = (
        template_bytes()
        .decode("utf-8")
        .replace("[trigger]\n", "[trigger]\n# anotação minha: não mexer no limiar\n")
    )
    path.write_text(user_text, encoding="utf-8")
    age(path)
    manager = ConfigManager(home)
    assert manager.last_report.ok

    edited = user_text.replace('key = "right_ctrl"', 'key = "banana"').replace(
        'mode = "paste"', 'mode = "type"'
    )
    path.write_text(edited, encoding="utf-8")
    age(path)
    report = manager.reload()

    assert manager.settings.injection.mode == "type"
    assert manager.settings.trigger.key == "right_ctrl"
    assert [e.key for e in report.errors] == ["trigger.key"]

    result = manager.set_value("overlay.preview", "off")

    assert result.saved is True
    on_disk = path.read_text(encoding="utf-8")
    comments_before = [line for line in edited.splitlines() if line.lstrip().startswith("#")]
    comments_after = [line for line in on_disk.splitlines() if line.lstrip().startswith("#")]
    assert comments_after == comments_before
    assert on_disk == edited.replace('preview = "auto"', 'preview = "off"', 1)


OLD = '[injection]\nmode = "paste"\n[trigger]\nkey = "right_ctrl"\n[recording]\nmax_seconds = 300\n'
NEW_CHUNKS = (
    '[injection]\nmode = "type"\n',
    '[trigger]\nkey = "right_alt"\n',
    "[recording]\nmax_seconds = 60\n",
)


def slow_writer(path: Path) -> None:
    with path.open("wb") as fh:
        for index, chunk in enumerate(NEW_CHUNKS):
            if index:
                time.sleep(0.100)
            fh.write(chunk.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())


@pytest.mark.parametrize("start_delay_s", [0.0, 0.05, 0.12, 0.18])
def test_it021_concurrent_save_is_never_applied_half_written(
    home: Path, start_delay_s: float
) -> None:
    path = home / "config.toml"
    path.write_text(OLD, encoding="utf-8")
    age(path)
    manager = ConfigManager(home)
    old_settings = manager.settings
    new_settings = Settings.model_validate(
        {
            "injection": {"mode": "type"},
            "trigger": {"key": "right_alt"},
            "recording": {"max_seconds": 60},
        }
    )

    writer = threading.Thread(target=slow_writer, args=(path,))
    writer.start()
    time.sleep(start_delay_s)
    manager.reload()
    writer.join()

    assert manager.settings in (old_settings, new_settings)
