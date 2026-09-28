"""Log rotation and redaction on disk (IT-041)."""

from __future__ import annotations

from pathlib import Path

import pytest

from talktype.logging_setup import BACKUP_COUNT, close_logging, log_event, setup_logging

pytestmark = pytest.mark.integration


def test_it041_rotation_keeps_five_backups_and_never_writes_text(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "talktype.log"
    logger = setup_logging(log_file, log_text=False, console=False)
    padding = "x" * 900
    try:
        for n in range(7000):  # about 6.5 MiB
            log_event(
                logger, "dictation_finished", id=n, text="segredo", status="inserted", pad=padding
            )
    finally:
        close_logging()

    names = sorted(p.name for p in log_file.parent.iterdir())
    expected = ["talktype.log", *(f"talktype.log.{n}" for n in range(1, BACKUP_COUNT + 1))]
    assert names == sorted(expected)
    for path in log_file.parent.iterdir():
        content = path.read_bytes()
        assert b"segredo" not in content
        assert b"text_len=7" in content
