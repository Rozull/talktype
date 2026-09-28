"""History across processes (IT-031)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from talktype.history import HistoryStatus, HistoryStore

pytestmark = pytest.mark.integration

WRITER = """
import sys
from pathlib import Path
from talktype.history import HistoryStore

store = HistoryStore(Path(sys.argv[1]))
for text in ("um", "dois", "três"):
    store.mark(store.add_pending(text), "inserted")
"""


def test_it031_entries_written_by_another_process_are_read_newest_first(home: Path) -> None:
    subprocess.run([sys.executable, "-c", WRITER, str(home)], check=True, timeout=60)

    entries = HistoryStore(home).entries()

    assert [e.text for e in entries] == ["três", "dois", "um"]
    assert all(e.status is HistoryStatus.INSERTED for e in entries)
    assert len({e.id for e in entries}) == 3
