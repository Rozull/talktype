"""`HistoryStore` on `tmp_path` (UT-125 … UT-133)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talktype.history import HistoryEntry, HistoryStatus, HistoryStore, RealFileOps, label
from talktype.strings import Msg

pytestmark = pytest.mark.unit


class RecordingFileOps(RealFileOps):
    """Real filesystem that records calls and can fail writes on demand."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_writes = False

    def write_bytes(self, path: Path, data: bytes) -> None:
        self.calls.append(("write", path.name))
        if self.fail_writes:
            raise PermissionError(13, "Acesso negado", str(path))
        super().write_bytes(path, data)

    def replace(self, src: Path, dst: Path) -> None:
        self.calls.append(("replace", f"{src.name}->{dst.name}"))
        super().replace(src, dst)


def read_file(home: Path) -> list[dict[str, str]]:
    return json.loads((home / "history.json").read_text(encoding="utf-8"))


def test_ut125_add_pending_then_mark_persists(home: Path) -> None:
    store = HistoryStore(home)

    entry_id = store.add_pending("a")
    assert store.mark(entry_id, "inserted") is True

    [saved] = read_file(home)
    assert saved["id"] == entry_id
    assert saved["text"] == "a"
    assert saved["status"] == "inserted"
    assert saved["created_at"]
    reopened = HistoryStore(home)
    assert [(e.id, e.text, e.status) for e in reopened.entries()] == [
        (entry_id, "a", HistoryStatus.INSERTED)
    ]


def test_ut126_history_keeps_the_newest_20(home: Path) -> None:
    store = HistoryStore(home)

    for n in range(21):
        store.add_pending(f"ditado {n}")

    texts = [e["text"] for e in read_file(home)]
    assert len(texts) == 20
    assert "ditado 0" not in texts
    assert texts[0] == "ditado 20"


def test_ut127_corrupt_file_is_quarantined(home: Path) -> None:
    (home / "history.json").write_bytes(b"{not json")

    store = HistoryStore(home)

    assert store.entries() == []
    assert store.load_warning is Msg.HISTORY_CORRUPT
    [corrupt] = list(home.glob("history.corrupt-*.json"))
    assert corrupt.read_bytes() == b"{not json"


def test_ut128_unwritable_history_keeps_entries_in_memory(home: Path) -> None:
    fs = RecordingFileOps()
    fs.fail_writes = True
    store = HistoryStore(home, fs=fs)

    entry_id = store.add_pending("perdido?")

    assert store.save_failed is True
    assert [e.id for e in store.entries()] == [entry_id]
    assert not (home / "history.json").exists()


def test_ut129_clear_leaves_an_empty_array(home: Path) -> None:
    store = HistoryStore(home)
    store.add_pending("a")

    store.clear()

    assert store.entries() == []
    assert read_file(home) == []


def test_ut130_save_is_tmp_then_replace_and_survives_a_failed_write(home: Path) -> None:
    fs = RecordingFileOps()
    store = HistoryStore(home, fs=fs)

    store.add_pending("primeiro")

    assert fs.calls == [
        ("write", "history.json.tmp"),
        ("replace", "history.json.tmp->history.json"),
    ]
    original = (home / "history.json").read_bytes()

    fs.fail_writes = True
    store.add_pending("segundo")

    assert (home / "history.json").read_bytes() == original
    assert store.save_failed is True


def test_ut131_repeated_text_gets_distinct_ids(home: Path) -> None:
    store = HistoryStore(home)

    first = store.add_pending("sim")
    second = store.add_pending("sim")

    assert first != second
    assert [e.text for e in store.entries()] == ["sim", "sim"]


def test_ut132_long_text_is_stored_in_full_and_labelled(home: Path) -> None:
    text = "a" * 20_000
    HistoryStore(home).add_pending(text)

    [entry] = HistoryStore(home).entries()

    assert entry.text == text
    assert label(entry) == "a" * 60 + "…"
    short = HistoryEntry("x", "curto", "2026-09-28T10:00:00.000-03:00", HistoryStatus.PENDING)
    assert label(short) == "curto"


def test_ut133_last_returns_newest_or_none(home: Path) -> None:
    store = HistoryStore(home)
    assert store.last() is None

    store.add_pending("um")
    newest = store.add_pending("dois")

    last = store.last()
    assert last is not None
    assert last.id == newest
    assert last.text == "dois"
