"""`HistoryStore`: the recent-dictation list in `history.json`.

The file is a JSON array of ``{id, text, created_at, status}``, newest first, capped at
`history.size`. Every save writes ``history.json.tmp`` and then `os.replace()`, so a crash
never leaves a half-written file. When the file cannot be written, entries stay in memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import typing
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from talktype.paths import Paths
from talktype.strings import Msg

LABEL_CHARS = 60


class HistoryStatus(StrEnum):
    PENDING = "pending"
    INSERTED = "inserted"
    FAILED_INSERT = "failed_insert"


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    id: str
    text: str
    created_at: str  # ISO-8601 with the local UTC offset
    status: HistoryStatus


def label(entry: HistoryEntry) -> str:
    """Menu label: the first 60 characters on one line, then "…" when longer."""
    text = " ".join(entry.text.split())
    return text if len(text) <= LABEL_CHARS else text[:LABEL_CHARS] + "…"


class FileOps(Protocol):
    """The filesystem calls `HistoryStore` makes (a seam for crash tests)."""

    def read_bytes(self, path: Path) -> bytes: ...
    def write_bytes(self, path: Path, data: bytes) -> None: ...
    def replace(self, src: Path, dst: Path) -> None: ...


class RealFileOps:
    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        with path.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def replace(self, src: Path, dst: Path) -> None:
        os.replace(src, dst)


def _parse_entry(raw: Any) -> HistoryEntry:
    if not isinstance(raw, dict):
        raise ValueError("entry is not an object")
    item = typing.cast(dict[str, Any], raw)
    entry_id, text, created_at = item.get("id"), item.get("text"), item.get("created_at")
    if not (isinstance(entry_id, str) and isinstance(text, str) and isinstance(created_at, str)):
        raise ValueError("entry has missing fields")
    return HistoryEntry(entry_id, text, created_at, HistoryStatus(item.get("status")))


class HistoryStore:
    """Thread-safe; used from the delivery thread and the Qt GUI thread."""

    def __init__(self, home: Path, *, size: int = 20, fs: FileOps | None = None) -> None:
        self.paths = Paths(home)
        self._fs: FileOps = RealFileOps() if fs is None else fs
        self._size = size
        self._lock = threading.Lock()
        self._entries: list[HistoryEntry] = []
        self.load_warning: Msg | None = None
        self.save_failed = False
        self._load()

    @property
    def path(self) -> Path:
        return self.paths.history

    # -- loading -----------------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = self._fs.read_bytes(self.path)
        except FileNotFoundError:
            return
        except OSError:
            self.load_warning = Msg.HISTORY_SAVE_FAILED
            return
        try:
            raw = json.loads(data.decode("utf-8"))
            if not isinstance(raw, list):
                raise ValueError("history is not a list")
            entries = [_parse_entry(item) for item in typing.cast(list[Any], raw)]
        except ValueError:
            self._quarantine(data)
            return
        self._entries = entries[: self._size]

    def _quarantine(self, data: bytes) -> None:
        """Keep the unreadable bytes as `history.corrupt-<ts>.json` and start empty."""
        self.load_warning = Msg.HISTORY_CORRUPT
        target = self.paths.history_corrupt(datetime.now())
        counter = 1
        while target.exists():
            target = target.with_name(f"{target.stem}-{counter}.json")
            counter += 1
        with contextlib.suppress(OSError):
            self._fs.write_bytes(target, data)

    # -- saving ------------------------------------------------------------------------

    def _save_locked(self) -> None:
        payload = json.dumps(
            [asdict(e) for e in self._entries], ensure_ascii=False, indent=1
        ).encode("utf-8")
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fs.write_bytes(tmp, payload)
            self._fs.replace(tmp, self.path)
        except OSError:
            self.save_failed = True
            return
        self.save_failed = False

    # -- public API --------------------------------------------------------------------

    def add_pending(self, text: str) -> str:
        """Record `text` as `pending` (before injection). Returns the new entry id."""
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            text=text,
            created_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            status=HistoryStatus.PENDING,
        )
        with self._lock:
            self._entries = [entry, *self._entries][: self._size]
            self._save_locked()
        return entry.id

    def mark(self, entry_id: str, status: HistoryStatus | str) -> bool:
        """Set the status of `entry_id`. False when the entry is no longer in the history."""
        new_status = HistoryStatus(status)
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    self._entries[index] = replace(entry, status=new_status)
                    self._save_locked()
                    return True
        return False

    def entries(self) -> list[HistoryEntry]:
        """Newest first."""
        with self._lock:
            return list(self._entries)

    def last(self) -> HistoryEntry | None:
        with self._lock:
            return self._entries[0] if self._entries else None

    def get(self, entry_id: str) -> HistoryEntry | None:
        with self._lock:
            return next((e for e in self._entries if e.id == entry_id), None)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._save_locked()

    def set_size(self, size: int) -> None:
        """Apply a new `history.size`; the oldest entries beyond it are dropped."""
        with self._lock:
            self._size = size
            if len(self._entries) > size:
                self._entries = self._entries[:size]
                self._save_locked()
