"""`ModelStore` (UT-105–UT-109, UT-233, UT-234)."""

from __future__ import annotations

import errno
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, IncompleteSnapshotError, LocalEntryNotFoundError

from talktype.asr import ModelLoadError
from talktype.models import DownloadCancelled, DownloadError, ModelStore, format_size
from talktype.strings import Msg
from tests.conftest import FakeClock

pytestmark = pytest.mark.unit

TURBO_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
Chunks = list[tuple[int, int]]  # (bytes in this update, file total)


class FakeHub:
    """`snapshot_download` stand-in. Online calls replay `chunks` through `tqdm_class`."""

    def __init__(self, tmp_path: Path, *, local: bool = False) -> None:
        self.root = tmp_path / "hub"
        self.local = local
        self.calls: list[dict[str, Any]] = []
        self.chunks: Chunks = [(10, 100), (90, 100)]
        self.online_error: BaseException | None = None
        self.local_error: Callable[[], Exception] = lambda: LocalEntryNotFoundError("absent")
        self.on_chunk: Callable[[int], None] | None = None

    @property
    def downloads(self) -> int:
        return sum(1 for c in self.calls if not c["local_files_only"])

    def snapshot(self, repo_id: str) -> Path:
        return self.root / repo_id.replace("/", "--") / "snapshots" / "abc123"

    def __call__(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, **kwargs})
        path = self.snapshot(repo_id)
        if kwargs["local_files_only"]:
            if not self.local:
                raise self.local_error()
            return str(path)
        if self.online_error is not None:
            raise self.online_error
        bar_class = kwargs["tqdm_class"]
        files = bar_class(total=1, desc="Fetching 1 files")  # the file-count bar is ignored
        transfer = bar_class(total=0, initial=0, unit="B", desc="Downloading bytes")
        written = bar_class(total=0, initial=0, unit="B", desc="Reconstructing")
        total = self.chunks[0][1] if self.chunks else 0
        written.total = total
        for i, (n, _total) in enumerate(self.chunks):
            if self.on_chunk is not None:
                self.on_chunk(i)
            transfer.update(n)
            written.update(n)
        files.update(1)
        # How huggingface_hub 1.33 closes the bars at the end of `snapshot_download`.
        transfer.total = transfer.n
        transfer.refresh()
        transfer.set_description_str("Download complete")
        written.set_description("Reconstruction complete")
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_bytes(b"x")
        return str(path)


def test_ut105_local_snapshot_needs_no_network(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, local=True)
    progress: Chunks = []

    path = ModelStore(snapshot_download=hub).ensure(
        "large-v3-turbo", lambda d, t: progress.append((d, t))
    )

    assert path == hub.snapshot(TURBO_REPO)
    assert [c["local_files_only"] for c in hub.calls] == [True]
    assert hub.calls[0]["repo_id"] == TURBO_REPO
    assert "model.bin" in hub.calls[0]["allow_patterns"]
    assert progress == []


def test_ut106_absent_model_downloads_with_progress(tmp_path: Path, logs: list[str]) -> None:
    hub = FakeHub(tmp_path)
    progress: Chunks = []

    path = ModelStore(snapshot_download=hub).ensure(
        "large-v3-turbo", lambda d, t: progress.append((d, t))
    )

    assert [c["local_files_only"] for c in hub.calls] == [True, False]
    assert progress == [(10, 100), (100, 100)]
    assert path == hub.snapshot(TURBO_REPO)
    assert "model_download model=large-v3-turbo bytes=100 total=100 result=ok" in logs


def test_ut107_corrupt_download_is_deleted_and_downloaded_once_more(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    deleted: list[Path] = []
    loads: list[Path] = []

    def rmtree(path: Path) -> None:
        deleted.append(path)

    def bad_load(path: Path) -> None:
        loads.append(path)
        raise ModelLoadError("model.bin truncated")

    store = ModelStore(snapshot_download=hub, rmtree=rmtree)
    with pytest.raises(ModelLoadError):
        store.ensure("large-v3-turbo", load=bad_load)

    assert hub.downloads == 2
    assert deleted == [hub.snapshot(TURBO_REPO)]
    assert len(loads) == 2


def test_ut107_second_download_that_loads_succeeds(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    attempts: list[Path] = []

    def load(path: Path) -> None:
        attempts.append(path)
        if len(attempts) == 1:
            raise ModelLoadError("corrupt")

    path = ModelStore(snapshot_download=hub, rmtree=lambda p: None).ensure("small", load=load)

    assert hub.downloads == 2
    assert path == attempts[-1]


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://huggingface.co/api/models/x")


@pytest.mark.parametrize(
    ("error", "msg", "fragment"),
    [
        (ConnectionError("network unreachable"), Msg.DOWNLOAD_OFFLINE, "sem conexão"),
        (
            LocalEntryNotFoundError("cannot locate files"),
            None,
            None,
        ),
        (
            HfHubHTTPError("403 Forbidden", response=httpx.Response(403, request=_request())),
            Msg.DOWNLOAD_BLOCKED,
            "huggingface.co",
        ),
        (OSError(errno.ENOSPC, "No space left on device"), Msg.DOWNLOAD_NO_SPACE, "1,6 GB"),
        (
            httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]"),
            Msg.DOWNLOAD_BLOCKED,
            "huggingface.co",
        ),
    ],
)
def test_ut108_error_mapping(
    tmp_path: Path, error: BaseException, msg: Msg | None, fragment: str | None
) -> None:
    hub = FakeHub(tmp_path)
    hub.online_error = error

    if msg is None:  # an unknown failure is re-raised unchanged
        with pytest.raises(type(error)):
            ModelStore(snapshot_download=hub).ensure("large-v3-turbo")
        return
    with pytest.raises(DownloadError) as info:
        ModelStore(snapshot_download=hub).ensure("large-v3-turbo")

    assert info.value.msg is msg
    assert fragment is not None and fragment in info.value.text


def test_ut108_connection_error_wrapped_by_the_hub_is_offline(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    try:
        try:
            raise httpx.ConnectError("getaddrinfo failed")
        except httpx.ConnectError as exc:
            raise LocalEntryNotFoundError("An error happened while trying to locate") from exc
    except LocalEntryNotFoundError as wrapped:
        hub.online_error = wrapped

    with pytest.raises(DownloadError) as info:
        ModelStore(snapshot_download=hub).ensure("large-v3-turbo")

    assert info.value.msg is Msg.DOWNLOAD_OFFLINE


def test_ut108_no_space_message_names_each_model_size() -> None:
    assert format_size(int(3.1 * 1024**3)) == "3,1 GB"


def test_ut109_cancel_stops_the_download_and_the_next_ensure_proceeds(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    hub.chunks = [(10, 100)] * 10
    store = ModelStore(snapshot_download=hub)
    seen: Chunks = []

    def cancel_after_first(i: int) -> None:
        if i == 1:
            store.cancel()

    hub.on_chunk = cancel_after_first
    with pytest.raises(DownloadCancelled):
        store.ensure("large-v3-turbo", lambda d, t: seen.append((d, t)))
    assert seen == [(10, 100)]

    hub.on_chunk = None
    path = store.ensure("small")
    assert path == hub.snapshot("Systran/faster-whisper-small")


def test_ut109_cancel_from_another_thread(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    hub.chunks = [(1, 100)] * 100
    store = ModelStore(snapshot_download=hub)
    started = threading.Event()
    release = threading.Event()

    def pause(i: int) -> None:
        if i == 5:
            started.set()
            assert release.wait(5)

    hub.on_chunk = pause
    errors: list[BaseException] = []

    def run() -> None:
        try:
            store.ensure("large-v3-turbo")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    store.cancel()
    release.set()
    thread.join(5)

    assert len(errors) == 1 and isinstance(errors[0], DownloadCancelled)


@pytest.mark.parametrize(
    "error",
    [
        lambda: LocalEntryNotFoundError("no snapshot"),
        lambda: IncompleteSnapshotError("partial snapshot", snapshot_path="/hub/partial"),
    ],
)
def test_ut233_partial_snapshot_takes_the_download_path(
    tmp_path: Path, error: Callable[[], Exception]
) -> None:
    hub = FakeHub(tmp_path)
    hub.local_error = error
    loaded: list[Path] = []

    path = ModelStore(snapshot_download=hub).ensure("large-v3-turbo", load=loaded.append)

    assert hub.downloads == 1
    assert loaded == [path]
    assert all(not c["local_files_only"] for c in hub.calls[1:])


def test_ut234_progress_has_no_timeout_and_forwards_every_update(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    total = 3 * 1024**3
    hub.chunks = [(total // 1000, total)] * 1000
    clock = FakeClock(0.0)
    hub.on_chunk = lambda i: clock.advance(0.6)  # 1000 chunks over 10 simulated minutes
    progress: Chunks = []

    ModelStore(snapshot_download=hub).ensure("large-v3", lambda d, t: progress.append((d, t)))

    assert clock.now() == pytest.approx(600.0)
    assert len(progress) == 1000
    assert progress[-1][0] == (total // 1000) * 1000
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)
    assert "etag_timeout" not in hub.calls[-1]
