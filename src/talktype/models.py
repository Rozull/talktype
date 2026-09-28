"""`ModelStore`: model presence, downloads with progress, and corruption recovery.

This is the only network client in the app. `snapshot_download` honors ``HTTPS_PROXY``
and ``HF_HUB_OFFLINE``.
"""

from __future__ import annotations

import errno
import shutil
import ssl
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from talktype.asr import GIB, ModelLoadError, model_spec
from talktype.logging_setup import get_logger, log_event
from talktype.strings import Msg

logger = get_logger("models")

ProgressCallback = Callable[[int, int], None]  # (downloaded bytes, total bytes)
SnapshotDownload = Callable[..., str]
Loader = Callable[[Path], object]


class DownloadCancelled(Exception):  # noqa: N818 - public name
    """`ModelStore.cancel()` stopped the download in progress."""


class DownloadError(Exception):
    """A download failed; `msg` and `text` are the user-facing message."""

    def __init__(self, msg: Msg, text: str) -> None:
        super().__init__(text)
        self.msg = msg
        self.text = text


def _default_snapshot_download(*args: Any, **kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def _hub_host() -> str:
    from huggingface_hub import constants

    return urlparse(constants.ENDPOINT).hostname or "huggingface.co"


def format_size(n_bytes: int) -> str:
    """``1.6 GiB`` → ``"1,6 GB"`` (Portuguese decimal comma)."""
    return f"{n_bytes / GIB:.1f} GB".replace(".", ",")


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _hub_cache() -> str:
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def map_download_error(
    exc: BaseException, model_id: str, cache_dir: Path | None = None
) -> DownloadError | None:
    """The user-facing error for a download failure, or None when it is not a known kind."""
    import httpx

    chain = list(_chain(exc))
    for item in chain:
        if isinstance(item, OSError) and item.errno == errno.ENOSPC:
            required = format_size(model_spec(model_id).download_bytes)
            return DownloadError(
                Msg.DOWNLOAD_NO_SPACE, Msg.DOWNLOAD_NO_SPACE.format(required=required)
            )
    for item in chain:
        if isinstance(item, PermissionError):
            path = cache_dir or _hub_cache()
            return DownloadError(
                Msg.MODEL_CACHE_NOT_WRITABLE, Msg.MODEL_CACHE_NOT_WRITABLE.format(path=path)
            )
    for item in chain:
        text = str(item).lower()
        blocked = (
            isinstance(item, (ssl.SSLError, httpx.ProxyError))
            or _status_code(item) in (403, 407)
            or (isinstance(item, httpx.ConnectError) and ("ssl" in text or "certificate" in text))
        )
        if blocked:
            host = _hub_host()
            return DownloadError(Msg.DOWNLOAD_BLOCKED, Msg.DOWNLOAD_BLOCKED.format(host=host))
    for item in chain:
        if isinstance(item, (ConnectionError, httpx.TransportError)):
            return DownloadError(Msg.DOWNLOAD_OFFLINE, Msg.DOWNLOAD_OFFLINE.format())
    return None


class _Progress:
    """Aggregates the byte progress bars of one download and forwards every update."""

    def __init__(self, callback: ProgressCallback | None, cancelled: threading.Event) -> None:
        self.callback = callback
        self.cancelled = cancelled
        self.lock = threading.Lock()
        self.downloaded = 0
        self.total = 0
        self.reported = False

    def forward(self) -> None:
        if self.cancelled.is_set():
            raise DownloadCancelled
        if self.callback is not None:
            self.reported = True
            self.callback(self.downloaded, self.total)

    def bar_class(self) -> type[Any]:
        progress = self

        class ProgressBar:
            """Minimal tqdm stand-in for `snapshot_download(tqdm_class=...)`.

            Only the byte bar that tracks bytes written ("Reconstructing …" in
            huggingface_hub 1.x) is forwarded; the file-count and network-transfer bars
            are accepted and ignored. There is no timeout: every chunk just reports.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del args
                self.n = kwargs.get("initial", 0) or 0
                self._total = kwargs.get("total", 0) or 0
                desc = str(kwargs.get("desc") or "")
                self.tracked = kwargs.get("unit") == "B" and not desc.startswith("Downloading")
                self.format_dict: dict[str, Any] = {"rate": None}

            @property
            def total(self) -> int:
                return self._total

            @total.setter
            def total(self, value: int | None) -> None:
                self._total = int(value or 0)
                if self.tracked:
                    with progress.lock:
                        progress.total = self._total

            def update(self, n: float | None = 1) -> None:
                if progress.cancelled.is_set():
                    raise DownloadCancelled
                self.n += n or 0
                if self.tracked:
                    with progress.lock:
                        progress.downloaded = int(self.n)
                        progress.total = max(progress.total, self._total)
                    progress.forward()

            def refresh(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def set_description(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def set_description_str(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs  # "Download complete" on the transfer bar (hub 1.33)

            def set_postfix(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def close(self) -> None:
                pass

            def __enter__(self) -> ProgressBar:
                return self

            def __exit__(self, *exc: object) -> None:
                pass

        return ProgressBar


class ModelStore:
    """Finds or downloads model snapshots in the Hugging Face cache.

    `ensure()` runs on the ASR loader thread; `cancel()` may be called from any thread.
    """

    def __init__(
        self,
        *,
        snapshot_download: SnapshotDownload | None = None,
        cache_dir: Path | None = None,
        rmtree: Callable[[Path], None] | None = None,
    ) -> None:
        self._snapshot_download = snapshot_download or _default_snapshot_download
        self._cache_dir = cache_dir
        self._rmtree = rmtree or (lambda p: shutil.rmtree(p, ignore_errors=True))
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.downloads = 0  # number of network downloads started, for diagnostics

    def cancel(self) -> None:
        """Abandon the download in progress; it raises `DownloadCancelled` at its next chunk."""
        with self._lock:
            self._cancel_event.set()

    def ensure(
        self,
        model_id: str,
        progress_cb: ProgressCallback | None = None,
        *,
        load: Loader | None = None,
    ) -> Path:
        """Return the snapshot directory of `model_id`, downloading it when absent.

        With `load`, the snapshot is also loaded; a `ModelLoadError` deletes the snapshot
        and downloads it once more, and a second failure raises `ModelLoadError`.
        Raises `DownloadError` (mapped message), `DownloadCancelled` or `ModelLoadError`.
        """
        with self._lock:
            cancelled = threading.Event()
            self._cancel_event = cancelled
        path = self._local(model_id)
        downloaded = False
        if path is None:
            path = self._download(model_id, progress_cb, cancelled)
            downloaded = True
        if load is None:
            return path
        try:
            load(path)
            return path
        except ModelLoadError as exc:
            log_event(logger, "model_corrupt", model=model_id, downloaded=downloaded, exc=str(exc))
        self._rmtree(path)
        path = self._download(model_id, progress_cb, cancelled)
        load(path)  # a second ModelLoadError propagates
        return path

    def _kwargs(self, model_id: str) -> dict[str, Any]:
        spec = model_spec(model_id)
        kwargs: dict[str, Any] = {"allow_patterns": list(spec.allow_patterns)}
        if self._cache_dir is not None:
            kwargs["cache_dir"] = str(self._cache_dir)
        return kwargs

    def _local(self, model_id: str) -> Path | None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        spec = model_spec(model_id)
        try:
            return Path(
                self._snapshot_download(
                    spec.repo_id, local_files_only=True, **self._kwargs(model_id)
                )
            )
        except (LocalEntryNotFoundError, FileNotFoundError):
            return None

    def _download(
        self, model_id: str, progress_cb: ProgressCallback | None, cancelled: threading.Event
    ) -> Path:
        spec = model_spec(model_id)
        progress = _Progress(progress_cb, cancelled)
        self.downloads += 1
        result = "ok"
        try:
            if cancelled.is_set():
                raise DownloadCancelled
            path = Path(
                self._snapshot_download(
                    spec.repo_id,
                    local_files_only=False,
                    tqdm_class=progress.bar_class(),
                    **self._kwargs(model_id),
                )
            )
        except DownloadCancelled:
            result = "cancelled"
            raise
        except Exception as exc:
            mapped = map_download_error(exc, model_id, self._cache_dir)
            result = "error" if mapped is None else mapped.msg.name.lower()
            if mapped is None:
                raise
            raise mapped from exc
        finally:
            log_event(
                logger,
                "model_download",
                model=model_id,
                bytes=progress.downloaded,
                total=progress.total,
                result=result,
            )
        return path
