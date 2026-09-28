"""One talktype per Windows user session, through a named mutex.

The ``Local\\`` namespace scopes the mutex to the session, so two signed-in users each
get their own instance. Windows closes the handle when the process dies,
so a crashed instance never blocks the next launch.
"""

from __future__ import annotations

from collections.abc import Callable

from talktype import win32

MUTEX_NAME = "Local\\talktype-singleton"

CreateMutex = Callable[[str], tuple[int, bool]]


class SingleInstance:
    """Holds the mutex handle for the lifetime of the process."""

    def __init__(
        self,
        name: str = MUTEX_NAME,
        *,
        create_mutex: CreateMutex = win32.CreateMutexW,
        close_handle: Callable[[int], bool] = win32.CloseHandle,
    ) -> None:
        self.name = name
        self._create = create_mutex
        self._close = close_handle
        self._handle = 0
        self.already_running = False

    def acquire(self) -> bool:
        """True when this process is the only instance. Idempotent."""
        if self._handle:
            return not self.already_running
        handle, existed = self._create(self.name)
        self._handle = handle
        self.already_running = existed
        if existed:
            self.release()
        return not existed

    def release(self) -> None:
        if self._handle:
            self._close(self._handle)
            self._handle = 0

    def __enter__(self) -> SingleInstance:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def acquire(name: str = MUTEX_NAME) -> SingleInstance | None:
    """The held guard, or None when another instance already owns `name`."""
    guard = SingleInstance(name)
    return guard if guard.acquire() else None
