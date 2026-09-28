"""Entry point: ``talktype`` and ``python -m talktype``.

Startup order:
1. the single-instance check: another instance means exit 0 with a notice;
2. paths and logging;
3. `prepare_cuda_dlls()`, before anything imports `ctranslate2`;
4. `build_app()` and the Qt event loop.

Exit codes: 0 for a normal exit or when another instance is running, 2 for a fatal
startup error, which is shown in a message box (or only logged under the
`TALKTYPE_TEST_NO_MSGBOX` seam).
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import sys
from collections.abc import Callable

from talktype import seams, win32
from talktype.logging_setup import close_logging, get_logger, log_event, setup_logging
from talktype.paths import Paths
from talktype.singleinstance import SingleInstance
from talktype.strings import Msg

EXIT_OK = 0
EXIT_FATAL = 2

logger = get_logger("main")


def show_message(text: str, *, error: bool = False) -> None:
    """A startup message: `MessageBoxW`, or only the log under the no-msgbox seam."""
    log_event(logger, "startup_message", level=logging.ERROR if error else logging.INFO, msg=text)
    if seams.current().no_msgbox:
        return
    flags = win32.MB_OK | win32.MB_SETFOREGROUND | win32.MB_TOPMOST
    flags |= win32.MB_ICONERROR if error else win32.MB_ICONINFORMATION
    with contextlib.suppress(OSError):
        win32.MessageBoxW(text, "talktype", flags)


def notify_already_running(paths: Paths) -> None:
    """Tell a second launch that talktype is already running."""
    with contextlib.suppress(OSError):
        setup_logging(paths.log_file, console=sys.stderr is not None)
    log_event(logger, "already_running")
    show_message(Msg.ALREADY_RUNNING.text)


def _run(paths: Paths) -> int:
    from talktype.asr_device import prepare_cuda_dlls

    prepare_cuda_dlls()  # must precede every ctranslate2 import

    from PySide6.QtWidgets import QApplication

    from talktype.app import build_app

    qt = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(qt, QApplication)
    qt.setQuitOnLastWindowClosed(False)
    qt.setApplicationName("talktype")
    app = build_app(paths.home)
    atexit.register(app.hook.release_stuck_ctrl)  # never leave a replayed Ctrl down
    qt.aboutToQuit.connect(app.shutdown)
    app.start()
    code = qt.exec()
    app.shutdown()
    return code


def main(
    *,
    instance: SingleInstance | None = None,
    run: Callable[[Paths], int] | None = None,
) -> int:
    paths = Paths.resolve()
    guard = instance or SingleInstance()
    if not guard.acquire():
        notify_already_running(paths)
        close_logging()
        return EXIT_OK
    try:
        try:
            paths.ensure()
            setup_logging(paths.log_file, console=sys.stderr is not None)
            log_event(logger, "startup", home=paths.home, python=sys.version.split()[0])
            seams.log_active(logger)
        except OSError as exc:
            show_message(Msg.STARTUP_FAILED.format(reason=str(exc)), error=True)
            return EXIT_FATAL
        try:
            return (run or _run)(paths)
        except Exception as exc:
            log_event(logger, "startup_failed", level=logging.ERROR, exc_info=True)
            show_message(
                Msg.STARTUP_FAILED.format(reason=f"{type(exc).__name__}: {exc}"), error=True
            )
            return EXIT_FATAL
    finally:
        guard.release()
        log_event(logger, "exit")
        close_logging()


if __name__ == "__main__":
    sys.exit(main())
