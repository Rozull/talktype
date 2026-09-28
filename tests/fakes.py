"""Fakes at the I/O boundaries.

`FakeEngine` implements `SpeechEngine`; `FakeNvml` mimics the four `pynvml` calls;
`FakeSoundDevice` mimics the `sounddevice` module; `FakeRegistry` is a registry reader.
`FakeInjector`, `FakeRewriter` and `FakeHttp` stand in for delivery's injector, rewriter and
HTTP client; `FakeWin32` is an in-memory desktop for `foreground`, `clipboard` and `injector`.
"""

from __future__ import annotations

import ctypes
import heapq
import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from talktype import win32
from talktype.asr import Audio, DeviceKind
from talktype.config import InjectionSettings, LlmSettings
from talktype.injector import InjectMode, InjectResult, InjectStatus
from tests.conftest import FakeClock

GIB = 1024**3


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


@dataclass
class TranscribeCall:
    samples: int
    language: str
    vocabulary: tuple[str, ...]
    fast: bool
    engine: FakeEngine


class FakeEngine:
    """Scriptable `SpeechEngine`. `order` records load / warm_up / transcribe / close."""

    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        device: DeviceKind = "cuda",
        *,
        text: str | Callable[[Audio], str] = "olá",
        supports_vocabulary: bool = True,
        log: list[TranscribeCall] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device: DeviceKind = device
        self.supports_vocabulary = supports_vocabulary
        self.text = text
        self.order: list[str] = []
        self.calls: list[TranscribeCall] = [] if log is None else log
        self.errors: list[Exception] = []  # raised by the next transcribe calls, in order
        self.load_error: Exception | None = None
        self.load_gate: threading.Event | None = None
        self.transcribe_gate: threading.Event | None = None
        self.transcribing = threading.Event()
        self.last_vocab_dropped = 0

    def load(self) -> None:
        self.order.append("load")
        if self.load_gate is not None:
            assert self.load_gate.wait(10)
        if self.load_error is not None:
            raise self.load_error

    def warm_up(self) -> None:
        self.order.append("warm_up")

    def transcribe(
        self, audio: Audio, *, language: str, vocabulary: Sequence[str], fast: bool
    ) -> str:
        self.order.append("transcribe")
        self.calls.append(TranscribeCall(audio.size, language, tuple(vocabulary), fast, self))
        self.transcribing.set()
        if self.transcribe_gate is not None:
            assert self.transcribe_gate.wait(10)
        if self.errors:
            raise self.errors.pop(0)
        return self.text(audio) if callable(self.text) else self.text

    def close(self) -> None:
        self.order.append("close")


class FakeEngineFactory:
    """`engine_factory` for `AsrWorker`; `prepare` customizes each new engine."""

    def __init__(self, prepare: Callable[[FakeEngine], None] | None = None) -> None:
        self.created: list[FakeEngine] = []
        self.calls: list[TranscribeCall] = []
        self.prepare = prepare

    def __call__(
        self, model_id: str, device: DeviceKind, compute_type: str, path: Path | None
    ) -> FakeEngine:
        del compute_type, path
        engine = FakeEngine(model_id, device, log=self.calls)
        if self.prepare is not None:
            self.prepare(engine)
        self.created.append(engine)
        return engine

    def by(self, model_id: str, device: DeviceKind | None = None) -> list[FakeEngine]:
        return [
            e
            for e in self.created
            if e.model_id == model_id and (device is None or e.device == device)
        ]


class FakeStore:
    """`ModelStore` stand-in: every model is present at `/models/<id>`."""

    def __init__(self) -> None:
        self.ensured: list[str] = []
        self.cancelled = 0

    def ensure(
        self,
        model_id: str,
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        load: Callable[[Path], object] | None = None,
    ) -> Path:
        del progress_cb
        self.ensured.append(model_id)
        path = Path("/models") / model_id
        if load is not None:
            load(path)
        return path

    def cancel(self) -> None:
        self.cancelled += 1


# --------------------------------------------------------------------------------------
# NVML
# --------------------------------------------------------------------------------------


class NVMLError(Exception):
    pass


@dataclass
class _MemInfo:
    free: int
    used: int
    total: int


class FakeNvml:
    """The `pynvml` calls `nvml_memory()` makes; `fail_init` raises `NVMLError`."""

    def __init__(self, free: float = 8 * GIB, *, fail_init: bool = False) -> None:
        self.free = int(free)
        self.fail_init = fail_init
        self.calls: list[str] = []

    def nvmlInit(self) -> None:  # noqa: N802
        self.calls.append("init")
        if self.fail_init:
            raise NVMLError("NVML Shared Library Not Found")

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:  # noqa: N802
        self.calls.append(f"handle:{index}")
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _MemInfo:  # noqa: N802
        del handle
        self.calls.append("meminfo")
        return _MemInfo(self.free, 12 * GIB - self.free, 12 * GIB)

    def nvmlShutdown(self) -> None:  # noqa: N802
        self.calls.append("shutdown")


# --------------------------------------------------------------------------------------
# sounddevice
# --------------------------------------------------------------------------------------


class PortAudioError(Exception):
    pass


@dataclass
class _Default:
    device: list[int] = field(default_factory=lambda: [0, 0])


class FakeStream:
    def __init__(self, owner: FakeSoundDevice, **kwargs: Any) -> None:
        self.owner = owner
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs.get("finished_callback")
        self.samplerate = kwargs["samplerate"]
        self.blocksize = kwargs["blocksize"]
        self.active = False
        self.closed = False

    def start(self) -> None:
        if self.owner.start_error is not None:
            raise self.owner.start_error
        self.active = True

    def stop(self) -> None:
        self.active = False
        if self.finished_callback is not None:
            self.finished_callback()

    def close(self) -> None:
        self.closed = True

    def feed(self, block: Audio) -> None:
        """Deliver one block through the PortAudio callback."""
        self.callback(block.reshape(-1, 1).astype(np.float32), block.size, None, None)

    def unplug(self) -> None:
        """The device disappears: the stream ends without `stop()`."""
        self.active = False
        if self.finished_callback is not None:
            self.finished_callback()


class FakeSoundDevice:
    """The parts of the `sounddevice` module `Recorder` uses."""

    PortAudioError = PortAudioError

    def __init__(
        self,
        devices: list[dict[str, Any]] | None = None,
        default_input: int = 0,
        *,
        rates: dict[int, list[int]] | None = None,
    ) -> None:
        self.devices = devices if devices is not None else [device("Microfone (Realtek)")]
        self.default = _Default([default_input, -1])
        self.rates = rates or {}  # device index -> supported rates (default: any)
        self.query_count = 0
        self.streams: list[FakeStream] = []
        self.start_error: Exception | None = None

    def query_devices(self) -> list[dict[str, Any]]:
        self.query_count += 1
        return list(self.devices)

    def check_input_settings(
        self, device: Any = None, channels: Any = None, dtype: Any = None, samplerate: Any = None
    ) -> None:
        del channels, dtype
        supported = self.rates.get(int(device))
        if supported is not None and int(samplerate) not in supported:
            raise PortAudioError("Invalid sample rate", -9997)

    def InputStream(self, *args: Any, **kwargs: Any) -> FakeStream:  # noqa: N802
        del args
        stream = FakeStream(self, **kwargs)
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeStream:
        return self.streams[-1]


def device(
    name: str, *, inputs: int = 1, hostapi: int = 0, rate: float = 44100.0
) -> dict[str, Any]:
    return {
        "name": name,
        "max_input_channels": inputs,
        "hostapi": hostapi,
        "default_samplerate": rate,
    }


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


class FakeRegistry:
    """A `reg_read_str(root, subkey, name)` reader over a dict; records every read.

    It also implements `autostart.Registry` (`read`, `write`, `delete`). Set `fail_writes`
    to an exception to make writes and deletes raise it (e.g. a policy `PermissionError`).
    """

    def __init__(self, values: dict[tuple[str, str], str] | None = None) -> None:
        self.values = values or {}
        self.reads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, str]] = []
        self.fail_writes: OSError | None = None

    def __call__(self, root: int, subkey: str, name: str) -> str | None:
        del root
        self.reads.append((subkey, name))
        return self.values.get((subkey, name))

    def read(self, root: int, subkey: str, name: str) -> str | None:
        return self(root, subkey, name)

    def write(self, root: int, subkey: str, name: str, value: str) -> None:
        del root
        if self.fail_writes is not None:
            raise self.fail_writes
        self.writes.append((subkey, name, value))
        self.values[(subkey, name)] = value

    def delete(self, root: int, subkey: str, name: str) -> bool:
        del root
        if self.fail_writes is not None:
            raise self.fail_writes
        return self.values.pop((subkey, name), None) is not None


def sine(seconds: float, rate: int = 16000, amplitude: float = 0.5, freq: float = 440.0) -> Audio:
    t = np.arange(round(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Delivery: injector, rewriter, HTTP
# --------------------------------------------------------------------------------------


class FakeInjector:
    """Records ``inject(text, mode)`` calls as ``(text, mode)`` and returns scripted results.

    `results` (an `InjectResult` or an `InjectStatus`) are returned in order, then `default`
    (INSERTED). `errors` are raised by the next calls instead. `delay_s` or `gate` hold each
    call; `started` is set once a call begins. `clipboard` holds the text a failure left on
    the clipboard. `order` (shared with other fakes) records "inject".
    """

    def __init__(
        self,
        results: Sequence[InjectResult | InjectStatus] = (),
        *,
        delay_s: float = 0.0,
        order: list[str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.settings: list[InjectionSettings | None] = []
        self.results: list[InjectResult | InjectStatus] = list(results)
        self.errors: list[BaseException] = []
        self.default = InjectResult(InjectStatus.INSERTED, clipboard_restored=True)
        self.delay_s = delay_s
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.order = order
        self.clipboard: str | None = None
        self.threads: set[int] = set()
        self.trigger_vk = win32.VK_RCONTROL
        self.closed = False

    def inject(
        self, text: str, mode: InjectMode | str, settings: InjectionSettings | None = None
    ) -> InjectResult:
        self.calls.append((text, str(mode)))
        self.settings.append(settings)
        self.threads.add(threading.get_ident())
        if self.order is not None:
            self.order.append("inject")
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.errors:
            raise self.errors.pop(0)
        result = self.results.pop(0) if self.results else self.default
        if isinstance(result, InjectStatus):
            failed = result is not InjectStatus.INSERTED
            result = InjectResult(result, text_on_clipboard=failed, clipboard_restored=not failed)
        if result.text_on_clipboard:
            self.clipboard = text
        return result

    def put_on_clipboard(self, text: str) -> bool:
        self.clipboard = text
        return True

    def close(self) -> None:
        self.closed = True


class FakeRewriter:
    """A `Rewriter`: returns `output` (text or function; None echoes) or raises `error`.

    `calls` records the texts; `factory` is a `rewriter_factory` that records the settings.
    """

    def __init__(
        self,
        output: str | Callable[[str], str] | None = None,
        *,
        error: Exception | None = None,
        delay_s: float = 0.0,
        order: list[str] | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.delay_s = delay_s
        self.order = order
        self.calls: list[str] = []
        self.settings: list[LlmSettings] = []
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.prewarms = 0

    def rewrite(self, text: str) -> str:
        self.calls.append(text)
        if self.order is not None:
            self.order.append("rewrite")
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.output is None:
            return text
        return self.output(text) if callable(self.output) else self.output

    def prewarm(self) -> None:
        self.prewarms += 1

    def factory(self, s: LlmSettings) -> FakeRewriter:
        self.settings.append(s)
        return self


class FakeHttp:
    """An `httpx.Client` over `httpx.MockTransport` with one scripted answer.

    `body` is sent as JSON with `status`; `error` (an exception, or a function of the request
    returning one) is raised instead. `requests` records every request.
    """

    def __init__(
        self,
        body: Any = None,
        *,
        status: int = 200,
        error: Exception | Callable[[httpx.Request], Exception] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.error = error
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error if isinstance(self.error, Exception) else self.error(request)
        return httpx.Response(self.status, json=self.body)

    def payload(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def ollama_reply(content: str) -> dict[str, Any]:
    return {"model": "qwen2.5:7b-instruct", "message": {"role": "assistant", "content": content}}


def openai_reply(content: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


# --------------------------------------------------------------------------------------
# Win32 desktop
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """One keyboard ``INPUT`` passed to ``SendInput``."""

    vk: int
    scan: int
    flags: int

    @property
    def up(self) -> bool:
        return bool(self.flags & win32.KEYEVENTF_KEYUP)

    @property
    def unicode(self) -> bool:
        return bool(self.flags & win32.KEYEVENTF_UNICODE)


class FakeWin32:
    """The `win32` functions `foreground`, `clipboard` and `injector` call, in memory.

    The desktop has a foreground window (`TARGET_HWND`, class "Notepad"), a clipboard with
    owner and delayed rendering, a message queue driven by `clock`, and a target app: after a
    Ctrl+V it reads ``CF_UNICODETEXT`` `paste_render_ms` later (None: it ignores the paste),
    which sends ``WM_RENDERFORMAT`` to the owner window. `SendInput` batches are recorded as
    `KeyEvent` lists; typed text accumulates in `typed`. `clipboard_calls` lists every
    clipboard API call made by talktype.
    """

    TARGET_HWND = 0x1000
    OTHER_HWND = 0x2000  # another app that copies to the clipboard
    _TARGET_APP = -2  # who has the clipboard open while the target reads it

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock or FakeClock(1000.0)
        # desktop and processes
        self.foreground = self.TARGET_HWND
        self.classes: dict[int, str] = {self.TARGET_HWND: "Notepad", self.OTHER_HWND: "Chrome"}
        self.pids: dict[int, int] = {self.TARGET_HWND: 4242, self.OTHER_HWND: 4343}
        self.own_pid = 100
        self.self_elevated = False
        self.elevated_pids: set[int] = set()
        self.denied_pids: set[int] = set()
        # keyboard
        self.batches: list[list[KeyEvent]] = []
        self.send_times: list[float] = []
        self.key_release_at: dict[int, float] = {}  # vk -> clock time the key goes up
        self.on_send: Callable[[int], None] | None = None  # called with the batch count
        self.send_error: OSError | None = None
        self.typed = ""
        self.submits = 0  # bare Enter presses the target received
        self._ctrl = self._shift = False
        # clipboard
        self.data: dict[int, bytes | None] = {}  # None: delayed rendering
        self.owner = 0
        self.opened_by: int | None = None
        self.locked = False
        self.bad_formats: set[int] = set()  # GlobalLock fails on handles of these formats
        self.clipboard_calls: list[str] = []
        self.sets: list[tuple[int, bytes | None]] = []
        self.seq = 1
        self._formats: dict[str, int] = {}
        self._memory: dict[int, bytes] = {}
        self._bad_handles: set[int] = set()
        self._next_handle = 0x10000
        # windows and messages
        self.windows: dict[int, Callable[[int, int, int, int], int]] = {}
        self._next_hwnd = 0x9000
        self._events: list[tuple[float, int, Callable[[], None]]] = []
        self._event_count = 0
        # the target app
        self.paste_render_ms: float | None = 50.0
        self.pasted: list[str] = []
        self.after_paste: Callable[[], None] | None = None

    # -- helpers for tests -------------------------------------------------------------

    def schedule(self, delay_s: float, action: Callable[[], None]) -> None:
        self._event_count += 1
        heapq.heappush(self._events, (self.clock.now() + delay_s, self._event_count, action))

    def seed(self, content: str | dict[int, bytes]) -> None:
        """Another app puts `content` on the clipboard."""
        data = (
            {win32.CF_UNICODETEXT: win32.unicode_text_bytes(content)}
            if isinstance(content, str)
            else dict(content)
        )
        self._take_ownership(self.OTHER_HWND)
        self.data = dict(data)

    def user_copy(self, text: str) -> None:
        """The user copies `text` in another app (it becomes the clipboard owner)."""
        self.seed(text)

    def text(self) -> str | None:
        data = self.data.get(win32.CF_UNICODETEXT)
        return None if data is None else win32.text_from_unicode_bytes(data)

    def format_id(self, name: str) -> int:
        return self._formats.setdefault(name, 0xC100 + len(self._formats))

    def events(self) -> list[KeyEvent]:
        return [event for batch in self.batches for event in batch]

    def _take_ownership(self, owner: int) -> None:
        previous = self.owner
        if previous in self.windows:
            self.windows[previous](previous, win32.WM_DESTROYCLIPBOARD, 0, 0)
        self.data = {}
        self.owner = owner
        self.seq += 1

    def _alloc(self, data: bytes) -> int:
        self._next_handle += 1
        self._memory[self._next_handle] = bytes(data)
        return self._next_handle

    def _render(self, fmt: int) -> bytes | None:
        if fmt not in self.data:
            return None
        if self.data[fmt] is None and self.owner in self.windows:
            self.windows[self.owner](self.owner, win32.WM_RENDERFORMAT, fmt, 0)
        return self.data.get(fmt)

    def _target_reads(self) -> None:
        """The target app handles Ctrl+V: it opens the clipboard and reads the text."""
        if self.locked or self.opened_by is not None:
            return
        self.opened_by = self._TARGET_APP
        try:
            data = self._render(win32.CF_UNICODETEXT)
        finally:
            self.opened_by = None
        if data is not None:
            self.pasted.append(win32.text_from_unicode_bytes(data))
        if self.after_paste is not None:
            self.after_paste()

    # -- foreground and processes ------------------------------------------------------

    def GetForegroundWindow(self) -> int:  # noqa: N802
        return self.foreground

    def GetClassNameW(self, hwnd: int) -> str:  # noqa: N802
        if hwnd not in self.classes:
            raise ctypes.WinError(win32.ERROR_INVALID_WINDOW_HANDLE)
        return self.classes[hwnd]

    def GetWindowThreadProcessId(self, hwnd: int) -> tuple[int, int]:  # noqa: N802
        return 1, self.pids.get(hwnd, 0)

    def GetCurrentProcessId(self) -> int:  # noqa: N802
        return self.own_pid

    def current_process_is_elevated(self) -> bool:
        return self.self_elevated

    def OpenProcess(self, access: int, pid: int, inherit: bool = False) -> int:  # noqa: N802
        del access, inherit
        if pid in self.denied_pids:
            raise ctypes.WinError(win32.ERROR_ACCESS_DENIED)
        return 0x50000 + pid

    def OpenProcessToken(self, process: int, access: int = win32.TOKEN_QUERY) -> int:  # noqa: N802
        del access
        return process + 0x100000

    def token_is_elevated(self, token: int) -> bool:
        return token - 0x150000 in self.elevated_pids

    def CloseHandle(self, handle: int) -> bool:  # noqa: N802
        del handle
        return True

    # -- keyboard ----------------------------------------------------------------------

    def is_key_down(self, vk: int) -> bool:
        return self.clock.now() < self.key_release_at.get(vk, -math.inf)

    def GetAsyncKeyState(self, vk: int) -> int:  # noqa: N802
        return -0x8000 if self.is_key_down(vk) else 0

    def SendInput(self, inputs: Sequence[win32.INPUT]) -> int:  # noqa: N802
        if self.send_error is not None:
            raise self.send_error
        batch = [KeyEvent(i.ki.wVk, i.ki.wScan, i.ki.dwFlags) for i in inputs]
        self.batches.append(batch)
        self.send_times.append(self.clock.now())
        self._receive(batch)
        if self.on_send is not None:
            self.on_send(len(self.batches))
        return len(inputs)

    def _receive(self, batch: list[KeyEvent]) -> None:
        units: list[int] = []

        def flush() -> None:
            data = b"".join(u.to_bytes(2, "little") for u in units)
            self.typed += data.decode("utf-16-le", errors="replace")
            units.clear()

        for event in batch:
            if event.unicode:
                if not event.up:
                    units.append(event.scan)
                continue
            flush()
            if event.vk == win32.VK_CONTROL:
                self._ctrl = not event.up
            elif event.vk == win32.VK_SHIFT:
                self._shift = not event.up
            elif event.vk == win32.VK_RETURN and not event.up:
                if self._shift:
                    self.typed += "\n"
                else:
                    self.submits += 1
            elif (
                event.vk == win32.VK_V
                and not event.up
                and self._ctrl
                and self.paste_render_ms is not None
            ):
                self.schedule(self.paste_render_ms / 1000, self._target_reads)
        flush()

    # -- clipboard ---------------------------------------------------------------------

    def OpenClipboard(self, hwnd: int = 0) -> bool:  # noqa: N802
        self.clipboard_calls.append("OpenClipboard")
        if self.locked or self.opened_by is not None:
            return False
        self.opened_by = hwnd
        return True

    def CloseClipboard(self) -> bool:  # noqa: N802
        self.clipboard_calls.append("CloseClipboard")
        was_open = self.opened_by is not None
        self.opened_by = None
        return was_open

    def EmptyClipboard(self) -> None:  # noqa: N802
        self.clipboard_calls.append("EmptyClipboard")
        if self.opened_by is None:
            raise ctypes.WinError(win32.ERROR_CLIPBOARD_NOT_OPEN)
        self._take_ownership(max(self.opened_by, 0))

    def SetClipboardData(self, fmt: int, handle: int = 0) -> int:  # noqa: N802
        self.clipboard_calls.append("SetClipboardData")
        if self.opened_by is None:
            return 0
        value = self._memory.pop(handle, b"") if handle else None
        self.data[fmt] = value
        self.sets.append((fmt, value))
        return handle

    def GetClipboardData(self, fmt: int) -> int:  # noqa: N802
        self.clipboard_calls.append("GetClipboardData")
        if self.opened_by is None:
            return 0
        data = self._render(fmt)
        if data is None:
            return 0
        handle = self._alloc(data)
        if fmt in self.bad_formats:
            self._bad_handles.add(handle)
        return handle

    def clipboard_formats(self) -> list[int]:
        self.clipboard_calls.append("EnumClipboardFormats")
        return list(self.data) if self.opened_by is not None else []

    def GetClipboardOwner(self) -> int:  # noqa: N802
        self.clipboard_calls.append("GetClipboardOwner")
        return self.owner

    def GetClipboardSequenceNumber(self) -> int:  # noqa: N802
        return self.seq

    def GetOpenClipboardWindow(self) -> int:  # noqa: N802
        if self.opened_by == self._TARGET_APP:
            return self.TARGET_HWND
        return max(self.opened_by or 0, 0)

    def RegisterClipboardFormatW(self, name: str) -> int:  # noqa: N802
        self.clipboard_calls.append("RegisterClipboardFormatW")
        return self.format_id(name)

    def global_from_bytes(self, data: bytes) -> int:
        return self._alloc(data)

    def global_to_bytes(self, handle: int) -> bytes:
        if handle in self._bad_handles:
            raise ctypes.WinError(win32.ERROR_INVALID_HANDLE)  # GlobalLock failed
        return self._memory[handle]

    def GlobalFree(self, handle: int) -> None:  # noqa: N802
        self._memory.pop(handle, None)

    def SetEnhMetaFileBits(self, data: bytes) -> int:  # noqa: N802
        return self._alloc(data)

    def GetEnhMetaFileBits(self, handle: int) -> bytes:  # noqa: N802
        return self._memory[handle]

    def DeleteEnhMetaFile(self, handle: int) -> bool:  # noqa: N802
        return self._memory.pop(handle, None) is not None

    # -- windows and messages ----------------------------------------------------------

    @staticmethod
    def WNDPROC(fn: Callable[[int, int, int, int], int]) -> Callable[[int, int, int, int], int]:  # noqa: N802
        return fn

    def create_message_window(self, class_name: str, proc: Callable[..., int]) -> int:
        del class_name
        self._next_hwnd += 1
        self.windows[self._next_hwnd] = proc
        return self._next_hwnd

    def destroy_message_window(self, hwnd: int, class_name: str) -> None:
        del class_name
        proc = self.windows.get(hwnd)
        owes = any(value is None for value in self.data.values())
        if proc is not None and self.owner == hwnd and owes:
            proc(hwnd, win32.WM_RENDERALLFORMATS, 0, 0)
        self.windows.pop(hwnd, None)

    def DefWindowProcW(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:  # noqa: N802
        del hwnd, msg, wparam, lparam
        return 0

    def MsgWaitForMultipleObjects(  # noqa: N802
        self, handles: Sequence[int], wait_all: bool, timeout_ms: int, wake_mask: int = 0
    ) -> int:
        del wait_all, wake_mask
        now = self.clock.now()
        limit = now + timeout_ms / 1000
        if self._events and self._events[0][0] <= limit:
            self.clock.set(max(now, self._events[0][0]))
            return win32.WAIT_OBJECT_0 + len(handles)
        self.clock.set(limit)
        return win32.WAIT_TIMEOUT

    def pump_pending_messages(self) -> bool:
        while self._events and self._events[0][0] <= self.clock.now():
            _, _, action = heapq.heappop(self._events)
            action()
        return True
