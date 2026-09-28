"""`App` controller with fakes at the I/O boundaries (UT-162–UT-187, UT-199, UT-208–UT-246)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from talktype import __main__ as entry
from talktype import config as config_module
from talktype import seams
from talktype.app import App, AppState, Bridge, DirectInjectDone
from talktype.asr import GIB, ModelLoadError
from talktype.asr_device import Device, select_device
from talktype.asr_worker import (
    AsrWorker,
    DownloadProgress,
    EngineFailed,
    EngineReady,
    PreviewResult,
)
from talktype.audio import Level, MicError, OpenInfo
from talktype.config import ConfigManager, set_value
from talktype.delivery import DeliveryOutcome, DeliveryStatus
from talktype.dictation import AsrOutcome, Dictation, DictationMode, DictationState
from talktype.history import HistoryStatus, HistoryStore
from talktype.injector import InjectResult, InjectStatus
from talktype.models import ModelStore
from talktype.overlay import countdown_text
from talktype.paths import Paths
from talktype.singleinstance import SingleInstance
from talktype.sounds import Cue, Sounds, builtin_path, wav_image
from talktype.strings import Msg
from talktype.tray import Tray, TrayState
from talktype.trigger import Effect
from talktype.winhook import RepasteRequested, SessionInterrupted
from tests.conftest import FakeClock
from tests.fakes import FakeEngineFactory, FakeNvml, FakeStore

pytestmark = pytest.mark.unit

TURBO = "large-v3-turbo"
CUDA = Device("cuda", "int8_float16", "auto", 8000)
CPU = Device("cpu", "int8", "no_gpu")
INSERTED = InjectResult(InjectStatus.INSERTED, False, True, False)


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeRecorder:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.microphone = ""
        self.open_error: MicError | None = None
        self.info = OpenInfo("Microfone (fake)", 16000)
        self.snapshots: list[float | None] = []
        self._buffer = np.zeros(0, dtype=np.float32)
        self.stopped_audio: list[int] = []

    @property
    def buffer_bytes(self) -> int:
        return self._buffer.nbytes

    def open(self) -> OpenInfo:
        self.order.append("recorder.open")
        if self.open_error is not None:
            raise self.open_error
        self._buffer = np.full(16000, 0.1, dtype=np.float32)
        return self.info

    def feed(self, seconds: float) -> None:
        more = np.full(round(seconds * 16000), 0.1, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, more])

    def stop(self) -> np.ndarray:
        self.order.append("recorder.stop")
        audio, self._buffer = self._buffer, np.zeros(0, dtype=np.float32)
        self.stopped_audio.append(audio.size)
        return audio

    def discard(self) -> None:
        self.order.append("recorder.discard")
        self._buffer = np.zeros(0, dtype=np.float32)

    def snapshot(self, seconds: float | None = None) -> np.ndarray:
        self.snapshots.append(seconds)
        return self._buffer[-round((seconds or 0) * 16000) :].copy()


class MsClock:
    """Integer-millisecond clock for the controller schedule."""

    def __init__(self, ms: int = 1_000_000) -> None:
        self.ms = ms

    def now_ms(self) -> int:
        return self.ms

    def advance(self, ms: int) -> None:
        self.ms += ms


class FakeAsr:
    def __init__(self, order: list[str], clock: MsClock) -> None:
        self.order = order
        self.clock = clock
        self.finals: list[tuple[Dictation, np.ndarray]] = []
        self.previews: list[tuple[int, int]] = []  # (dictation id, clock ms)
        self.reloads: list[tuple[str, str, bool]] = []
        self.dropped: list[int] = []
        self.started: list[tuple[str | None, str]] = []
        self.stopped = False

    def start(self, model_id: str | None = None, device: str = "auto") -> None:
        self.started.append((model_id, device))

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.stopped = True

    def reload(self, model_id: str, device: str = "auto", *, force: bool = False) -> bool:
        self.reloads.append((model_id, device, force))
        return True

    def submit_final(self, dictation: Dictation, audio: np.ndarray) -> None:
        self.order.append("asr.submit_final")
        self.finals.append((dictation, audio))

    def submit_preview(self, dictation: Dictation, audio: np.ndarray) -> bool:
        self.previews.append((dictation.id, self.clock.now_ms()))
        return True

    def drop_previews(self, dictation_id: int) -> None:
        self.dropped.append(dictation_id)


class RecordingInjector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.result = INSERTED
        self.trigger_vk = 0

    def inject(self, text: str, mode: str, settings: Any = None) -> InjectResult:
        del settings
        self.calls.append((text, str(mode)))
        return self.result


class FakeDelivery:
    def __init__(self, injector: RecordingInjector) -> None:
        self.injector = injector
        self.submitted: list[tuple[Dictation, AsrOutcome]] = []
        self.cancelled: list[int] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout_s: float = 5.0) -> None:
        del timeout_s
        self.stopped = True

    def submit(self, d: Dictation, outcome: AsrOutcome) -> None:
        self.submitted.append((d, outcome))

    def cancel(self, dictation_id: int) -> None:
        self.cancelled.append(dictation_id)

    def inject_text(
        self, text: str, settings: Any, on_result: Callable[[InjectResult], None] | None = None
    ) -> None:
        result = self.injector.inject(text, settings.injection.mode, settings.injection)
        if on_result is not None:
            on_result(result)


class FakeHook:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.start_error: OSError | None = None

    def start(self, timeout_s: float = 5.0) -> None:
        del timeout_s
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.calls.append("stop")

    def abort(self) -> None:
        self.calls.append("abort")

    def reset(self) -> None:
        self.calls.append("reset")

    def set_enabled(self, enabled: bool) -> None:
        self.calls.append(("set_enabled", enabled))

    def update_settings(self, trigger: Any, repaste_hotkey: str) -> None:
        self.calls.append(("update_settings", repaste_hotkey))

    def release_stuck_ctrl(self) -> bool:
        self.calls.append("release_stuck_ctrl")
        return False


class FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.state = "hidden"
        self.countdown = ""
        self.texts: list[str] = []  # every outcome or message text, in order
        self.previews: list[tuple[int, int, str]] = []

    @property
    def hwnd(self) -> int:
        return 0

    def show_recording(self, dictation_id: int, *, handsfree: bool = False) -> None:
        self.calls.append(("show_recording", dictation_id, handsfree))
        self.state = "handsfree" if handsfree else "recording"
        self.countdown = ""

    def set_level(self, level: float) -> None:
        self.calls.append(("set_level", level))

    def set_remaining(self, remaining_s: float | None) -> None:
        self.calls.append(("set_remaining", remaining_s))
        self.countdown = countdown_text(remaining_s)

    def show_preview(self, dictation_id: int, seq: int, text: str) -> bool:
        self.previews.append((dictation_id, seq, text))
        return True

    def show_transcribing(self) -> None:
        self.calls.append(("show_transcribing",))
        self.state = "transcribing"

    def show_inserting(self) -> None:
        self.calls.append(("show_inserting",))
        self.state = "inserting"

    def show_outcome(self, status: str | Msg) -> None:
        assert isinstance(status, Msg)
        self.calls.append(("show_outcome", status))
        self.texts.append(status.text)
        self.state = "outcome"

    def show_message(self, text: str) -> None:
        self.calls.append(("show_message", text))
        self.texts.append(text)
        self.state = "outcome"

    def hide_now(self) -> None:
        self.calls.append(("hide_now",))
        self.state = "hidden"


class FakeSounds:
    def __init__(self) -> None:
        self.cues: list[str] = []

    def play_start(self) -> None:
        self.cues.append("start")

    def play_end(self) -> None:
        self.cues.append("end")

    def configure(self, settings: Any) -> None:
        del settings


class FakePrewarmer:
    def __init__(self, calls: list[str], endpoint: str) -> None:
        self.calls = calls
        self.endpoint = endpoint

    def prewarm(self) -> None:
        self.calls.append(self.endpoint)


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


class Harness:
    def __init__(
        self,
        home: Path,
        *,
        asr: Any = None,
        sounds: Any = None,
        bridge: Bridge | None = None,
    ) -> None:
        self.home = home
        self.clock = MsClock()
        self.order: list[str] = []
        self.config = ConfigManager(home, clock=FakeClock(time.time() + 3600))
        self.history = HistoryStore(home)
        self.recorder = FakeRecorder(self.order)
        self.asr = asr if asr is not None else FakeAsr(self.order, self.clock)
        self.injector = RecordingInjector()
        self.delivery = FakeDelivery(self.injector)
        self.hook = FakeHook()
        self.overlay = FakeOverlay()
        self.sounds = sounds if sounds is not None else FakeSounds()
        self.prewarms: list[str] = []
        self.notices: list[str] = []
        self.opened: list[Path] = []
        self.tray = Tray(
            self.config,
            history=self.history.entries,
            devices=lambda: [],
            autostart_enabled=lambda: False,
            toggle_guard=lambda key, value: self.app.guard_toggle(key, value),
        )
        self.tray.notify = self._notify  # type: ignore[method-assign]
        self.app = App(
            config=self.config,
            history=self.history,
            recorder=self.recorder,
            asr=self.asr,
            delivery=self.delivery,
            hook=self.hook,
            overlay=self.overlay,
            tray=self.tray,
            sounds=self.sounds,
            bridge=bridge,
            clock=self.clock,
            rewriter_factory=lambda s: FakePrewarmer(self.prewarms, s.endpoint),
            open_path=self.opened.append,
            quit_app=lambda: None,
        )

    def _notify(self, text: str, *, error: bool = False) -> None:
        del error
        self.notices.append(text)

    # -- driving -------------------------------------------------------------------------

    def start(self, *, ready: bool = True, device: Device = CUDA) -> None:
        self.app.start()
        if ready:
            self.ready(device=device)

    def ready(self, *, model: str = TURBO, device: Device = CUDA, vocab: bool = True) -> None:
        self.app.on_worker_event(EngineReady(model, device, vocab))

    def commit(self, mode: DictationMode = DictationMode.HOLD) -> Dictation | None:
        effect = Effect.COMMIT_HOLD if mode is DictationMode.HOLD else Effect.START_HANDSFREE
        before = set(self.app._active)
        self.app.handle_effects((Effect.OPEN_MIC, effect))
        new = set(self.app._active) - before
        return self.app._active[new.pop()].dictation if new else None

    def committed(self, mode: DictationMode = DictationMode.HOLD) -> Dictation:
        d = self.commit(mode)
        assert d is not None
        return d

    def finish(self) -> None:
        self.app.handle_effect(Effect.FINISH)

    def asr_text(self, d: Dictation, text: str = "olá", **kwargs: Any) -> None:
        self.app.on_worker_event(AsrOutcome(d.id, "text", text, speech_s=1.0, **kwargs))

    def deliver(
        self, d: Dictation, status: DeliveryStatus = DeliveryStatus.INSERTED, **kwargs: Any
    ) -> None:
        self.app.on_delivery_event(DeliveryOutcome(d.id, status, text="olá", **kwargs))

    def dictate(self, text: str = "olá") -> Dictation:
        d = self.committed()
        self.advance(1000)
        self.finish()
        self.asr_text(d, text)
        self.deliver(d)
        return d

    def advance(self, ms: int, *, step: int = 20) -> None:
        for _ in range(ms // step):
            self.clock.advance(step)
            self.app.tick()


@pytest.fixture
def h(qtbot: QtBot, home: Path) -> Iterator[Harness]:
    del qtbot  # only needed for the QApplication
    harness = Harness(home)
    yield harness
    harness.app.shutdown()


def started(h: Harness, **kwargs: Any) -> Harness:
    h.start(**kwargs)
    return h


# --------------------------------------------------------------------------------------
# Commit, finish, cancel
# --------------------------------------------------------------------------------------


def test_ut162_commit_hold_creates_a_dictation_with_a_snapshot(h: Harness) -> None:
    started(h)
    d = h.committed()
    assert d.mode is DictationMode.HOLD
    assert d.state is DictationState.RECORDING
    assert d.settings is h.config.settings  # the immutable snapshot taken at commit
    assert h.sounds.cues == ["start"]
    assert ("show_recording", d.id, False) in h.overlay.calls
    assert h.overlay.state == "recording"
    h.advance(700)  # resolve_preview("auto", cuda) is True: the preview timer runs
    assert [did for did, _ in h.asr.previews] == [d.id]


def test_ut163_commit_while_loading_notifies_and_aborts(h: Harness) -> None:
    started(h, ready=False)
    assert h.app.state is AppState.LOADING
    d = h.commit()
    assert d is None
    assert h.overlay.texts == [Msg.MODEL_LOADING.text]
    assert "abort" in h.hook.calls
    assert "recorder.discard" in h.order
    assert h.app._active == {}


def test_ut164_commit_while_downloading_notifies(h: Harness) -> None:
    started(h, ready=False)
    h.app.on_worker_event(DownloadProgress(TURBO, 10, 100))
    assert h.app.state is AppState.DOWNLOADING
    assert h.tray.state is TrayState.DOWNLOADING
    assert h.commit() is None
    assert h.overlay.texts == [Msg.MODEL_DOWNLOADING.text]
    assert h.app._active == {}


def test_ut165_finish_stops_the_recorder_then_submits_the_final(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.recorder.feed(1.0)
    h.order.clear()
    h.finish()
    assert h.order == ["recorder.stop", "asr.submit_final"]
    assert h.asr.finals[0][0] is d
    assert h.asr.finals[0][1].size == 2 * 16000
    assert d.state is DictationState.TRANSCRIBING


def test_ut166_cancel_discards_without_a_job_or_history(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.order.clear()
    h.app.handle_effect(Effect.CANCEL)
    assert h.order == ["recorder.discard"]
    assert h.overlay.calls[-1] == ("show_outcome", Msg.CANCELLED)
    assert h.sounds.cues == ["start", "end"]
    assert d.state is DictationState.CANCELLED
    assert h.asr.finals == []
    assert h.delivery.submitted == []
    assert h.history.entries() == []
    assert h.app._active == {}


def test_ut167_limit_timer_finishes_and_shows_the_countdown(h: Harness, logs: list[str]) -> None:
    h.config.set_value("recording.max_seconds", 3)
    started(h)
    d = h.committed()
    h.advance(500)  # 2.5 s remaining
    assert h.overlay.countdown == "0:02"
    h.advance(2480)
    assert h.asr.finals == []
    h.advance(20)  # 3000 ms after the commit
    assert "abort" in h.hook.calls
    assert [f[0] for f in h.asr.finals] == [d]
    assert d.state is DictationState.TRANSCRIBING
    assert any(line.startswith("dictation_stopped") and "reason=limit" in line for line in logs)


def test_ut168_pause_cancels_the_recording_and_keeps_the_engine(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.app.pause()
    assert d.state is DictationState.CANCELLED
    assert h.asr.finals == []
    assert ("set_enabled", False) in h.hook.calls
    assert h.app.state is AppState.PAUSED
    assert h.tray.state is TrayState.PAUSED
    assert not h.asr.stopped


def test_ut169_resume_enables_the_trigger(h: Harness) -> None:
    started(h)
    h.app.pause()
    h.app.resume()
    assert h.hook.calls[-1] == ("set_enabled", True)
    assert h.app.state is AppState.READY
    assert h.tray.state is TrayState.READY


def test_ut170_startup_never_starts_paused(qtbot: QtBot, home: Path, logs: list[str]) -> None:
    del qtbot
    first = Harness(home)
    first.start()
    first.app.pause()
    first.app.shutdown()
    logs.clear()

    second = Harness(home)  # the previous session ended paused
    second.app.start()
    assert second.app.state is AppState.LOADING
    second.ready()
    assert second.app.state is AppState.READY
    states = [line for line in logs if line.startswith("app_state")]
    assert states and all("to=PAUSED" not in line for line in states)
    second.app.shutdown()


def test_ut171_reload_and_import_wait_for_the_active_dictation(h: Harness, home: Path) -> None:
    started(h)
    d1 = h.committed()
    assert set_value(h.config.path, "recording.max_seconds", 10).saved
    h.app.reload_config()
    assert h.config.settings.recording.max_seconds == 300  # deferred
    h.advance(1000)
    h.finish()
    h.asr_text(d1)
    assert h.config.settings.recording.max_seconds == 300
    h.deliver(d1)
    assert d1.state is DictationState.INSERTED
    assert h.config.settings.recording.max_seconds == 10  # applied after INSERTED

    other = home.parent / "other.toml"
    other.write_text("[recording]\nmax_seconds = 20\n", encoding="utf-8")
    d2 = h.committed()
    h.app.import_config(other)
    assert h.config.settings.recording.max_seconds == 10
    h.finish()
    h.asr_text(d2)
    h.deliver(d2, DeliveryStatus.FAILED_INSERT)
    assert d2.state is DictationState.FAILED_INSERT
    assert h.config.settings.recording.max_seconds == 20


def test_ut172_tray_toggle_leaves_the_running_snapshot(h: Harness) -> None:
    started(h)
    d1 = h.committed()
    h.tray.cleanup_action.trigger()
    assert h.config.settings.cleanup.enabled is False
    assert d1.settings.cleanup.enabled is True
    h.finish()
    h.asr_text(d1)
    h.deliver(d1)
    d2 = h.committed()
    assert d2.settings.cleanup.enabled is False


@pytest.mark.parametrize("reason", ["lock", "suspend"])
def test_ut173_lock_or_suspend_interrupts_the_recording(h: Harness, reason: str) -> None:
    started(h)
    d = h.committed()
    h.order.clear()
    h.app.on_hook_event(SessionInterrupted(reason))  # type: ignore[arg-type]
    assert d.state is DictationState.INTERRUPTED
    assert "recorder.discard" in h.order
    assert h.overlay.calls[-1] == ("show_outcome", Msg.INTERRUPTED)
    assert h.hook.calls[-1] == "reset"
    assert h.asr.finals == []
    assert h.app._active == {}
    d2 = h.committed()  # after resume a new dictation works
    assert d2.state is DictationState.RECORDING


def test_ut174_mic_loss_finishes_with_the_captured_audio(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.recorder.feed(0.5)
    h.app.on_device_lost()
    assert d.state is DictationState.TRANSCRIBING
    assert h.asr.finals[0][1].size == round(1.5 * 16000)
    h.asr_text(d)
    h.deliver(d)
    assert h.overlay.calls[-1] == ("show_outcome", Msg.MIC_LOST)


@pytest.mark.parametrize(
    ("kind", "msg"),
    [("blocked", Msg.MIC_BLOCKED), ("no_device", Msg.MIC_NO_DEVICE), ("busy", Msg.MIC_BUSY)],
)
def test_ut175_mic_errors_show_their_message(h: Harness, kind: str, msg: Msg) -> None:
    started(h)
    h.recorder.open_error = MicError(kind)  # type: ignore[arg-type]
    assert h.commit() is None
    assert h.overlay.texts == [msg.text]
    assert "abort" in h.hook.calls
    assert h.app._active == {}
    assert h.sounds.cues == []


# --------------------------------------------------------------------------------------
# Re-paste, transitions, notices
# --------------------------------------------------------------------------------------


def test_ut176_repaste_hotkey(h: Harness) -> None:
    started(h)
    h.app.on_hook_event(RepasteRequested())
    assert h.overlay.texts == [Msg.NOTHING_TO_REPASTE.text]
    assert h.injector.calls == []

    entry = h.history.add_pending("x")
    h.history.mark(entry, HistoryStatus.INSERTED)
    h.app.on_hook_event(RepasteRequested())
    assert h.injector.calls == [("x", "paste")]

    h.committed()
    h.app.on_hook_event(RepasteRequested())  # ignored during a recording
    assert h.injector.calls == [("x", "paste")]
    h.app.handle_effect(Effect.CANCEL)

    h.app.pause()
    h.config.set_value("injection.mode", "type")
    h.app.on_hook_event(RepasteRequested())  # still works while paused, in the current mode
    assert h.injector.calls == [("x", "paste"), ("x", "type")]


def test_ut176_failed_repaste_is_reported(h: Harness) -> None:
    started(h)
    h.history.add_pending("x")
    h.injector.result = InjectResult(InjectStatus.FAILED_ELEVATED, True, False, False)
    h.app.repaste()
    assert h.overlay.texts[-1] == Msg.INSERT_FAILED_ELEVATED.text


def test_ut177_illegal_transition_raises(h: Harness) -> None:
    started(h)
    d = h.dictate()
    assert d.state is DictationState.INSERTED
    with pytest.raises(AssertionError):
        h.app._transition(d, DictationState.RECORDING)


@pytest.mark.parametrize(
    ("peak", "msg"),
    [(-70.0, Msg.NOTHING_DETECTED_MUTED_HINT), (-20.0, Msg.NOTHING_DETECTED)],
)
def test_ut178_no_speech_muted_hint(h: Harness, peak: float, msg: Msg) -> None:
    started(h)
    d = h.committed()
    h.app.on_level(Level(0.0001, peak))
    h.finish()
    h.app.on_worker_event(AsrOutcome(d.id, "no_speech", speech_s=0.1))
    assert d.state is DictationState.DISCARDED
    assert h.overlay.calls[-1] == ("show_outcome", msg)
    assert h.history.entries() == []


def test_ut179_onboarding_is_shown_once(qtbot: QtBot, home: Path) -> None:
    del qtbot
    first = Harness(home)
    assert not Paths(home).state.exists()
    first.start()
    assert first.notices.count(Msg.ONBOARDING.text) == 1
    first.ready()  # a later READY (model switch) does not repeat it
    assert first.notices.count(Msg.ONBOARDING.text) == 1
    assert json.loads(Paths(home).state.read_text(encoding="utf-8")) == {"onboarding_shown": True}
    first.app.shutdown()

    second = Harness(home)
    second.start()
    assert Msg.ONBOARDING.text not in second.notices
    second.app.shutdown()


def test_ut180_switch_to_parakeet_reloads_and_warns_about_vocabulary(h: Harness) -> None:
    started(h)
    assert set_value(h.config.path, "asr.vocabulary", ["Kubernetes"]).saved
    assert set_value(h.config.path, "asr.model", "parakeet-v3").saved
    h.app.reload_config()
    assert h.asr.reloads == [("parakeet-v3", "auto", False)]
    h.ready(model="parakeet-v3", device=Device("cpu", "int8", "parakeet_cpu_only"), vocab=False)
    assert Msg.VOCAB_IGNORED.format(model="parakeet-v3") in h.notices
    assert h.tray.status_text == Msg.TRAY_STATUS.format(model="parakeet-v3", device="CPU")


def test_ut181_no_previews_when_the_preview_resolves_off(h: Harness) -> None:
    started(h, device=CPU)  # preview = "auto" on CPU resolves to False
    h.committed()
    h.advance(5000)
    assert h.asr.previews == []
    assert h.recorder.snapshots == []


def test_ut182_stale_preview_results_are_dropped(h: Harness) -> None:
    started(h)
    d = h.committed()
    for seq in (3, 2, 4):
        h.app.on_worker_event(PreviewResult(d.id, seq, f"texto {seq}"))
    h.app.on_worker_event(PreviewResult(d.id + 99, 5, "outro ditado"))
    assert [p[1] for p in h.overlay.previews] == [3, 4]


def test_ut183_commit_while_transcribing_keeps_order(h: Harness) -> None:
    started(h)
    d1 = h.committed()
    h.finish()
    assert d1.state is DictationState.TRANSCRIBING
    d2 = h.committed()
    assert d2.id > d1.id
    h.finish()
    h.asr_text(d1, "um")
    h.asr_text(d2, "dois")
    assert [(d.id, o.text) for d, o in h.delivery.submitted] == [(d1.id, "um"), (d2.id, "dois")]


def test_ut184_cpu_fallback_notice_is_shown_once(h: Harness) -> None:
    started(h)
    for _ in range(2):
        d = h.committed()
        h.finish()
        h.asr_text(d, used_cpu_fallback=True)
        h.deliver(d)
    assert h.notices.count(Msg.GPU_FALLBACK_CPU.text) == 1


def test_ut185_forced_device_fallback_notice(h: Harness) -> None:
    started(h, ready=False)
    h.ready(device=Device("cpu", "int8", "forced_unavailable"))
    assert Msg.DEVICE_FORCED_FALLBACK.format(model=TURBO) in h.notices


def test_ut186_ai_toggle_refuses_a_remote_endpoint(h: Harness) -> None:
    h.config.path.write_text('[llm]\nendpoint = "http://example.com:11434"\n', encoding="utf-8")
    h.config.reload()
    started(h)
    h.tray.rewrite_action.trigger()
    assert h.config.settings.llm.enabled is False
    assert Msg.LLM_ENDPOINT_NOT_LOCAL.text in h.notices
    assert not h.tray.rewrite_action.isChecked()
    assert h.prewarms == []


def test_ut187_ai_toggle_prewarms_once(h: Harness) -> None:
    started(h)
    h.tray.rewrite_action.trigger()
    assert h.config.settings.llm.enabled is True
    assert h.tray.rewrite_action.isChecked()
    assert h.prewarms == ["http://127.0.0.1:11434"]


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def test_ut199_second_instance_exits_zero_without_an_app(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    monkeypatch.setenv("TALKTYPE_HOME", str(home))
    notified: list[Path] = []
    monkeypatch.setattr(entry, "notify_already_running", lambda paths: notified.append(paths.home))
    closed: list[int] = []
    instance = SingleInstance(
        "x", create_mutex=lambda _name: (7, True), close_handle=lambda h: not closed.append(h)
    )

    def run(paths: Paths) -> int:
        raise AssertionError("App must not be constructed")

    assert entry.main(instance=instance, run=run) == 0
    assert notified == [home]
    assert closed == [7]


def test_ut199_already_running_notice_goes_to_the_log_under_the_seam(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    monkeypatch.setattr(seams, "current", lambda: seams.Seams(no_msgbox=True))
    boxes: list[str] = []
    monkeypatch.setattr(entry.win32, "MessageBoxW", lambda text, *a, **k: boxes.append(text))
    entry.notify_already_running(Paths(home))
    entry.close_logging()
    assert boxes == []
    assert "already_running" in Paths(home).log_file.read_text(encoding="utf-8")


def test_fatal_startup_error_exits_two(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("TALKTYPE_HOME", str(home))
    monkeypatch.setattr(seams, "current", lambda: seams.Seams(no_msgbox=True))
    instance = SingleInstance(
        "x", create_mutex=lambda _name: (7, False), close_handle=lambda _h: True
    )

    def run(paths: Paths) -> int:
        raise RuntimeError("boom")

    assert entry.main(instance=instance, run=run) == 2
    log = Paths(home).log_file.read_text(encoding="utf-8")
    assert "startup_failed" in log
    assert "boom" in log


# --------------------------------------------------------------------------------------
# Scale, scheduling and races
# --------------------------------------------------------------------------------------


def test_ut208_hundred_dictations_release_everything(h: Harness) -> None:
    started(h)
    for index in range(100):
        d = h.dictate(f"ditado {index}")
        assert d.state is DictationState.INSERTED
        assert h.app._active == {}
        assert h.recorder.buffer_bytes == 0
    assert len(h.delivery.submitted) == 100


def test_ut210_preview_uses_only_the_last_window(qtbot: QtBot) -> None:
    del qtbot
    engines = FakeEngineFactory()
    worker = AsrWorker(store=FakeStore(), engine_factory=engines, device_selector=lambda m, r: CUDA)
    worker.start(TURBO)
    try:
        deadline = time.monotonic() + 5
        while not worker.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        settings = config_module.Settings()
        d = Dictation(1, DictationMode.HANDSFREE, settings, 0)
        assert worker.submit_preview(d, np.zeros(5 * 60 * 16000, dtype=np.float32))
        deadline = time.monotonic() + 5
        while not engines.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        call = engines.calls[0]
        assert call.fast is True
        assert call.samples == 15 * 16000
    finally:
        worker.stop()


def test_ut210_app_schedules_previews_every_interval(h: Harness) -> None:
    started(h)
    d = h.committed(DictationMode.HANDSFREE)
    start = h.clock.now_ms()
    h.advance(7000)
    times = [t - start for did, t in h.asr.previews if did == d.id]
    assert times == [700 * n for n in range(1, 11)]
    assert set(h.recorder.snapshots) == {15.0}


def test_ut211_cancel_just_before_the_limit_wins(h: Harness) -> None:
    h.config.set_value("recording.max_seconds", 3)
    started(h)
    d = h.committed()
    h.advance(2980)
    h.clock.advance(19)  # 2999 ms
    h.app.handle_effect(Effect.CANCEL)
    h.advance(1000)
    assert d.state is DictationState.CANCELLED
    assert h.asr.finals == []


def test_ut212_cancel_while_transcribing_drops_the_outcome(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.finish()
    h.app.handle_effect(Effect.CANCEL)
    assert d.state is DictationState.CANCELLED
    h.asr_text(d)
    assert h.delivery.submitted == []
    assert h.app._active == {}


def test_ut214_finish_and_limit_in_the_same_tick_submit_once(h: Harness) -> None:
    h.config.set_value("recording.max_seconds", 3)
    started(h)
    d = h.committed()
    h.clock.advance(3000)
    h.finish()
    h.app.tick()
    h.app.handle_effect(Effect.FINISH)
    assert [f[0] for f in h.asr.finals] == [d]


def test_ut215_overlay_shows_transcribing_until_the_outcome(h: Harness) -> None:
    started(h)
    d = h.committed()
    h.finish()
    assert h.overlay.state == "transcribing"
    h.app.on_worker_event(PreviewResult(d.id, 1, "tarde demais"))
    assert h.overlay.state == "transcribing"
    h.asr_text(d)
    assert h.overlay.state == "transcribing"
    h.deliver(d)
    assert h.overlay.state == "outcome"
    assert h.overlay.calls[-1] == ("show_outcome", Msg.INSERTED)


def test_ut216_cue_order(qtbot: QtBot, home: Path) -> None:
    del qtbot
    log: list[str] = []
    names = {wav_image(builtin_path(cue)): cue.value for cue in Cue}

    def play_sound(sound: bytes | None, flags: int) -> bool:
        del flags
        if sound is not None:
            log.append(f"play:{names[sound]}")
        return True

    sounds = Sounds(config_module.RecordingSettings(), play_sound=play_sound)
    h = Harness(home, sounds=sounds)
    h.start()
    d1 = h.committed()
    assert log == ["play:start"]
    h.finish()
    assert log[-1] == "play:end"
    log.clear()
    d2 = h.committed()  # the end cue may still be playing: PlaySound stops it
    assert log == ["play:start"]
    h.app.handle_effect(Effect.CANCEL)
    assert log[-1] == "play:end"
    log.clear()
    h.app.on_worker_event(AsrOutcome(d1.id, "no_speech", speech_s=0.0))
    assert log == []  # the no-speech path already played its end cue on release
    d3 = h.committed(DictationMode.HANDSFREE)
    h.finish()
    h.app.on_worker_event(AsrOutcome(d3.id, "no_speech", speech_s=0.0))
    assert log == ["play:start", "play:end"]
    del d2
    h.app.shutdown()


# --------------------------------------------------------------------------------------
# Settings, models and files
# --------------------------------------------------------------------------------------


def test_ut226_language_change_needs_no_engine_reload(h: Harness) -> None:
    started(h)
    assert set_value(h.config.path, "asr.language", "en").saved
    h.app.reload_config()
    assert h.asr.reloads == []
    h.committed()
    h.finish()
    assert h.asr.finals[0][0].settings.asr.language == "en"


class Hub:
    """`snapshot_download` stand-in: turbo is cached; other models fail as scripted."""

    def __init__(self, root: Path, errors: dict[str, list[BaseException]]) -> None:
        self.root = root
        self.errors = errors
        self.downloads: list[str] = []

    def __call__(self, repo_id: str, *, local_files_only: bool, **kwargs: Any) -> str:
        from huggingface_hub.errors import LocalEntryNotFoundError

        name = repo_id.rsplit("/", 1)[-1]
        if local_files_only:
            if "turbo" in name and not self.errors.get("turbo"):
                return str(self.root / name)
            raise LocalEntryNotFoundError("not cached")
        self.downloads.append(name)
        pending = self.errors.get("turbo" if "turbo" in name else name, [])
        if pending:
            raise pending.pop(0)
        return str(self.root / name)


def real_worker(h_bridge: Bridge, store: Any, **kwargs: Any) -> AsrWorker:
    return AsrWorker(
        store=store,
        engine_factory=kwargs.pop("factory", FakeEngineFactory()),
        device_selector=kwargs.pop("selector", lambda m, r: CUDA),
        on_event=h_bridge.worker_event.emit,
        **kwargs,
    )


def test_ut230_unwritable_cache_keeps_the_previous_engine(
    qtbot: QtBot, home: Path, tmp_path: Path
) -> None:
    hub = Hub(tmp_path, {"faster-whisper-small": [PermissionError(13, "denied")]})
    bridge = Bridge()
    worker = real_worker(bridge, ModelStore(snapshot_download=hub, cache_dir=tmp_path / "hub"))
    h = Harness(home, asr=worker, bridge=bridge)
    h.app.start()
    try:
        qtbot.waitUntil(lambda: h.app.state is AppState.READY, timeout=5000)
        assert set_value(h.config.path, "asr.model", "small").saved
        h.app.reload_config()
        text = Msg.MODEL_CACHE_NOT_WRITABLE.format(path=tmp_path / "hub")
        qtbot.waitUntil(lambda: text in h.notices, timeout=5000)
        assert h.app.state is AppState.READY
        assert worker.engine is not None and worker.engine.model_id == TURBO
    finally:
        h.app.shutdown()


def test_ut232_retry_download_ensures_again(qtbot: QtBot, home: Path, tmp_path: Path) -> None:
    hub = Hub(tmp_path, {"turbo": [ConnectionError("offline")]})
    bridge = Bridge()
    worker = real_worker(bridge, ModelStore(snapshot_download=hub))
    h = Harness(home, asr=worker, bridge=bridge)
    h.app.start()
    try:
        qtbot.waitUntil(lambda: h.app.state is AppState.ERROR, timeout=5000)
        assert Msg.DOWNLOAD_OFFLINE.text in h.notices
        assert h.tray.retry_action.isVisible()
        h.tray.retry_action.trigger()
        qtbot.waitUntil(lambda: h.app.state is AppState.READY, timeout=5000)
        assert len(hub.downloads) == 1  # the retry found turbo after the first failure
        assert not h.tray.retry_action.isVisible()
    finally:
        h.app.shutdown()


def test_ut236_load_failure_sets_error_and_blocks_dictation(h: Harness) -> None:
    started(h, ready=False)
    reason = str(ModelLoadError("snapshot corrompido"))
    h.app.on_worker_event(EngineFailed(TURBO, Msg.APP_ERROR, Msg.APP_ERROR.format(reason=reason)))
    assert h.app.state is AppState.ERROR
    assert h.app.error_reason == reason
    assert h.tray.state is TrayState.ERROR
    assert h.commit() is None
    assert h.overlay.texts[-1] == Msg.APP_ERROR.format(reason=reason)


def test_ut239_pause_is_idempotent(h: Harness) -> None:
    started(h)
    h.app.pause()
    h.app.pause()
    assert h.app.state is AppState.PAUSED
    assert h.hook.calls.count(("set_enabled", False)) == 1


def test_ut240_pause_while_downloading_applies_after_the_download(h: Harness) -> None:
    started(h, ready=False)
    h.app.on_worker_event(DownloadProgress(TURBO, 10, 100))
    h.app.pause()
    assert h.app.state is AppState.DOWNLOADING
    assert h.asr.reloads == [] and not h.asr.stopped  # the download continues
    h.app.on_worker_event(DownloadProgress(TURBO, 100, 100))
    h.ready()
    assert h.app.state is AppState.PAUSED
    h.app.resume()
    assert h.app.state is AppState.READY


def test_ut242_import_with_too_little_vram_runs_on_cpu(
    qtbot: QtBot, home: Path, tmp_path: Path
) -> None:
    nvml = FakeNvml(free=8 * GIB)
    bridge = Bridge()
    engines = FakeEngineFactory()
    worker = real_worker(
        bridge,
        FakeStore(),
        factory=engines,
        selector=lambda model, requested: select_device(model, requested, nvml=nvml),
    )
    h = Harness(home, asr=worker, bridge=bridge)
    h.app.start()
    try:
        qtbot.waitUntil(lambda: h.app.state is AppState.READY, timeout=5000)
        nvml.free = 2 * GIB
        exported = tmp_path / "x.toml"
        exported.write_text('[asr]\nmodel = "large-v3"\n', encoding="utf-8")
        h.app.import_config(exported)
        assert h.config.settings.asr.model == "large-v3"
        text = Msg.DEVICE_FALLBACK_CPU.format(model="large-v3")
        qtbot.waitUntil(lambda: text in h.notices, timeout=5000)
        assert worker.engine is not None
        assert (worker.engine.model_id, worker.engine.device) == ("large-v3", "cpu")
        assert h.tray.status_text == Msg.TRAY_STATUS.format(model="large-v3", device="CPU")
    finally:
        h.app.shutdown()


def test_ut243_export_failure(h: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started(h)
    target = tmp_path / "somente-leitura" / "x.toml"

    def denied(path: Path, data: bytes) -> None:
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr(config_module, "atomic_write", denied)
    h.app.export_config(target)
    assert Msg.EXPORT_FAILED.format(path=target) in h.notices
    assert not target.exists()


def test_ut246_repaste_three_times(h: Harness) -> None:
    started(h)
    h.history.add_pending("de novo")
    for _ in range(3):
        h.app.on_hook_event(RepasteRequested())
    assert h.injector.calls == [("de novo", "paste")] * 3


# --------------------------------------------------------------------------------------
# Bridge
# --------------------------------------------------------------------------------------


def test_bridge_marshals_worker_threads_onto_the_gui_thread(qtbot: QtBot, home: Path) -> None:
    bridge = Bridge()
    h = Harness(home, bridge=bridge)
    h.start()
    seen: list[int] = []
    h.overlay.show_message = lambda text: seen.append(threading.get_ident())  # type: ignore[method-assign]
    failed = InjectResult(InjectStatus.FAILED_NO_FOCUS, True, False, False)
    thread = threading.Thread(
        target=lambda: bridge.delivery_event.emit(DirectInjectDone("repaste", failed))
    )
    thread.start()
    thread.join()
    qtbot.waitUntil(lambda: bool(seen), timeout=2000)
    assert seen == [threading.get_ident()]
    h.app.shutdown()
