"""Keyboard journeys against `python -m talktype` (E2E-001–E2E-010, E2E-020)."""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt

from talktype import win32
from talktype.config import TriggerSettings
from tests import desktop
from tests.e2e.conftest import (
    RCTRL,
    AppProcess,
    SubmitEdit,
    TextHarness,
    configure,
    contains_terms,
    double_tap,
    duration_ms,
    fixture_wav,
    history,
    hold_trigger,
    key_terms,
    normalize,
    wait_text,
)

pytestmark = pytest.mark.e2e

Launch = Callable[..., AppProcess]
PREOPEN_MS = 300  # the WAV starts playing at OPEN_MIC, 300 ms before the commit


def speak_ms(name: str) -> int:
    """How long to keep recording after the commit so the whole fixture is captured."""
    return duration_ms(name) - PREOPEN_MS + 400


# --------------------------------------------------------------------------------------
# E2E-001: hold-to-talk
# --------------------------------------------------------------------------------------


def test_e2e001_hold_to_talk_inserts_the_transcript(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    app = launch(home, fixture_wav("deploy_pr"))
    desktop.set_clipboard_text("sentinela")
    harness.open()

    committed_s = hold_trigger(app.log, speak_ms("deploy_pr"))
    hold_s = TriggerSettings().hold_threshold_ms / 1000
    assert hold_s <= committed_s <= hold_s + 0.3 + 0.2, committed_s  # 0.2 s of log polling slack

    text = wait_text(harness, lambda t: contains_terms(t, key_terms("deploy_pr")))
    assert contains_terms(text, key_terms("deploy_pr")), text
    finished = app.log.wait("dictation_finished", status="inserted")
    assert int(finished["total_ms"]) < 1500, finished
    desktop.pump(1000)
    assert desktop.clipboard_text() == "sentinela"
    entries = history(home)
    assert entries[0]["status"] == "inserted"
    assert normalize(entries[0]["text"]) == normalize(text)


# --------------------------------------------------------------------------------------
# E2E-002, E2E-003: normal keyboard use
# --------------------------------------------------------------------------------------


def test_e2e002_a_right_ctrl_tap_passes_through(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    app = launch(home, fixture_wav("curta_sim"))
    harness.open()
    desktop.tap(RCTRL, 80)
    desktop.pump(600)
    assert harness.keys.pressed(Qt.Key.Key_Control)
    assert harness.keys.released(Qt.Key.Key_Control)
    assert app.log.events("mic_opened") == []
    assert app.log.events("dictation_committed") == []


def test_e2e003_right_ctrl_a_selects_all(home: Path, launch: Launch, harness: TextHarness) -> None:
    app = launch(home, fixture_wav("curta_sim"))
    harness.setPlainText("abc")
    harness.open()
    desktop.down(RCTRL)
    desktop.pump(50)
    desktop.tap(ord("A"), 40)
    desktop.pump(50)
    desktop.up(RCTRL)  # about 150 ms in total
    desktop.pump(400)
    assert harness.textCursor().selectedText() == "abc"
    assert harness.text() == "abc"
    assert app.log.events("mic_opened") == []


# --------------------------------------------------------------------------------------
# E2E-004: hands-free
# --------------------------------------------------------------------------------------


def test_e2e004_double_tap_hands_free(home: Path, launch: Launch, harness: TextHarness) -> None:
    app = launch(home, fixture_wav("deploy_pr"))
    harness.open()
    double_tap(80, 150)
    app.log.wait("dictation_committed", timeout_s=3, mode="handsfree")
    desktop.pump(duration_ms("deploy_pr") + 400)
    desktop.tap(RCTRL, 80)
    text = wait_text(harness, lambda t: contains_terms(t, key_terms("deploy_pr")))
    assert contains_terms(text, key_terms("deploy_pr"))
    app.log.wait("dictation_finished", status="inserted", mode="handsfree")


# --------------------------------------------------------------------------------------
# E2E-005: cancel
# --------------------------------------------------------------------------------------


def test_e2e005_esc_cancels_without_reaching_the_app(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    app = launch(home, fixture_wav("deploy_pr"))
    harness.open()
    desktop.down(RCTRL)
    try:
        app.log.wait("dictation_committed", timeout_s=4)
        desktop.pump(100)
        desktop.tap(win32.VK_ESCAPE, 60)
        desktop.pump(100)
    finally:
        desktop.up(RCTRL)
    app.log.wait("dictation_finished", status="cancelled")
    desktop.pump(500)
    assert harness.text() == ""
    assert not harness.keys.pressed(Qt.Key.Key_Escape)
    assert history(home) == []


# --------------------------------------------------------------------------------------
# E2E-006: limit
# --------------------------------------------------------------------------------------


def test_e2e006_recording_limit_auto_finishes(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    configure(home, recording__max_seconds=3)
    app = launch(home, fixture_wav("deploy_pr"))
    harness.open()
    double_tap(80, 150)
    app.log.wait("dictation_committed", timeout_s=3, mode="handsfree")
    started = time.monotonic()
    app.log.wait("dictation_stopped", timeout_s=6, reason="limit")
    elapsed = time.monotonic() - started
    assert 2.5 <= elapsed <= 4.0, elapsed
    text = wait_text(harness, lambda t: "então" in normalize(t))
    assert "então" in normalize(text)
    app.log.wait("dictation_finished", status="inserted")


# --------------------------------------------------------------------------------------
# E2E-007: no speech
# --------------------------------------------------------------------------------------


def test_e2e007_silence_is_discarded(home: Path, launch: Launch, harness: TextHarness) -> None:
    app = launch(home, fixture_wav("silencio_3s"))
    harness.open()
    hold_trigger(app.log, 3000)
    app.log.wait("dictation_finished", status="no_speech")
    desktop.pump(500)
    assert harness.text() == ""
    assert history(home) == []


# --------------------------------------------------------------------------------------
# E2E-008: type mode
# --------------------------------------------------------------------------------------


def test_e2e008_type_mode_never_touches_the_clipboard(
    home: Path, launch: Launch, harness: TextHarness
) -> None:
    configure(home, injection__mode="type")
    app = launch(home, fixture_wav("curta_sim"))
    desktop.set_clipboard_text("sentinela")
    sequence = desktop.clipboard_sequence()
    harness.open()
    hold_trigger(app.log, speak_ms("curta_sim"))
    text = wait_text(harness, lambda t: "sim" in normalize(t))
    assert "sim" in normalize(text)
    app.log.wait("dictation_finished", status="inserted")
    assert desktop.clipboard_sequence() == sequence
    assert desktop.clipboard_text() == "sentinela"


# --------------------------------------------------------------------------------------
# E2E-009: failure path and recovery
# --------------------------------------------------------------------------------------


def test_e2e009_no_focus_leaves_the_text_on_the_clipboard(
    home: Path, launch: Launch, qtbot: Any
) -> None:
    app = launch(home, fixture_wav("curta_sim"))
    first = TextHarness()
    qtbot.addWidget(first)
    first.open()
    desktop.down(RCTRL)
    try:
        app.log.wait("dictation_committed", timeout_s=4)
        desktop.pump(speak_ms("curta_sim"))
        first.close()  # the desktop gets the focus before the release
        desktop.focus_desktop()
        desktop.pump(200)
    finally:
        desktop.up(RCTRL)
    app.log.wait("dictation_finished", status="failed_insert")
    entries = history(home)
    assert entries[0]["status"] == "failed_insert"
    transcript = entries[0]["text"]
    assert "sim" in normalize(transcript)
    assert desktop.clipboard_text() == transcript

    second = TextHarness("talktype e2e recovery")
    qtbot.addWidget(second)
    second.open()
    desktop.chord(win32.VK_LCONTROL, ord("V"))
    text = wait_text(second, lambda t: bool(t))
    desktop.pump(500)
    assert second.text() == transcript  # inserted exactly once
    del text


# --------------------------------------------------------------------------------------
# E2E-010: line breaks never submit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["paste", "type"])
def test_e2e010_line_breaks_never_submit(home: Path, launch: Launch, qtbot: Any, mode: str) -> None:
    configure(home, injection__mode=mode)
    app = launch(home, fixture_wav("nova_linha"))
    box = SubmitEdit()
    qtbot.addWidget(box)
    box.open()
    hold_trigger(app.log, speak_ms("nova_linha"))
    text = wait_text(box, lambda t: "segundo" in normalize(t))
    app.log.wait("dictation_finished", status="inserted")
    desktop.pump(300)
    assert box.submits == 0
    first, _, rest = box.text().partition("\n")
    assert "primeiro item" in normalize(first), text
    assert "segundo item" in normalize(rest), text
    assert "nova linha" not in normalize(box.text())


# --------------------------------------------------------------------------------------
# E2E-020: privacy
# --------------------------------------------------------------------------------------


class TcpSampler:
    """Samples `Get-NetTCPConnection -OwningProcess <pid>` about every 200 ms."""

    def __init__(self, pid: int, out: Path) -> None:
        script = (
            f"while ($true) {{ Get-NetTCPConnection -OwningProcess {pid} "
            "-ErrorAction SilentlyContinue | ForEach-Object { "
            "'{0}|{1}|{2}' -f $_.LocalAddress, $_.RemoteAddress, $_.State }; "
            "'tick'; Start-Sleep -Milliseconds 200 }"
        )
        self.out = out
        self._file = out.open("wb")
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=self._file,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> list[str]:
        self.proc.kill()
        self.proc.wait(10)
        self._file.close()
        lines = self.out.read_text(encoding="utf-8", errors="replace").splitlines()
        assert lines.count("tick") >= 3, "the sampler did not run"
        return [line for line in lines if line and line != "tick"]


@pytest.fixture
def sampler(tmp_path: Path) -> Iterator[Callable[[int], TcpSampler]]:
    made: list[TcpSampler] = []

    def make(pid: int) -> TcpSampler:
        made.append(TcpSampler(pid, tmp_path / f"tcp-{pid}-{len(made)}.txt"))
        return made[-1]

    yield make
    for item in made:
        with contextlib.suppress(Exception):
            item.proc.kill()


def audio_files(*roots: Path) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found += [p for p in root.rglob("*") if p.suffix.lower() in (".wav", ".pcm", ".raw")]
    return found


def test_e2e020_nothing_leaves_the_machine(
    home: Path,
    tmp_path: Path,
    launch: Launch,
    harness: TextHarness,
    sampler: Callable[[int], TcpSampler],
) -> None:
    app = launch(home, fixture_wav("curta_sim"))
    watch = sampler(app.pid)
    harness.open()
    hold_trigger(app.log, speak_ms("curta_sim"))
    wait_text(harness, lambda t: "sim" in normalize(t))
    app.log.wait("dictation_finished", status="inserted")
    desktop.pump(600)
    assert watch.stop() == []  # llm disabled: no TCP connection at all
    app.stop()

    home_llm = tmp_path / "home-llm"
    configure(home_llm, llm__enabled=True)
    app = launch(home_llm, fixture_wav("curta_sim"))
    watch = sampler(app.pid)
    harness.clear()
    harness.open()
    hold_trigger(app.log, speak_ms("curta_sim"))
    app.log.wait("dictation_finished", timeout_s=30)
    desktop.pump(600)
    remotes = {line.split("|")[1] for line in watch.stop()}
    assert remotes <= {"127.0.0.1", "0.0.0.0", "::", "::1"}, remotes

    # Killed mid-recording: no audio file anywhere.
    desktop.down(RCTRL)
    try:
        app.log.wait("dictation_committed", timeout_s=4)
        desktop.pump(500)
        app.proc.kill()
        app.proc.wait(10)
    finally:
        desktop.up(RCTRL)
    temp = Path(tempfile_dir())
    roots = [home, home_llm, *temp.glob("talktype*")]
    assert audio_files(*[r for r in roots if r.is_dir()]) == []


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()
