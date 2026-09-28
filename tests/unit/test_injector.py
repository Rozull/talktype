"""Injector, clipboard and foreground against `FakeWin32` (UT-144 … UT-161, UT-221, UT-223)."""

from __future__ import annotations

import math
import tomllib

import pytest

from talktype import win32
from talktype.clipboard import MARKER_NAMES, MARKER_VALUE, ClipboardOwner, ClipboardSnapshot
from talktype.config import InjectionSettings, Settings
from talktype.foreground import probe_target
from talktype.injector import (
    InjectMode,
    Injector,
    InjectResult,
    InjectStatus,
    message_for,
)
from talktype.postprocess import postprocess
from talktype.strings import Msg
from tests.conftest import FIXTURES_AUDIO
from tests.fakes import FakeWin32, KeyEvent

pytestmark = pytest.mark.unit

OVERLAY = 0x3000
PASTE = InjectMode.PASTE
TYPE = InjectMode.TYPE


@pytest.fixture
def fake() -> FakeWin32:
    desktop = FakeWin32()
    desktop.classes[OVERLAY] = "Qt6110QWindowToolSaveBits"
    desktop.pids[OVERLAY] = desktop.own_pid
    return desktop


def injector(fake: FakeWin32) -> Injector:
    return Injector(overlay_hwnd=lambda: OVERLAY, api=fake, clock=fake.clock)


def key(vk: int, *, up: bool = False) -> KeyEvent:
    return KeyEvent(vk, 0, win32.KEYEVENTF_KEYUP if up else 0)


def unicode(unit: int) -> list[KeyEvent]:
    return [
        KeyEvent(0, unit, win32.KEYEVENTF_UNICODE),
        KeyEvent(0, unit, win32.KEYEVENTF_UNICODE | win32.KEYEVENTF_KEYUP),
    ]


SHIFT_ENTER = [
    key(win32.VK_SHIFT),
    key(win32.VK_RETURN),
    key(win32.VK_RETURN, up=True),
    key(win32.VK_SHIFT, up=True),
]
CTRL_V = [
    key(win32.VK_CONTROL),
    key(win32.VK_V),
    key(win32.VK_V, up=True),
    key(win32.VK_CONTROL, up=True),
]


def v_presses(fake: FakeWin32) -> int:
    return sum(1 for e in fake.events() if e.vk == win32.VK_V and not e.up)


# --- Pre-checks ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [PASTE, TYPE])
def test_ut144_no_focus_leaves_text_on_clipboard(fake: FakeWin32, mode: InjectMode) -> None:
    fake.seed("sentinela")
    fake.foreground = 0

    result = injector(fake).inject("olá", mode)

    assert result == InjectResult(
        InjectStatus.FAILED_NO_FOCUS, text_on_clipboard=True, clipboard_restored=False
    )
    assert fake.text() == "olá"
    assert fake.batches == []
    assert message_for(result) is Msg.INSERT_FAILED_NO_FOCUS


def test_ut145_elevated_target_is_refused_without_input(fake: FakeWin32) -> None:
    fake.elevated_pids.add(fake.pids[fake.TARGET_HWND])

    result = injector(fake).inject("olá", PASTE)

    assert result.status is InjectStatus.FAILED_ELEVATED
    assert result.text_on_clipboard is True
    assert fake.text() == "olá"
    assert fake.batches == []
    assert message_for(result) is Msg.INSERT_FAILED_ELEVATED


def test_ut146_access_denied_counts_as_elevated(fake: FakeWin32) -> None:
    fake.denied_pids.add(fake.pids[fake.TARGET_HWND])

    result = injector(fake).inject("olá", PASTE)

    assert result.status is InjectStatus.FAILED_ELEVATED
    assert fake.batches == []


def test_elevated_talktype_may_insert_into_elevated_target(fake: FakeWin32) -> None:
    fake.self_elevated = True
    fake.elevated_pids.add(fake.pids[fake.TARGET_HWND])

    assert injector(fake).inject("olá", PASTE).status is InjectStatus.INSERTED


def test_ut161_probe_classes(fake: FakeWin32) -> None:
    for cls in ("Progman", "WorkerW", "Shell_TrayWnd"):
        fake.classes[0x4000] = cls
        fake.foreground = 0x4000
        assert probe_target(OVERLAY, api=fake).status == "no_focus"

    fake.foreground = OVERLAY
    assert probe_target(OVERLAY, api=fake).status == "no_focus"

    fake.foreground = fake.TARGET_HWND
    probe = probe_target(OVERLAY, api=fake)
    assert (probe.status, probe.class_name, probe.hwnd) == ("ok", "Notepad", fake.TARGET_HWND)


def test_probe_of_a_vanished_window_is_no_focus(fake: FakeWin32) -> None:
    fake.foreground = 0x4444  # no class: the window is already gone

    assert probe_target(OVERLAY, api=fake).status == "no_focus"


# --- Paste mode ---------------------------------------------------------------------------


def test_ut147_confirmed_paste_then_restore(fake: FakeWin32) -> None:
    fake.seed("sentinela")
    started = fake.clock.now()
    rendered_at: list[float] = []
    fake.after_paste = lambda: rendered_at.append(fake.clock.now())
    inj = injector(fake)

    result = inj.inject("Olá, mundo", PASTE)

    assert result == InjectResult(
        InjectStatus.INSERTED, text_on_clipboard=False, clipboard_restored=True
    )
    assert fake.pasted == ["Olá, mundo"]
    assert rendered_at[0] - started == pytest.approx(0.050, abs=0.002)
    assert fake.clock.now() - rendered_at[0] == pytest.approx(0.150, abs=0.002)  # restore delay
    assert fake.text() == "sentinela"
    assert fake.owner == inj.clipboard.hwnd  # restored while our window owned it
    assert message_for(result) is None


def test_ut148_unconfirmed_paste_leaves_text(fake: FakeWin32) -> None:
    fake.seed("sentinela")
    fake.paste_render_ms = None  # the target ignores Ctrl+V
    started = fake.clock.now()

    result = injector(fake).inject("perdido?", PASTE)

    assert result.status is InjectStatus.FAILED_UNCONFIRMED
    assert result.text_on_clipboard is True
    assert result.clipboard_restored is False
    assert fake.clock.now() - started >= 0.500
    assert fake.data[win32.CF_UNICODETEXT] is not None  # rendered as normal content
    assert fake.text() == "perdido?"
    assert v_presses(fake) == 1
    assert message_for(result) is Msg.INSERT_FAILED_CLIPBOARD


def test_paste_confirm_timeout_follows_settings(fake: FakeWin32) -> None:
    fake.paste_render_ms = 700

    slow = injector(fake).inject("a", PASTE, InjectionSettings(paste_confirm_timeout_ms=1000))

    assert slow.status is InjectStatus.INSERTED


def test_ut149_user_copy_before_restore_is_kept(fake: FakeWin32) -> None:
    fake.seed("sentinela")
    fake.after_paste = lambda: fake.user_copy("novo")

    result = injector(fake).inject("ditado", PASTE)

    assert result.status is InjectStatus.INSERTED
    assert result.clipboard_restored is False
    assert result.text_on_clipboard is False
    assert fake.text() == "novo"


def test_ut150_locked_clipboard(fake: FakeWin32) -> None:
    fake.locked = True
    started = fake.clock.now()

    result = injector(fake).inject("x", PASTE)

    assert result == InjectResult(InjectStatus.FAILED_CLIPBOARD_LOCKED, text_on_clipboard=False)
    assert fake.clipboard_calls.count("OpenClipboard") == 5
    assert fake.clock.now() - started == pytest.approx(0.080)  # 4 retries, 20 ms apart
    assert fake.batches == []
    assert message_for(result) is Msg.INSERT_FAILED_OPEN_HISTORY


def test_ut151_unreadable_format_marks_restore_partial(fake: FakeWin32) -> None:
    fake.seed({win32.CF_UNICODETEXT: win32.unicode_text_bytes("a"), win32.CF_DIB: b"dib"})
    fake.bad_formats.add(win32.CF_DIB)

    result = injector(fake).inject("ditado", PASTE)

    assert result.status is InjectStatus.INSERTED
    assert result.clipboard_restored is True
    assert result.restore_partial is True
    assert message_for(result) is Msg.CLIPBOARD_PARTIAL_RESTORE
    assert fake.text() == "a"
    assert win32.CF_DIB not in fake.data


def test_ut151_snapshot_reports_partial(fake: FakeWin32) -> None:
    fake.seed({win32.CF_UNICODETEXT: win32.unicode_text_bytes("a"), win32.CF_DIB: b"dib"})
    fake.bad_formats.add(win32.CF_DIB)

    snapshot = ClipboardSnapshot.capture(api=fake, clock=fake.clock)

    assert snapshot.partial is True
    assert snapshot.get(win32.CF_UNICODETEXT) == win32.unicode_text_bytes("a")


def test_snapshot_skips_synthesized_handles_without_partial(fake: FakeWin32) -> None:
    fake.seed({win32.CF_DIB: b"dib", win32.CF_BITMAP: b"", win32.CF_ENHMETAFILE: b"emf"})

    snapshot = ClipboardSnapshot.capture(api=fake, clock=fake.clock)

    assert snapshot.partial is False
    assert [(f.format, f.kind) for f in snapshot.formats] == [
        (win32.CF_DIB, "hglobal"),
        (win32.CF_ENHMETAFILE, "emf"),
    ]


def test_ut152_empty_clipboard_is_restored_empty(fake: FakeWin32) -> None:
    result = injector(fake).inject("ditado", PASTE)

    assert result.clipboard_restored is True
    assert fake.data == {}
    last_empty = len(fake.clipboard_calls) - 1 - fake.clipboard_calls[::-1].index("EmptyClipboard")
    assert "SetClipboardData" not in fake.clipboard_calls[last_empty:]


@pytest.mark.parametrize(("text", "rendered"), [("a\nb", "a\r\nb"), ("a", "a")])
def test_ut153_line_breaks_are_crlf(fake: FakeWin32, text: str, rendered: str) -> None:
    injector(fake).inject(text, PASTE)

    assert fake.pasted == [rendered]


def _marker_sets(fake: FakeWin32) -> list[int]:
    ids = {fake.format_id(name) for name in MARKER_NAMES}
    return [fmt for fmt, value in fake.sets if fmt in ids and value == MARKER_VALUE]


def test_ut154_offer_and_restore_carry_exclusion_markers(fake: FakeWin32) -> None:
    fake.seed("sentinela")
    offered: list[int] = []
    fake.after_paste = lambda: offered.extend(_marker_sets(fake))

    injector(fake).inject("ditado", PASTE)

    assert sorted(offered) == sorted(fake.format_id(name) for name in MARKER_NAMES)
    assert len(_marker_sets(fake)) == 2 * len(MARKER_NAMES)  # the offer and the restore
    for name in MARKER_NAMES:
        assert fake.data[fake.format_id(name)] == (0).to_bytes(4, "little")


def test_failure_path_text_carries_markers(fake: FakeWin32) -> None:
    fake.paste_render_ms = None

    injector(fake).inject("x", PASTE)

    for name in MARKER_NAMES:
        assert fake.data[fake.format_id(name)] == MARKER_VALUE


def test_ut159_paste_keys_are_layout_independent_virtual_keys(fake: FakeWin32) -> None:
    injector(fake).inject("x", PASTE)

    assert fake.batches == [CTRL_V]
    assert all(not e.flags & win32.KEYEVENTF_SCANCODE for e in fake.batches[0])


def test_closing_the_owner_renders_owed_text(fake: FakeWin32) -> None:
    owner = ClipboardOwner(api=fake, clock=fake.clock)
    assert owner.offer("devido") is not None
    assert fake.data[win32.CF_UNICODETEXT] is None

    owner.close()

    assert fake.text() == "devido"


# --- Type mode ----------------------------------------------------------------------------


def test_ut155_type_sequence(fake: FakeWin32) -> None:
    result = injector(fake).inject("olá\n\n😀", TYPE)

    assert result == InjectResult(InjectStatus.INSERTED)
    expected = [
        *unicode(ord("o")),
        *unicode(ord("l")),
        *unicode(ord("á")),
        *SHIFT_ENTER,
        *SHIFT_ENTER,
        *unicode(0xD83D),
        *unicode(0xDE00),
    ]
    assert fake.events() == expected
    assert fake.typed == "olá\n\n😀"
    assert fake.submits == 0


def test_type_mode_crlf_is_one_line_break(fake: FakeWin32) -> None:
    injector(fake).inject("a\r\nb\rc", TYPE)

    assert fake.typed == "a\nb\nc"


def test_ut156_foreground_change_stops_typing(fake: FakeWin32) -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(40))

    def switch(batches: int) -> None:
        if batches == 1:
            fake.foreground = fake.OTHER_HWND

    fake.on_send = switch

    result = injector(fake).inject(text, TYPE)

    assert result.status is InjectStatus.PARTIAL
    assert result.text_on_clipboard is True
    assert len(fake.batches) == 1
    assert fake.typed == text[:16]
    assert fake.text() == text
    assert message_for(result) is Msg.INSERT_PARTIAL


def test_ut157_type_success_never_touches_the_clipboard(fake: FakeWin32) -> None:
    fake.seed("sentinela")
    fake.clipboard_calls.clear()

    result = injector(fake).inject("ação, coração — ñ 😀", TYPE)

    assert result.status is InjectStatus.INSERTED
    assert fake.clipboard_calls == []
    assert fake.text() == "sentinela"


def test_ut158_type_into_elevated_target_sends_nothing(fake: FakeWin32) -> None:
    fake.elevated_pids.add(fake.pids[fake.TARGET_HWND])

    result = injector(fake).inject("olá", TYPE)

    assert result.status is InjectStatus.FAILED_ELEVATED
    assert fake.batches == []
    assert fake.text() == "olá"


def test_type_send_input_failure(fake: FakeWin32) -> None:
    fake.send_error = OSError(0, "blocked")

    result = injector(fake).inject("olá", TYPE)

    assert result.status is InjectStatus.FAILED_UNCONFIRMED
    assert fake.text() == "olá"


def _fixture_texts() -> list[str]:
    data = tomllib.loads((FIXTURES_AUDIO / "expected.toml").read_text(encoding="utf-8"))
    texts = [table["text"] for table in data.values() if table["text"]]
    return [*texts, *(postprocess(t, Settings()) for t in texts), "p1\n\np2\r\np3\n"]


def _returns_are_shifted(events: list[KeyEvent]) -> bool:
    shift = False
    for event in events:
        if event.vk == win32.VK_SHIFT:
            shift = not event.up
        elif event.vk == win32.VK_RETURN and not shift:
            return False
    return True


@pytest.mark.parametrize("mode", [PASTE, TYPE])
def test_ut160_no_bare_return_in_any_sequence(mode: InjectMode) -> None:
    for text in _fixture_texts():
        fake = FakeWin32()
        injector(fake).inject(text, mode)

        assert _returns_are_shifted(fake.events()), text
        assert fake.submits == 0
        if mode is TYPE:
            assert fake.typed == text.replace("\r\n", "\n")


def test_ut221_long_type_is_fast(fake: FakeWin32) -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(300))
    started = fake.clock.now()

    result = injector(fake).inject(text, TYPE, InjectionSettings(type_chunk_chars=16))

    assert result.status is InjectStatus.INSERTED
    assert len(fake.batches) == math.ceil(300 / 16)
    assert fake.clock.now() - started <= 0.095
    assert fake.typed == text


# --- Trigger release ----------------------------------------------------------------------


def test_ut223_waits_for_trigger_release(fake: FakeWin32) -> None:
    started = fake.clock.now()
    fake.key_release_at[win32.VK_RCONTROL] = started + 0.120

    injector(fake).inject("x", TYPE)

    first = fake.send_times[0] - started
    assert 0.120 - 1e-9 <= first <= 0.130 + 1e-9


def test_ut223_proceeds_after_500_ms(fake: FakeWin32, logs: list[str]) -> None:
    started = fake.clock.now()
    fake.key_release_at[win32.VK_RCONTROL] = math.inf

    result = injector(fake).inject("x", PASTE)

    assert result.status is InjectStatus.INSERTED
    assert fake.send_times[0] - started == pytest.approx(0.500, abs=0.011)
    assert any(line.startswith("trigger_still_down") for line in logs)


def test_waits_for_the_repaste_hotkey_modifiers(fake: FakeWin32) -> None:
    """Ctrl+Alt+Shift+V fires on the V down: Ctrl+V must wait until Alt and Shift are up."""
    started = fake.clock.now()
    fake.key_release_at[win32.VK_LCONTROL] = started + 0.030
    fake.key_release_at[win32.VK_LMENU] = started + 0.060
    fake.key_release_at[win32.VK_LSHIFT] = started + 0.090

    result = injector(fake).inject("x", PASTE)

    assert result.status is InjectStatus.INSERTED
    assert 0.090 - 1e-9 <= fake.send_times[0] - started <= 0.100 + 1e-9


def test_trigger_vk_can_be_reassigned(fake: FakeWin32) -> None:
    started = fake.clock.now()
    fake.key_release_at[win32.VK_F13] = started + 0.050
    inj = injector(fake)
    inj.trigger_vk = win32.VK_F13

    inj.inject("x", TYPE)

    assert fake.send_times[0] - started == pytest.approx(0.050, abs=0.011)


# --- Messages -----------------------------------------------------------------------------


def test_message_for_every_status() -> None:
    def msg(status: InjectStatus, **kw: bool) -> Msg | None:
        return message_for(InjectResult(status, **kw))

    assert msg(InjectStatus.INSERTED) is None
    assert msg(InjectStatus.INSERTED, restore_partial=True) is Msg.CLIPBOARD_PARTIAL_RESTORE
    on = {"text_on_clipboard": True}
    assert msg(InjectStatus.FAILED_NO_FOCUS, **on) is Msg.INSERT_FAILED_NO_FOCUS
    assert msg(InjectStatus.FAILED_ELEVATED, **on) is Msg.INSERT_FAILED_ELEVATED
    assert msg(InjectStatus.FAILED_UNCONFIRMED, **on) is Msg.INSERT_FAILED_CLIPBOARD
    assert msg(InjectStatus.PARTIAL, **on) is Msg.INSERT_PARTIAL
    assert msg(InjectStatus.FAILED_CLIPBOARD_LOCKED) is Msg.INSERT_FAILED_OPEN_HISTORY
    assert msg(InjectStatus.FAILED_NO_FOCUS) is Msg.INSERT_FAILED_OPEN_HISTORY
