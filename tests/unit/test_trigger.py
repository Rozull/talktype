"""`TriggerMachine` transition tables (UT-001 … UT-033, UT-209, UT-213).

Each table is a list of ``(step, expected)``: key and mouse steps expect a `Decision`;
tick, abort and reset steps expect a tuple of effects. Defaults: hold 2000 ms, tap 200 ms,
double-tap 400 ms, pre-open 300 ms, trigger Right Ctrl.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from talktype import trigger
from talktype.config import TriggerSettings
from talktype.trigger import (
    REPLAY_EFFECTS,
    TRIGGER_KEYS,
    Decision,
    Effect,
    KeyEvent,
    TriggerMachine,
    TriggerState,
    TriggerTiming,
    trigger_vk,
)

pytestmark = pytest.mark.unit

RCTRL = 0xA3
LCTRL = 0xA2
LSHIFT = 0xA0
ESC = 0x1B
KEY_A = 0x41
KEY_C = 0x43
KEY_V = 0x56
F13 = 0x7C

E = Effect
PASS = Decision(False)
SUPPRESS = Decision(True)


def swallow(*effects: Effect) -> Decision:
    return Decision(True, effects)


# --------------------------------------------------------------------------------------
# Table driver
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    kind: str  # "down", "up", "mouse", "tick", "abort", "reset"
    t_ms: int = 0
    vk: int = 0

    def __str__(self) -> str:
        return f"{self.kind}(vk=0x{self.vk:02X}, t={self.t_ms})"


def down(vk: int, t_ms: int) -> Step:
    return Step("down", t_ms, vk)


def up(vk: int, t_ms: int) -> Step:
    return Step("up", t_ms, vk)


def mouse(t_ms: int) -> Step:
    return Step("mouse", t_ms)


def tick(t_ms: int) -> Step:
    return Step("tick", t_ms)


ABORT = Step("abort")
RESET = Step("reset")

Row = tuple[Step, Decision | tuple[Effect, ...]]


def apply(machine: TriggerMachine, step: Step) -> Decision | tuple[Effect, ...]:
    if step.kind in ("down", "up"):
        return machine.on_key(KeyEvent(step.vk, step.kind == "down", step.t_ms))
    if step.kind == "mouse":
        return machine.on_mouse_down(step.t_ms)
    if step.kind == "tick":
        return machine.on_tick(step.t_ms)
    if step.kind == "abort":
        return machine.abort()
    assert step.kind == "reset"
    return machine.reset()


def drive(machine: TriggerMachine, table: list[Row]) -> None:
    for index, (step, expected) in enumerate(table):
        got = apply(machine, step)
        assert got == expected, f"row {index}: {step} returned {got}, expected {expected}"


# These cases are written for a 2000 ms hold; the shipped default is 1000 ms.
CONTRACT = TriggerTiming(hold_ms=2000)


def make(timing: TriggerTiming | None = None, vk: int = RCTRL) -> TriggerMachine:
    return TriggerMachine(timing or CONTRACT, vk)


def committed_hold() -> list[Row]:
    return [(down(RCTRL, 0), SUPPRESS), (tick(1700), (E.OPEN_MIC,)), (tick(2001), (E.COMMIT_HOLD,))]


def into_handsfree(start: int = 0) -> list[Row]:
    return [
        (down(RCTRL, start), SUPPRESS),
        (up(RCTRL, start + 100), swallow(E.REPLAY_CTRL_TAP)),
        (down(RCTRL, start + 200), SUPPRESS),
        (up(RCTRL, start + 250), swallow(E.OPEN_MIC, E.START_HANDSFREE)),
    ]


# --------------------------------------------------------------------------------------
# Hold, thresholds, chords and modifiers
# --------------------------------------------------------------------------------------


def test_ut001_hold_commits_after_threshold() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(10), ()),
            (tick(1000), ()),
            (tick(1700), (E.OPEN_MIC,)),
            (tick(1800), ()),
            (tick(2001), (E.COMMIT_HOLD,)),
            (tick(2100), ()),
        ],
    )
    assert machine.state is TriggerState.HOLD
    assert machine.needs_tick is False


def test_ut002_release_after_commit_finishes() -> None:
    machine = make()
    drive(machine, [*committed_hold(), (up(RCTRL, 5000), swallow(E.FINISH))])
    assert machine.state is TriggerState.IDLE


def test_ut003_release_at_tap_threshold_is_a_tap() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(100), ()),
            (tick(200), ()),
            (up(RCTRL, 200), swallow(E.REPLAY_CTRL_TAP)),
            # recorded as a tap: a second tap within the window starts hands-free
            (down(RCTRL, 600), SUPPRESS),
            (up(RCTRL, 650), swallow(E.OPEN_MIC, E.START_HANDSFREE)),
        ],
    )


def test_ut004_release_at_hold_threshold_discards_the_preopened_mic() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(1700), (E.OPEN_MIC,)),
            (tick(2000), ()),
            (up(RCTRL, 2000), swallow(E.DISCARD_MIC, E.REPLAY_CTRL_TAP)),
            (tick(2010), ()),
        ],
    )


def test_ut005_medium_press_replays_a_tap_that_does_not_count_for_double_tap() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(500), ()),
            (tick(1000), ()),
            (up(RCTRL, 1000), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, 1100), SUPPRESS),
            (up(RCTRL, 1150), swallow(E.REPLAY_CTRL_TAP)),
        ],
    )


def test_ut006_key_before_commit_is_a_chord() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (down(KEY_C, 50), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (up(KEY_C, 80), PASS),
        ],
    )
    assert machine.in_chord is True
    drive(machine, [(up(RCTRL, 120), PASS)])
    assert machine.in_chord is False
    assert machine.state is TriggerState.IDLE


def test_ut007_chord_after_preopen_discards_the_mic() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(1700), (E.OPEN_MIC,)),
            (down(KEY_C, 1800), swallow(E.DISCARD_MIC, E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (tick(2100), ()),
            (up(KEY_C, 2200), PASS),
            (up(RCTRL, 2300), PASS),
        ],
    )


def test_ut008_mouse_button_before_commit_is_a_ctrl_click() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (mouse(100), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (up(RCTRL, 200), PASS),
        ],
    )


def test_ut009_left_ctrl_is_never_the_trigger() -> None:
    drive(
        make(),
        [(down(LCTRL, 0), PASS), (tick(2500), ()), (up(LCTRL, 3000), PASS)],
    )


def test_ut010_trigger_after_another_key_passes_through() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(LSHIFT, 0), PASS),
            (down(RCTRL, 10), PASS),
            (down(RCTRL, 40), PASS),  # auto-repeat
            (tick(2100), ()),
            (tick(3000), ()),
            (up(RCTRL, 3000), PASS),
            (up(LSHIFT, 3010), PASS),
        ],
    )
    assert machine.state is TriggerState.IDLE


def test_ut011_trigger_while_left_ctrl_is_held_passes_through() -> None:
    drive(
        make(),
        [(down(LCTRL, 0), PASS), (down(RCTRL, 5), PASS), (tick(2500), ()), (up(RCTRL, 2600), PASS)],
    )


def test_stale_held_key_is_forgotten_through_the_probe() -> None:
    # A key-up lost across the secure desktop must not make the trigger pass forever.
    physically_down: set[int] = set()
    machine = TriggerMachine(CONTRACT, RCTRL, key_is_down=physically_down.__contains__)
    drive(machine, [(down(LSHIFT, 0), PASS)])  # its release is never seen
    drive(machine, [(down(RCTRL, 5000), SUPPRESS), (up(RCTRL, 5050), swallow(E.REPLAY_CTRL_TAP))])
    physically_down.add(LSHIFT)
    drive(machine, [(down(LSHIFT, 6000), PASS), (down(RCTRL, 6010), PASS)])


# --------------------------------------------------------------------------------------
# Double-tap hands-free
# --------------------------------------------------------------------------------------


def test_ut012_double_tap_starts_handsfree() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (up(RCTRL, 100), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, 350), SUPPRESS),
            (up(RCTRL, 420), swallow(E.OPEN_MIC, E.START_HANDSFREE)),
        ],
    )
    assert machine.state is TriggerState.HANDSFREE


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (400, swallow(E.OPEN_MIC, E.START_HANDSFREE)),
        (401, swallow(E.REPLAY_CTRL_TAP)),
    ],
)
def test_ut013_double_tap_window_boundary(gap: int, expected: Decision) -> None:
    second = 100 + gap
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (up(RCTRL, 100), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, second), SUPPRESS),
            (up(RCTRL, second + 50), expected),
        ],
    )


def test_ut014_press_in_handsfree_finishes() -> None:
    machine = make()
    drive(
        machine,
        [*into_handsfree(), (down(RCTRL, 5000), SUPPRESS), (up(RCTRL, 5150), swallow(E.FINISH))],
    )
    assert machine.state is TriggerState.IDLE


def test_ut015_triple_tap_starts_and_finishes_then_a_fourth_is_a_plain_tap() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (up(RCTRL, 20), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, 100), SUPPRESS),
            (up(RCTRL, 120), swallow(E.OPEN_MIC, E.START_HANDSFREE)),
            (down(RCTRL, 200), SUPPRESS),
            (up(RCTRL, 220), swallow(E.FINISH)),
            (down(RCTRL, 300), SUPPRESS),
            (up(RCTRL, 320), swallow(E.REPLAY_CTRL_TAP)),
        ],
    )


def test_ut016_chord_in_handsfree_keeps_recording() -> None:
    machine = make()
    drive(
        machine,
        [
            *into_handsfree(),
            (down(RCTRL, 3000), SUPPRESS),
            (down(KEY_V, 3050), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (up(KEY_V, 3080), PASS),
        ],
    )
    assert machine.in_chord is True
    drive(machine, [(up(RCTRL, 3100), PASS)])
    assert machine.state is TriggerState.HANDSFREE
    assert machine.in_chord is False
    drive(machine, [(down(RCTRL, 4000), SUPPRESS), (up(RCTRL, 4050), swallow(E.FINISH))])


def test_ut017_tap_then_hold_commits_a_hold() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (up(RCTRL, 100), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, 200), SUPPRESS),
            (tick(1899), ()),
            (tick(1900), (E.OPEN_MIC,)),
            (tick(2200), ()),
            (tick(2201), (E.COMMIT_HOLD,)),
            (up(RCTRL, 2500), swallow(E.FINISH)),
        ],
    )


def test_ut018_tap_then_long_press_is_a_plain_tap() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (up(RCTRL, 100), swallow(E.REPLAY_CTRL_TAP)),
            (down(RCTRL, 300), SUPPRESS),
            (tick(600), ()),
            (up(RCTRL, 900), swallow(E.REPLAY_CTRL_TAP)),
        ],
    )


# --------------------------------------------------------------------------------------
# Esc cancel and passthrough
# --------------------------------------------------------------------------------------


def test_ut019_esc_cancels_a_hold_and_is_swallowed() -> None:
    machine = make()
    drive(
        machine,
        [
            *committed_hold(),
            (down(ESC, 2500), swallow(E.CANCEL)),
            (down(ESC, 2530), SUPPRESS),  # auto-repeat
            (up(ESC, 2550), SUPPRESS),
            (up(RCTRL, 3000), SUPPRESS),
        ],
    )
    assert machine.state is TriggerState.IDLE


def test_ut213_esc_after_a_cancel_reaches_the_app() -> None:
    machine = make()
    drive(
        machine,
        [
            *committed_hold(),
            (down(ESC, 2500), swallow(E.CANCEL)),
            (up(ESC, 2550), SUPPRESS),
            (up(RCTRL, 3000), SUPPRESS),
            (down(ESC, 3500), PASS),
            (up(ESC, 3550), PASS),
        ],
    )


def test_ut020_esc_cancels_handsfree_and_is_swallowed() -> None:
    machine = make()
    drive(
        machine,
        [
            *into_handsfree(),
            (down(ESC, 3000), swallow(E.CANCEL)),
            (up(ESC, 3050), SUPPRESS),
            (down(ESC, 4000), PASS),
            (up(ESC, 4050), PASS),
        ],
    )
    assert machine.state is TriggerState.IDLE


def test_ut021_esc_in_idle_passes() -> None:
    drive(make(), [(down(ESC, 0), PASS), (up(ESC, 50), PASS)])


def test_ut022_esc_before_commit_is_ctrl_esc() -> None:
    drive(
        make(),
        [
            (down(RCTRL, 0), SUPPRESS),
            (down(ESC, 500), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (up(ESC, 550), PASS),
            (up(RCTRL, 600), PASS),
        ],
    )


def test_ut023_key_during_hold_aborts_into_a_chord() -> None:
    machine = make()
    drive(
        machine,
        [
            *committed_hold(),
            (
                down(LSHIFT, 2500),
                swallow(E.ABORT_SILENT, E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED),
            ),
        ],
    )
    assert machine.state is TriggerState.CHORD
    drive(
        machine,
        [(down(ESC, 2600), PASS), (up(ESC, 2650), PASS), (up(LSHIFT, 2700), PASS)],
    )
    drive(machine, [(up(RCTRL, 2800), PASS)])


# --------------------------------------------------------------------------------------
# Abort, reset, enable, timing, repeat, custom key, pre-open, stray events
# --------------------------------------------------------------------------------------


def test_ut024_abort_swallows_the_rest_of_the_press() -> None:
    drive(
        make(),
        [
            *committed_hold(),
            (ABORT, ()),
            (tick(2500), ()),
            (up(RCTRL, 3000), SUPPRESS),
            (down(RCTRL, 5000), SUPPRESS),
            (tick(6699), ()),
            (tick(6700), (E.OPEN_MIC,)),
        ],
    )


def test_abort_in_handsfree_returns_to_idle() -> None:
    machine = make()
    drive(machine, [*into_handsfree(), (ABORT, ())])
    assert machine.state is TriggerState.IDLE


def test_ut025_reset_in_chord_releases_the_replayed_ctrl() -> None:
    machine = make()
    drive(
        machine,
        [
            (down(RCTRL, 0), SUPPRESS),
            (down(KEY_C, 50), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            (RESET, (E.REPLAY_CTRL_UP,)),
        ],
    )
    assert machine.state is TriggerState.IDLE
    assert machine.in_chord is False


def test_ut026_reset_in_hold_aborts_silently() -> None:
    machine = make()
    drive(machine, [*committed_hold(), (RESET, (E.ABORT_SILENT,))])
    assert machine.state is TriggerState.IDLE
    drive(machine, [(up(RCTRL, 3000), PASS)])


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ([], ()),
        ([(down(RCTRL, 0), SUPPRESS)], ()),
        ([(down(RCTRL, 0), SUPPRESS), (tick(1700), (E.OPEN_MIC,))], (E.DISCARD_MIC,)),
        (into_handsfree(), (E.ABORT_SILENT,)),
        (
            [
                *into_handsfree(),
                (down(RCTRL, 3000), SUPPRESS),
                (down(KEY_V, 3050), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED)),
            ],
            (E.ABORT_SILENT, E.REPLAY_CTRL_UP),
        ),
    ],
    ids=["idle", "pending", "pending_mic_open", "handsfree", "handsfree_chord"],
)
def test_reset_from_every_state(setup: list[Row], expected: tuple[Effect, ...]) -> None:
    machine = make()
    drive(machine, [*setup, (RESET, expected)])
    assert machine.state is TriggerState.IDLE


def test_ut027_disabled_trigger_passes_through() -> None:
    machine = make()
    machine.set_enabled(False)
    assert machine.enabled is False
    drive(
        machine,
        [
            (down(RCTRL, 0), PASS),
            (tick(1700), ()),
            (tick(2500), ()),
            (tick(3000), ()),
            (up(RCTRL, 3000), PASS),
        ],
    )
    machine.set_enabled(True)
    drive(machine, [(down(RCTRL, 4000), SUPPRESS), (tick(6001), (E.OPEN_MIC, E.COMMIT_HOLD))])


def test_disabling_during_a_press_never_commits() -> None:
    machine = make()
    drive(machine, [(down(RCTRL, 0), SUPPRESS), (tick(1700), (E.OPEN_MIC,))])
    machine.set_enabled(False)
    drive(
        machine,
        [
            (tick(1710), (E.DISCARD_MIC,)),
            (tick(2500), ()),
            (up(RCTRL, 2600), swallow(E.REPLAY_CTRL_TAP)),
        ],
    )


def test_ut028_timing_change_applies_from_the_next_press() -> None:
    machine = make()
    drive(machine, [(down(RCTRL, 0), SUPPRESS)])
    machine.update_timing(TriggerTiming(hold_ms=500))
    drive(
        machine,
        [
            (tick(600), ()),
            (tick(1700), (E.OPEN_MIC,)),
            (tick(2000), ()),
            (tick(2001), (E.COMMIT_HOLD,)),
            (up(RCTRL, 2100), swallow(E.FINISH)),
            (down(RCTRL, 3000), SUPPRESS),
            (tick(3199), ()),
            (tick(3200), (E.OPEN_MIC,)),
            (tick(3500), ()),
            (tick(3501), (E.COMMIT_HOLD,)),
        ],
    )
    assert machine.timing.hold_ms == 500


def test_trigger_key_change_applies_from_the_next_press() -> None:
    machine = make()
    drive(machine, [(down(RCTRL, 0), SUPPRESS)])
    machine.update_timing(CONTRACT, F13)
    assert machine.trigger_vk == RCTRL  # the current press keeps its key
    drive(machine, [(down(F13, 50), swallow(E.REPLAY_CTRL_DOWN, E.REPLAY_SWALLOWED))])
    drive(machine, [(up(F13, 60), PASS), (up(RCTRL, 100), PASS)])
    assert machine.trigger_vk == F13
    drive(machine, [(down(RCTRL, 200), PASS), (up(RCTRL, 250), PASS)])
    drive(machine, [(down(F13, 300), SUPPRESS), (up(F13, 350), swallow(E.REPLAY_CTRL_TAP))])


def test_ut029_auto_repeat_is_swallowed_without_duplicate_effects() -> None:
    machine = make()
    effects: list[Effect] = []
    assert machine.on_key(KeyEvent(RCTRL, True, 0)) == SUPPRESS
    for t in range(10, 3001, 10):
        if t % 30 == 0:
            assert machine.on_key(KeyEvent(RCTRL, True, t)) == SUPPRESS
        effects.extend(machine.on_tick(t))
    assert effects == [E.OPEN_MIC, E.COMMIT_HOLD]
    assert machine.on_key(KeyEvent(RCTRL, False, 3100)) == swallow(E.FINISH)


def test_ut030_custom_trigger_key() -> None:
    assert trigger_vk("f13") == F13
    machine = make(vk=F13)
    drive(
        machine,
        [
            (down(RCTRL, 0), PASS),
            (up(RCTRL, 50), PASS),
            (down(F13, 100), SUPPRESS),
            (tick(1800), (E.OPEN_MIC,)),
            (tick(2101), (E.COMMIT_HOLD,)),
            (up(F13, 2500), swallow(E.FINISH)),
        ],
    )


def test_ut031_preopen_boundary_and_zero_preopen() -> None:
    drive(make(), [(down(RCTRL, 0), SUPPRESS), (tick(1699), ()), (tick(1700), (E.OPEN_MIC,))])
    drive(
        make(TriggerTiming(hold_ms=2000, preopen_ms=0)),
        [
            (down(RCTRL, 0), SUPPRESS),
            (tick(1700), ()),
            (tick(2000), ()),
            (tick(2001), (E.OPEN_MIC, E.COMMIT_HOLD)),
        ],
    )


def test_ut032_handsfree_start_effects_are_ordered() -> None:
    machine = make()
    machine.on_key(KeyEvent(RCTRL, True, 0))
    machine.on_key(KeyEvent(RCTRL, False, 50))
    machine.on_key(KeyEvent(RCTRL, True, 100))
    decision = machine.on_key(KeyEvent(RCTRL, False, 150))
    assert decision.suppress is True
    assert decision.effects == (Effect.OPEN_MIC, Effect.START_HANDSFREE)


def test_ut033_stray_release_in_idle_passes() -> None:
    machine = make()
    drive(machine, [(up(RCTRL, 100), PASS), (tick(200), ())])
    assert machine.state is TriggerState.IDLE


# --------------------------------------------------------------------------------------
# Performance, configuration and purity
# --------------------------------------------------------------------------------------


def test_ut209_ten_thousand_events_stay_within_the_callback_budget() -> None:
    events = [KeyEvent(RCTRL if i % 4 < 2 else KEY_A, i % 2 == 0, i * 7) for i in range(10_000)]
    best = float("inf")
    for _ in range(3):  # the best of three runs, to ignore scheduler noise
        machine = make()
        start = time.perf_counter()
        for event in events:
            machine.on_key(event)
        best = min(best, time.perf_counter() - start)
    assert best < 0.050, f"{best * 1000:.1f} ms"


def test_trigger_keys_cover_every_config_name() -> None:
    names = ["right_ctrl", "right_alt", "right_shift", "caps_lock", "scroll_lock", "pause"]
    names += [f"f{n}" for n in range(13, 25)]
    assert list(TRIGGER_KEYS) == names
    assert TRIGGER_KEYS["right_ctrl"] == RCTRL
    assert TRIGGER_KEYS["f24"] == 0x87
    with pytest.raises(ValueError, match="banana"):
        trigger_vk("banana")


def test_timing_from_settings() -> None:
    settings = TriggerSettings(
        hold_threshold_ms=1500, tap_threshold_ms=150, double_tap_window_ms=300, mic_preopen_ms=0
    )
    assert TriggerTiming.from_settings(settings) == TriggerTiming(1500, 150, 300, 0)
    assert TriggerTiming.from_settings(TriggerSettings()) == TriggerTiming()


def test_replay_effects_are_the_four_replays() -> None:
    assert {e for e in Effect if e.name.startswith("REPLAY_")} == REPLAY_EFFECTS


def test_trigger_module_is_pure() -> None:
    tree = ast.parse(Path(trigger.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "ctypes" not in imported
    assert "talktype.win32" not in imported
    assert imported <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "typing",
        "talktype.config",
    }
