"""`TriggerMachine`: key, mouse and tick events to suppress decisions and effects.

Pure and I/O-free: no ctypes, no Win32, no clock. The hook thread feeds it
events stamped with the ``GetTickCount`` clock and executes what it returns. The gestures
are:

- **Hold.** The trigger key-down is swallowed while its meaning is unknown. Held alone for
  ``hold_ms - preopen_ms`` it emits `OPEN_MIC`; held for more than ``hold_ms`` it emits
  `COMMIT_HOLD`, and the release emits `FINISH`. The app never sees the key.
- **Normal press.** A release before the commit replays a trigger tap (`REPLAY_CTRL_TAP`),
  after `DISCARD_MIC` when the microphone was pre-opened. A press of at most ``tap_ms`` is
  a tap and counts toward double-tap detection.
- **Chord.** Another key or a mouse button before the commit swallows that event too and
  replays the trigger down followed by it (`REPLAY_CTRL_DOWN`, `REPLAY_SWALLOWED`). Later
  events and the trigger release pass through.
- **Hands-free.** A second press starting at most ``double_tap_ms`` after a tap ended, and
  itself a tap, emits `(OPEN_MIC, START_HANDSFREE)`. The next press finishes it.
- **Esc.** During a recording Esc cancels it; the Esc down and up never reach the app.
- A trigger press while any other key is held passes through untouched.

Every threshold is "not yet" at its exact value, except the pre-open, which fires at it.
The ``REPLAY_CTRL_*`` effects refer to the trigger key itself (Right Ctrl by default).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talktype.config import TriggerSettings

VK_ESCAPE = 0x1B

TRIGGER_KEYS: dict[str, int] = {
    "right_ctrl": 0xA3,  # VK_RCONTROL
    "right_alt": 0xA5,  # VK_RMENU
    "right_shift": 0xA1,  # VK_RSHIFT
    "caps_lock": 0x14,  # VK_CAPITAL
    "scroll_lock": 0x91,  # VK_SCROLL
    "pause": 0x13,  # VK_PAUSE
    **{f"f{n}": 0x7C + n - 13 for n in range(13, 25)},  # VK_F13..VK_F24
}


def trigger_vk(name: str) -> int:
    """The virtual key of a `trigger.key` config name. Raises ValueError."""
    try:
        return TRIGGER_KEYS[name]
    except KeyError:
        raise ValueError(f"unknown trigger key: {name!r}") from None


# --------------------------------------------------------------------------------------
# Events, effects and decisions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyEvent:
    vk: int
    down: bool
    t_ms: int  # GetTickCount-based timestamp from KBDLLHOOKSTRUCT.time


class Effect(Enum):
    OPEN_MIC = auto()
    DISCARD_MIC = auto()
    COMMIT_HOLD = auto()
    START_HANDSFREE = auto()
    FINISH = auto()
    CANCEL = auto()
    ABORT_SILENT = auto()
    REPLAY_CTRL_DOWN = auto()
    REPLAY_CTRL_UP = auto()
    REPLAY_CTRL_TAP = auto()
    REPLAY_SWALLOWED = auto()


# Effects the hook thread executes with SendInput; every other effect goes to the app.
REPLAY_EFFECTS: frozenset[Effect] = frozenset(
    {
        Effect.REPLAY_CTRL_DOWN,
        Effect.REPLAY_CTRL_UP,
        Effect.REPLAY_CTRL_TAP,
        Effect.REPLAY_SWALLOWED,
    }
)


@dataclass(frozen=True, slots=True)
class Decision:
    suppress: bool
    effects: tuple[Effect, ...] = ()


@dataclass(frozen=True, slots=True)
class TriggerTiming:
    hold_ms: int = 1000
    tap_ms: int = 200
    double_tap_ms: int = 400
    preopen_ms: int = 300

    @classmethod
    def from_settings(cls, s: TriggerSettings) -> TriggerTiming:
        return cls(
            hold_ms=s.hold_threshold_ms,
            tap_ms=s.tap_threshold_ms,
            double_tap_ms=s.double_tap_window_ms,
            preopen_ms=s.mic_preopen_ms,
        )


class TriggerState(StrEnum):
    IDLE = "idle"
    PENDING = "pending"  # trigger down swallowed, meaning still unknown
    HOLD = "hold"  # committed hold recording, trigger still down
    HANDSFREE = "handsfree"  # hands-free recording, trigger up
    HF_PENDING = "hf_pending"  # hands-free recording, trigger down (swallowed)
    CHORD = "chord"  # trigger down replayed; everything passes until its release
    HF_CHORD = "hf_chord"  # a chord made while a hands-free recording goes on
    SPENT = "spent"  # recording cancelled or aborted, trigger still down (swallowed)
    PASSTHROUGH = "passthrough"  # trigger pressed with another key held: untouched


_S = TriggerState
_PASS = Decision(False)
_SUPPRESS = Decision(True)
_CHORD = Decision(True, (Effect.REPLAY_CTRL_DOWN, Effect.REPLAY_SWALLOWED))
_CHORD_DISCARD = Decision(
    True, (Effect.DISCARD_MIC, Effect.REPLAY_CTRL_DOWN, Effect.REPLAY_SWALLOWED)
)
_CHORD_ABORT = Decision(
    True, (Effect.ABORT_SILENT, Effect.REPLAY_CTRL_DOWN, Effect.REPLAY_SWALLOWED)
)
_CANCEL = Decision(True, (Effect.CANCEL,))
_FINISH = Decision(True, (Effect.FINISH,))
_TAP = Decision(True, (Effect.REPLAY_CTRL_TAP,))
_TAP_DISCARD = Decision(True, (Effect.DISCARD_MIC, Effect.REPLAY_CTRL_TAP))
_HANDSFREE = Decision(True, (Effect.OPEN_MIC, Effect.START_HANDSFREE))
_HANDSFREE_OPEN = Decision(True, (Effect.START_HANDSFREE,))
_OPEN: tuple[Effect, ...] = (Effect.OPEN_MIC,)
_COMMIT: tuple[Effect, ...] = (Effect.COMMIT_HOLD,)
_OPEN_COMMIT: tuple[Effect, ...] = (Effect.OPEN_MIC, Effect.COMMIT_HOLD)
_DISCARD: tuple[Effect, ...] = (Effect.DISCARD_MIC,)

_RESET_EFFECTS: dict[TriggerState, tuple[Effect, ...]] = {
    _S.HOLD: (Effect.ABORT_SILENT,),
    _S.HANDSFREE: (Effect.ABORT_SILENT,),
    _S.HF_PENDING: (Effect.ABORT_SILENT,),
    _S.HF_CHORD: (Effect.ABORT_SILENT, Effect.REPLAY_CTRL_UP),
    _S.CHORD: (Effect.REPLAY_CTRL_UP,),
}


# --------------------------------------------------------------------------------------
# The machine
# --------------------------------------------------------------------------------------


class TriggerMachine:
    """Single-threaded; the hook thread serializes every call (see `winhook.HookThread`).

    `key_is_down` optionally reports whether a key is physically down. It is consulted
    only when the trigger is pressed while other keys look held, to forget key-ups that
    were never seen (for example across the secure desktop), so a missed release cannot
    turn every later trigger press into a passthrough.
    """

    __slots__ = (
        "_config_vk",
        "_down",
        "_enabled",
        "_esc_swallow",
        "_key_is_down",
        "_last_tap_up",
        "_mic_open",
        "_press_t",
        "_press_timing",
        "_second",
        "_state",
        "_timing",
        "_vk",
    )

    def __init__(
        self,
        timing: TriggerTiming,
        trigger_vk: int,
        *,
        key_is_down: Callable[[int], bool] | None = None,
    ) -> None:
        self._timing = timing
        self._config_vk = trigger_vk
        self._key_is_down = key_is_down
        self._enabled = True
        self._state = _S.IDLE
        self._vk = trigger_vk  # the key of the current press; the configured one when idle
        self._press_timing = timing
        self._press_t = 0
        self._mic_open = False
        self._second = False  # this press may complete a double-tap
        self._last_tap_up: int | None = None
        self._esc_swallow = False  # the next Esc up is swallowed (its down cancelled)
        self._down: set[int] = set()  # other keys currently held

    # -- properties --------------------------------------------------------------------

    @property
    def state(self) -> TriggerState:
        return self._state

    @property
    def needs_tick(self) -> bool:
        """A press is pending, so the 10 ms tick must run."""
        return self._state is _S.PENDING

    @property
    def in_chord(self) -> bool:
        """A trigger down was replayed and its release has not been seen yet."""
        return self._state is _S.CHORD or self._state is _S.HF_CHORD

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def trigger_vk(self) -> int:
        """The key the ``REPLAY_CTRL_*`` effects refer to (the current press's key)."""
        return self._vk

    @property
    def timing(self) -> TriggerTiming:
        return self._timing

    # -- configuration -----------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        """Disabled, a new trigger press passes through; a press in progress completes
        without committing (a pre-opened microphone is discarded at the next tick)."""
        self._enabled = enabled
        if not enabled:
            self._last_tap_up = None

    def update_timing(self, timing: TriggerTiming, trigger_vk: int | None = None) -> None:
        """New thresholds and, optionally, a new trigger key; they apply from the next press."""
        self._timing = timing
        if trigger_vk is not None and trigger_vk != self._config_vk:
            self._config_vk = trigger_vk
            self._last_tap_up = None
        if self._state is _S.IDLE:
            self._vk = self._config_vk

    # -- events ------------------------------------------------------------------------

    def on_key(self, ev: KeyEvent) -> Decision:
        if ev.vk == self._vk:
            return self._trigger_down(ev.t_ms) if ev.down else self._trigger_up(ev.t_ms)
        if ev.down:
            return self._other_down(ev.vk)
        self._down.discard(ev.vk)
        if ev.vk == VK_ESCAPE and self._esc_swallow:
            self._esc_swallow = False
            return _SUPPRESS
        return _PASS

    def on_mouse_down(self, t_ms: int) -> Decision:
        del t_ms
        state = self._state
        if state is _S.PENDING:
            return self._to_chord()
        if state is _S.HF_PENDING:
            self._state = _S.HF_CHORD
            return _CHORD
        if state is _S.SPENT:
            self._state = _S.CHORD
            return _CHORD
        if state is _S.IDLE:
            self._last_tap_up = None
        return _PASS

    def on_tick(self, t_ms: int) -> tuple[Effect, ...]:
        if self._state is not _S.PENDING:
            return ()
        if not self._enabled:
            if self._mic_open:
                self._mic_open = False
                return _DISCARD
            return ()
        timing = self._press_timing
        elapsed = t_ms - self._press_t
        if elapsed > timing.hold_ms:
            self._state = _S.HOLD
            effects = _COMMIT if self._mic_open else _OPEN_COMMIT
            self._mic_open = True
            return effects
        if (
            not self._mic_open
            and timing.preopen_ms > 0
            and elapsed >= timing.hold_ms - timing.preopen_ms
        ):
            self._mic_open = True
            return _OPEN
        return ()

    # -- commands from the app ---------------------------------------------------------

    def abort(self) -> tuple[Effect, ...]:
        """End the recording without effects (limit hit, pause, model not ready).

        A trigger still held stays swallowed until its release; a pending press is left
        alone, since it has not started a recording.
        """
        state = self._state
        if state is _S.HOLD or state is _S.HF_PENDING:
            self._state = _S.SPENT
        elif state is _S.HANDSFREE:
            self._to_idle()
        elif state is _S.HF_CHORD:
            self._state = _S.CHORD
        return ()

    def reset(self) -> tuple[Effect, ...]:
        """Back to idle (session lock, suspend, hook reinstall), releasing a replayed down."""
        state = self._state
        if state is _S.PENDING:
            effects = _DISCARD if self._mic_open else ()
        else:
            effects = _RESET_EFFECTS.get(state, ())
        self._to_idle()
        self._down.clear()
        self._last_tap_up = None
        self._esc_swallow = False
        return effects

    # -- transitions -------------------------------------------------------------------

    def _to_idle(self) -> None:
        self._state = _S.IDLE
        self._vk = self._config_vk
        self._press_timing = self._timing
        self._mic_open = False
        self._second = False

    def _to_chord(self) -> Decision:
        self._state = _S.CHORD
        if self._mic_open:
            self._mic_open = False
            return _CHORD_DISCARD
        return _CHORD

    def _others_held(self) -> bool:
        if self._down and self._key_is_down is not None:
            probe = self._key_is_down
            self._down = {vk for vk in self._down if probe(vk)}
        return bool(self._down)

    def _trigger_down(self, t_ms: int) -> Decision:
        state = self._state
        if state is _S.IDLE:
            if not self._enabled:
                return _PASS
            if self._others_held():
                self._state = _S.PASSTHROUGH
                self._last_tap_up = None
                return _PASS
            timing = self._timing
            last = self._last_tap_up
            self._second = last is not None and t_ms - last <= timing.double_tap_ms
            self._last_tap_up = None
            self._state = _S.PENDING
            self._press_timing = timing
            self._press_t = t_ms
            self._mic_open = False
            return _SUPPRESS
        if state is _S.HANDSFREE:
            self._state = _S.HF_PENDING
            return _SUPPRESS
        if state is _S.PENDING or state is _S.HOLD or state is _S.HF_PENDING:
            return _SUPPRESS  # auto-repeat
        if state is _S.SPENT:
            return _SUPPRESS
        return _PASS  # CHORD, HF_CHORD, PASSTHROUGH: the app saw the down

    def _trigger_up(self, t_ms: int) -> Decision:
        state = self._state
        if state is _S.PENDING:
            tap = t_ms - self._press_t <= self._press_timing.tap_ms
            if tap and self._second and self._enabled:
                decision = _HANDSFREE_OPEN if self._mic_open else _HANDSFREE
                self._state = _S.HANDSFREE
                self._mic_open = False
                self._second = False
                return decision
            decision = _TAP_DISCARD if self._mic_open else _TAP
            first = not self._second
            self._to_idle()
            self._last_tap_up = t_ms if tap and first and self._enabled else None
            return decision
        if state is _S.HOLD or state is _S.HF_PENDING:
            self._to_idle()
            return _FINISH
        if state is _S.SPENT:
            self._to_idle()
            return _SUPPRESS
        if state is _S.HF_CHORD:
            self._state = _S.HANDSFREE
            return _PASS
        if state is _S.CHORD or state is _S.PASSTHROUGH:
            self._to_idle()
        return _PASS  # also a stray release in IDLE or HANDSFREE

    def _other_down(self, vk: int) -> Decision:
        self._down.add(vk)
        state = self._state
        is_esc = vk == VK_ESCAPE
        if state is _S.IDLE:
            self._last_tap_up = None
            return _SUPPRESS if is_esc and self._esc_swallow else _PASS
        if state is _S.PENDING:
            return self._to_chord()
        if state is _S.HOLD:
            if is_esc:
                self._state = _S.SPENT
                self._esc_swallow = True
                return _CANCEL
            self._state = _S.CHORD
            return _CHORD_ABORT
        if state is _S.HANDSFREE:
            if is_esc:
                self._to_idle()
                self._esc_swallow = True
                return _CANCEL
            return _PASS
        if state is _S.HF_PENDING:
            if is_esc:
                self._state = _S.SPENT
                self._esc_swallow = True
                return _CANCEL
            self._state = _S.HF_CHORD
            return _CHORD
        if state is _S.SPENT:
            if is_esc:
                self._esc_swallow = True  # nothing left to cancel; keep Esc from the app
                return _SUPPRESS
            self._state = _S.CHORD
            return _CHORD
        return _PASS  # CHORD, HF_CHORD, PASSTHROUGH
