"""
core/state_machine.py

Unified gesture state machine.
All gesture state lives here. The analyser calls tick() once per frame
with the current ArmReading; the machine returns a Gesture to fire or None.

Design:
- Each gesture has explicit entry/exit conditions, not scattered flags.
- Cooldowns are per-gesture and global.
- SNAP is delayed by snap_sustain_frames so concurrent gestures
  (BOTH_UP, T_POSE, CROSS_ARMS) can supersede it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from cameras.arm_tracker import ArmReading, ArmState

log = logging.getLogger(__name__)


# ── Public Gesture enum ───────────────────────────────────────────────────────

class Gesture(Enum):
    SNAP        = auto()
    HOLD        = auto()
    DOUBLE_SNAP = auto()
    CROSS_ARMS  = auto()
    T_POSE      = auto()
    RAISE_BOTH  = auto()

    @property
    def event_name(self) -> str:
        return f"endora-{self.name.lower().replace('_', '-')}"

    def __str__(self) -> str:
        return self.event_name


@dataclass
class StateMachineConfig:
    """Timing thresholds. All values in seconds unless noted."""
    cooldown_s: float = 2.0
    snap_forearm_min: float = 0.10
    hold_duration_s: float = 1.5
    double_snap_window_s: float = 3.0

    # Sustained-state gestures need to be held this long before firing.
    # This disambiguates transitional poses (e.g. briefly looking like T_POSE
    # while raising both arms).
    sustain_s: float = 0.5

    # Minimum time the arm must be held up before SNAP fires, measured from
    # the first confirmed SINGLE_UP frame.  ArmTracker already adds state_confirm_s
    # (0.20s) before we see SINGLE_UP, so total intentional-raise time is
    # state_confirm_s + snap_sustain_s.  0.50s filters out casual/accidental
    # arm movements while remaining instant for deliberate raises.
    snap_sustain_s: float = 0.50

    # Deprecated — replaced by snap_sustain_s.  Kept so old configs don't error.
    snap_sustain_frames: int = 1

    # grlib snap_roll threshold: if |reading.snap_roll| >= this value,
    # snap fires even when forearm_dy is below snap_forearm_min.
    # 0.0 = disabled (rely on forearm_dy only).
    snap_roll_threshold: float = 0.0


# ── Internal per-arm-raise state ──────────────────────────────────────────────

@dataclass
class _RaiseState:
    """State tracked for the duration of a single SINGLE_UP raise."""
    snap_fired:    bool  = False
    hold_fired:    bool  = False
    snap_fired_at: float = 0.0
    up_frames:     int   = 0
    entered_at:    float = 0.0   # monotonic time of first SINGLE_UP frame


@dataclass
class _SustainState:
    """How long each sustained-state gesture has been held continuously."""
    entered_at: dict = field(default_factory=dict)  # ArmState → monotonic time


# ── State Machine ────────────────────────────────────────────────────────────

class GestureStateMachine:
    def __init__(self, config: StateMachineConfig,
                 on_near_miss: Optional[Callable[[str, str, ArmReading], None]] = None):
        self.c = config
        self._raise = _RaiseState()
        self._sustain = _SustainState()
        # Optional callback(gesture_name, reason, reading) for near-miss events.
        self._on_near_miss = on_near_miss

        # Ring buffer of recent SNAP fire times for DOUBLE_SNAP detection.
        self._snap_times: list[float] = []

        # Per-gesture last fired (for per-gesture cooldown) and global.
        # -inf sentinel means "never fired" — avoids blocking very first tick.
        self._last_fired: dict[Gesture, float] = {g: float('-inf') for g in Gesture}
        self._last_fired_any: float = float('-inf')

        self.total_emitted = 0

    # ── Public API ────────────────────────────────────────────────────────

    def tick(self, reading: Optional[ArmReading], now: float) -> Optional[Gesture]:
        """
        Advance one frame. Returns a gesture to fire, or None.
        `now` is a monotonic timestamp in seconds.
        """
        # No pose or arm down → reset raise state, clear sustain timers.
        if reading is None or reading.state == ArmState.DOWN:
            self._reset_raise()
            self._sustain.entered_at.clear()
            return None

        state = reading.state

        # Cooldown gate for sustained-state gestures only — these need the
        # cooldown to avoid rapid re-fire while the user holds the pose.
        # SINGLE_UP doesn't need the cooldown because the per-raise flags
        # (_snap_fired, _hold_fired) already prevent repeated firing, and
        # enforcing cooldown here blocks DOUBLE_SNAP from working after SNAP.
        if state != ArmState.SINGLE_UP:
            if now - self._last_fired_any < self.c.cooldown_s:
                return None

        # Dispatch per state
        if state == ArmState.SINGLE_UP:
            self._sustain.entered_at.clear()  # no sustained state active
            return self._tick_single_up(reading, now)

        if state in (ArmState.BOTH_UP, ArmState.T_POSE, ArmState.CROSS_ARMS):
            self._reset_raise()
            return self._tick_sustained(state, now)

        return None

    # ── Handlers ──────────────────────────────────────────────────────────

    def _tick_single_up(self, reading: ArmReading, now: float) -> Optional[Gesture]:
        r = self._raise
        r.up_frames += 1
        if r.up_frames == 1:
            r.entered_at = now  # record when this raise began

        arm_vertical = reading.forearm_dy >= self.c.snap_forearm_min
        roll_snap = (
            self.c.snap_roll_threshold > 0
            and abs(reading.snap_roll) >= self.c.snap_roll_threshold
        )
        snap_condition = arm_vertical or roll_snap

        # HOLD: arm still vertical, SNAP already fired, enough time passed
        if (r.snap_fired and not r.hold_fired and snap_condition
                and (now - r.snap_fired_at) >= self.c.hold_duration_s):
            r.hold_fired = True
            return self._fire(Gesture.HOLD, now)

        # SNAP: arm has been held up long enough (time-based, rate-independent)
        if (not r.snap_fired and snap_condition
                and (now - r.entered_at) >= self.c.snap_sustain_s):
            return self._fire_snap(now)

        # Near-miss: arm is up but snap condition not met — log for tuning.
        if not r.snap_fired and self._on_near_miss and r.up_frames > 1:
            if not snap_condition:
                reason = (f"forearm_dy={reading.forearm_dy:.3f} < {self.c.snap_forearm_min}"
                          f" (min), snap_roll={reading.snap_roll:.3f}")
                self._on_near_miss("SNAP", reason, reading)
            elif (now - r.entered_at) < self.c.snap_sustain_s:
                held = now - r.entered_at
                reason = f"sustain={held:.3f}s < {self.c.snap_sustain_s}s required"
                self._on_near_miss("SNAP", reason, reading)

        return None

    def _tick_sustained(self, state: ArmState, now: float) -> Optional[Gesture]:
        # Track continuous time in this state
        entered = self._sustain.entered_at.get(state)
        if entered is None:
            self._sustain.entered_at = {state: now}  # reset others
            return None

        if (now - entered) < self.c.sustain_s:
            return None

        # Held long enough — fire
        gesture = {
            ArmState.BOTH_UP:    Gesture.RAISE_BOTH,
            ArmState.T_POSE:     Gesture.T_POSE,
            ArmState.CROSS_ARMS: Gesture.CROSS_ARMS,
        }[state]
        self._sustain.entered_at.clear()
        return self._fire(gesture, now)

    # ── SNAP + DOUBLE_SNAP logic ──────────────────────────────────────────

    def _fire_snap(self, now: float) -> Gesture:
        r = self._raise
        r.snap_fired = True
        r.snap_fired_at = now

        # DOUBLE_SNAP: prior SNAP within window?
        self._snap_times[:] = [
            t for t in self._snap_times
            if now - t < self.c.double_snap_window_s
        ]
        if self._snap_times:
            self._snap_times.clear()
            return self._fire(Gesture.DOUBLE_SNAP, now)

        self._snap_times.append(now)
        return self._fire(Gesture.SNAP, now)

    # ── Fire helper ───────────────────────────────────────────────────────

    def _fire(self, gesture: Gesture, now: float) -> Gesture:
        self._last_fired[gesture] = now
        self._last_fired_any = now
        self.total_emitted += 1
        log.info("Gesture fired: %s", gesture)
        return gesture

    def _reset_raise(self) -> None:
        self._raise = _RaiseState()
