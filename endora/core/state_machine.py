"""
core/state_machine.py

Unified gesture state machine.
All gesture state lives here. The analyser calls tick() once per frame
with the current ArmReading; the machine returns a Gesture to fire or None.

Design:
- Each gesture has explicit entry/exit conditions, not scattered flags.
- SNAP is delayed by snap_sustain_s so concurrent gestures
  (BOTH_UP, T_POSE, CROSS_ARMS) can supersede it.

Cooldowns: this layer keeps ONE, the global gate on sustained poses, and its
only job is to stop the residual motion of a gesture immediately reading as a
different one — so it is sized like fusion's cross_gesture_cooldown_s, not
like a per-gesture repeat guard. Repeat suppression belongs elsewhere: a held
pose is latched by sustained_rearm_s, a raise by the per-raise flags, and the
per-gesture cooldown that decides what actually reaches Home Assistant lives
in core/fusion.py. Three cooldowns in two layers is how a spurious CROSS_ARMS
came to swallow a real SNAP for two seconds.
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
    FOLDED_ARMS = auto()

    @property
    def event_name(self) -> str:
        return f"endora-{self.name.lower().replace('_', '-')}"

    def __str__(self) -> str:
        return self.event_name


@dataclass
class StateMachineConfig:
    """Timing thresholds. All values in seconds unless noted."""
    # Deprecated in this layer — the per-gesture repeat cooldown lives in
    # core/fusion.py, which is what decides whether an event reaches Home
    # Assistant. Kept so existing configs still load. See the module
    # docstring for which layer owns which cooldown.
    cooldown_s: float = 2.0
    # The one cooldown this layer keeps: after any gesture fires, how long
    # before a SUSTAINED pose may fire. Its only job is to stop the residual
    # motion of one gesture immediately reading as a different one, so it is
    # sized like fusion's setting of the same name rather than like a repeat
    # guard — repeats are already prevented by sustained_rearm_s for a pose
    # and by the per-raise flags for a raise.
    #
    # At the old 2.0 s this gate did far more than it was meant to: snapping
    # and then deliberately folding your arms produced the fold 2.4 s late,
    # because the gate returns BEFORE the sustain timer is seeded, so you pay
    # the cooldown and then the full sustain_s again. That is the same fault
    # already fixed one layer up, where a spurious CROSS_ARMS was swallowing
    # a real SNAP for two seconds; the fix never reached here.
    cross_gesture_cooldown_s: float = 0.5
    # Minimum arm elevation for a raise to count as SNAP/HOLD. The tracker
    # has already applied its own raise_elevation_min to reach SINGLE_UP, so
    # this only bites when set higher — i.e. to demand a straighter-up arm
    # for firing than for merely showing "arm up" on the overlay.
    snap_elevation_min: float = 0.70
    # Deprecated — the forearm-verticality test was replaced by elevation,
    # which measures the whole arm rather than the forearm alone. Kept so
    # old configs don't error.
    snap_forearm_min: float = 0.10
    hold_duration_s: float = 1.5
    double_snap_window_s: float = 3.0

    # Sustained-state gestures need to be held this long before firing.
    # This disambiguates transitional poses (e.g. briefly looking like T_POSE
    # while raising both arms).
    sustain_s: float = 0.5
    # How long a sustained pose may go unmatched without the hold being
    # considered released. See _SustainState: the pose is steady, the
    # keypoints are not.
    sustain_gap_s: float = 0.85
    # The most any single interval between two matched frames may contribute
    # towards sustain_s, expressed in TICKS rather than seconds: an ordinary
    # sampling step is one tick and is credited in full, while a forgiven gap
    # spanning many ticks is credited only this much, so the gap cannot stand
    # in for the hold it interrupted.
    #
    # Measured in ticks because a constant in seconds would be a rate
    # -dependent threshold, and this project has been bitten by those at both
    # ends already (person pruning, the motion gate). tick() is called on
    # EVERY frame whatever the pose, so the machine can measure its own tick
    # interval and needs no configured frame rate — see _tick_dt.
    sustain_credit_ticks: float = 2.5

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

    # ── Flourish ──────────────────────────────────────────────────────────
    # The gesture is a theatrical arm sweep — up and (usually) straight back
    # down — not a raised hand held still. Requiring the sweep instead of a
    # hold is also what makes every static false positive impossible by
    # construction: an arm resting on an armrest, draped over a backrest, or
    # holding a phone produces a sweep rate of essentially zero, however
    # much it resembles a raise geometrically.
    snap_require_flourish: bool = True
    # Elevation the arm must gain (see ArmReading.sweep_climb). 0.60 admits a
    # flourish that starts from an armrest (~0.7 available) while rejecting a
    # draped arm merely shifting position (~0.4).
    flourish_min_climb: float = 0.60
    # …and how fast, in elevation units per second. A resting arm sits near
    # 0; a deliberate flourish runs 2-3. 0.80 leaves room for a slow,
    # unhurried sweep without admitting drift.
    flourish_min_rate: float = 0.80
    # …and an upper bound, because the minimum alone rewards nonsense. The
    # rate is a climb divided by the interval it happened in, so a keypoint
    # that jumps between two adjacent samples reports an enormous one. A
    # false snap measured 8.82/s and an earlier one 15.25/s, while every
    # genuine snap recorded here sits between 0.89 and 1.85. An arm does not
    # sweep faster than this; a detector glitch does.
    flourish_max_rate: float = 4.00

    # Legacy hold-style gating, used only when snap_require_flourish is off:
    # require the arm to have risen, and to be held still, before firing.
    snap_require_rise: bool = True
    snap_require_still: bool = False

    # A sustained-pose gesture (CROSS_ARMS / T_POSE / RAISE_BOTH) fires once
    # per pose entry and then latches: it cannot re-fire until the pose has
    # been ABSENT for this many seconds. Without the latch, sitting with
    # arms crossed re-fired CROSS_ARMS on every cooldown — ~100 fires in 20
    # minutes of normal TV-watching in live feedback (2026-07-11).
    sustained_rearm_s: float = 2.0

    # ── Per-gesture enable ────────────────────────────────────────────────
    # A gesture you never perform is not free: it still fires HA events, and
    # until v1.9.130 an unwanted one could suppress a real gesture through
    # the shared cooldown. Turn off what you don't use.
    enable_snap: bool = True
    enable_hold: bool = True
    enable_double_snap: bool = True
    enable_cross_arms: bool = True
    enable_t_pose: bool = True
    enable_raise_both: bool = True
    enable_folded_arms: bool = True


# ── Internal per-arm-raise state ──────────────────────────────────────────────

@dataclass
class _RaiseState:
    """State tracked for the duration of a single SINGLE_UP raise."""
    # Whether a snap was DETECTED during this raise. Suppresses re-detection
    # every frame, and is set even when the gesture is switched off.
    snap_fired:    bool  = False
    # Whether that detection actually became an event. HOLD extends an
    # emitted raise gesture, so it keys off this one: with enable_snap off
    # the raise produced nothing, and a HOLD 1.5 s later is a gesture the
    # user switched off arriving under a different name.
    snap_emitted:  bool  = False
    hold_fired:    bool  = False
    snap_fired_at: float = 0.0
    up_frames:     int   = 0
    entered_at:    float = 0.0   # monotonic time of first SINGLE_UP frame
    # Latched once a qualifying sweep is seen during THIS raise. The sweep
    # only qualifies transiently — for the moments around the top of the
    # arc — so requiring it to coincide with every other condition made
    # firing a matter of timing luck. Remembering it makes the question
    # "did this raise arrive on a flourish", which is what we actually mean.
    flourish_seen: bool  = False
    # Trajectory-gate near-miss reasons already logged for this raise — these
    # gates can stay blocked for minutes (hand propped against head), and
    # logging them per-frame would flood feedback.jsonl.
    gates_logged:  set   = field(default_factory=set)


@dataclass
class _SustainState:
    """How long each sustained-state gesture has been held.

    "Held" is deliberately not "matched on every frame". Recorded traces of
    someone holding folded arms show the wrist keypoints flickering badly —
    two hands pressed together look like one blob, so the model guesses
    which wrist is where, and the measured gap swings between 0.06 and 0.85
    shoulder widths in adjacent frames while the person has not moved. The
    pose is steady; the measurement of it is not. Demanding an unbroken run
    meant the gesture fired only when a good frame happened to land under
    the timer, which reads as "it takes ages to recognise".

    So a gap shorter than sustain_gap_s does not end the hold — but neither
    does it COUNT as holding. Those are separate questions, and conflating
    them is a hole: measuring elapsed wall-clock from the first sighting let
    two lone frames 0.6 s apart, with nothing at all in between, satisfy a
    0.5 s sustain. Instead each matched frame credits the interval since the
    previous one, capped at sustain_credit_max_s, so an ordinary sampling
    step is credited in full and a gap contributes almost nothing.
    """
    entered_at: dict = field(default_factory=dict)  # ArmState → first seen
    last_seen: dict = field(default_factory=dict)   # ArmState → most recent
    held_s: dict = field(default_factory=dict)      # ArmState → credited time


# ── State Machine ────────────────────────────────────────────────────────────

# Which gesture each sustained pose fires. Module level so the analyser can
# read it too — it plays a pose's sound at onset rather than waiting for the
# hold to complete, and needs to know which sound that is.
POSE_GESTURE = {
    ArmState.BOTH_UP:     Gesture.RAISE_BOTH,
    ArmState.T_POSE:      Gesture.T_POSE,
    ArmState.CROSS_ARMS:  Gesture.CROSS_ARMS,
    ArmState.FOLDED_ARMS: Gesture.FOLDED_ARMS,
}


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

        # Sustained-pose latch: ArmState → last time the pose was observed
        # while latched. A pose that fired stays latched (no re-fire) until
        # it goes unobserved for sustained_rearm_s (see StateMachineConfig).
        self._pose_latch: dict[ArmState, float] = {}

        # Last time ANY gesture fired, for the sustained-pose cooldown gate.
        # -inf sentinel means "never fired" — avoids blocking the first tick.
        # There is deliberately no per-gesture clock here: the real
        # per-gesture cooldown lives in core/fusion.py, which is the layer
        # that decides what reaches Home Assistant. A second copy here was
        # written on every fire and read by nothing.
        self._last_fired_any: float = float('-inf')

        # Observed interval between tick() calls — the analyser's real frame
        # cadence, measured rather than configured. See sustain_credit_ticks.
        self._tick_dt: Optional[float] = None
        self._last_tick: Optional[float] = None

        self.total_emitted = 0

    # ── Tick cadence ──────────────────────────────────────────────────────

    # Intervals above this are a stall, a restart or a clock jump rather than
    # a frame rate, and must not drag the estimate up with them.
    _MAX_PLAUSIBLE_TICK_S = 1.0

    def _note_tick(self, now: float) -> None:
        last, self._last_tick = self._last_tick, now
        if last is None:
            return
        dt = now - last
        if not (0.0 < dt <= self._MAX_PLAUSIBLE_TICK_S):
            return
        self._tick_dt = dt if self._tick_dt is None else (
            0.8 * self._tick_dt + 0.2 * dt)

    def _credit_cap(self) -> float:
        """Most that one interval may contribute towards sustain_s.

        Falls back to sustain_gap_s — i.e. to crediting whatever the gap
        logic already forgave, which is the old behaviour — until enough
        ticks have been seen to know the cadence.
        """
        if self._tick_dt is None:
            return self.c.sustain_gap_s
        return min(self.c.sustain_credit_ticks * self._tick_dt,
                   self.c.sustain_gap_s)

    # ── Public API ────────────────────────────────────────────────────────

    def tick(self, reading: Optional[ArmReading], now: float) -> Optional[Gesture]:
        """
        Advance one frame. Returns a gesture to fire, or None.
        `now` is a monotonic timestamp in seconds.
        """
        self._note_tick(now)

        # Re-arm latched sustained poses that have gone unobserved long
        # enough. Must run every tick regardless of state — the pose stops
        # being observed precisely when its _tick_sustained stops running.
        if self._pose_latch:
            self._pose_latch = {
                s: t for s, t in self._pose_latch.items()
                if now - t <= self.c.sustained_rearm_s
            }

        # No pose, or the arm is down → the raise is over.
        #
        # The sustained-pose timers are deliberately NOT cleared here. They
        # used to be, which made sustain_gap_s unreachable: every gap this
        # layer is supposed to forgive arrives as a DOWN frame, and clearing
        # on DOWN wiped the timer before _tick_sustained could weigh the gap.
        # Measured before the change — one DOWN frame in three, a 0.1 s gap
        # against a 0.85 s tolerance — a three-second hold never fired at
        # all. Every bit of flicker tolerance the system actually had came
        # from the tracker's state_release_s, so the protection this layer
        # documents did not exist and lowering that one setting would have
        # silently broken held poses. Staleness is now decided in one place,
        # by comparing against sustain_gap_s where the docstring says it is.
        if reading is None or reading.state == ArmState.DOWN:
            self._reset_raise()
            return None

        state = reading.state

        # Latched sustained pose still being held: refresh the latch clock
        # and stay silent. Checked BEFORE the cooldown gate — the gate
        # returns early, and a latch that isn't refreshed while the pose is
        # merely cooldown-blocked would expire mid-hold and re-fire.
        if state in self._pose_latch:
            self._pose_latch[state] = now
            self._sustain.entered_at.clear()
            self._sustain.last_seen.clear()
            self._sustain.held_s.clear()
            return None

        # Cooldown gate for sustained-state gestures only — these need the
        # cooldown to avoid rapid re-fire while the user holds the pose.
        # SINGLE_UP doesn't need the cooldown because the per-raise flags
        # (_snap_fired, _hold_fired) already prevent repeated firing, and
        # enforcing cooldown here blocks DOUBLE_SNAP from working after SNAP.
        if state != ArmState.SINGLE_UP:
            if now - self._last_fired_any < self.c.cross_gesture_cooldown_s:
                return None

        # Dispatch per state
        if state == ArmState.SINGLE_UP:
            self._sustain.entered_at.clear()  # no sustained state active
            self._sustain.last_seen.clear()
            return self._tick_single_up(reading, now)

        if state in (ArmState.BOTH_UP, ArmState.T_POSE, ArmState.CROSS_ARMS,
                     ArmState.FOLDED_ARMS):
            self._reset_raise()
            return self._tick_sustained(state, now)

        return None

    # ── Handlers ──────────────────────────────────────────────────────────

    def _near_miss(self, gesture: str, reason: str, reading: ArmReading) -> None:
        """Record a blocked gesture. Logged as well as written to feedback,
        because feedback.jsonl has to be exported by hand while the log can be
        read live — and 'the arm was seen but nothing fired' is precisely the
        case where the reason needs to be visible immediately."""
        log.info("Gesture %s blocked: %s", gesture, reason)
        if self._on_near_miss:
            self._on_near_miss(gesture, reason, reading)

    def _tick_single_up(self, reading: ArmReading, now: float) -> Optional[Gesture]:
        r = self._raise
        r.up_frames += 1
        if r.up_frames == 1:
            r.entered_at = now  # record when this raise began

        # Elevation is dimensionless (sine of the arm's angle above the
        # horizon), so no per-person scaling is needed or possible here.
        arm_vertical = reading.elevation >= self.c.snap_elevation_min
        roll_snap = (
            self.c.snap_roll_threshold > 0
            and abs(reading.snap_roll) >= self.c.snap_roll_threshold
        )
        snap_condition = arm_vertical or roll_snap
        # Motion evidence. The flourish test asks "did this arm just sweep up
        # fast?", which is the gesture itself; the legacy path asks the
        # weaker "did it rise, and is it being held?".
        if self.c.snap_require_flourish:
            if (reading.sweep_climb >= self.c.flourish_min_climb
                    and self.c.flourish_min_rate <= reading.sweep_rate
                        <= self.c.flourish_max_rate):
                r.flourish_seen = True
            gate_ok = r.flourish_seen
            rise_ok = still_ok = True
            # The sweep IS the evidence that this was deliberate, which is
            # the whole job snap_sustain_s used to do — so don't charge for
            # it twice. Requiring both also meant the sustain window had to
            # elapse while the sweep still qualified, and when it didn't the
            # gesture fired seconds late or not at all.
            sustain_needed = 0.0
        else:
            rise_ok  = (not self.c.snap_require_rise) or reading.rose_recently
            still_ok = (not self.c.snap_require_still) or reading.wrist_still
            gate_ok = rise_ok and still_ok
            sustain_needed = self.c.snap_sustain_s

        # HOLD: arm still vertical, a raise gesture already EMITTED, enough
        # time passed. See _RaiseState.snap_emitted for why not snap_fired.
        if (r.snap_emitted and not r.hold_fired and snap_condition
                and (now - r.snap_fired_at) >= self.c.hold_duration_s):
            r.hold_fired = True
            return self._fire(Gesture.HOLD, now)

        # SNAP: arm has been held up long enough (time-based, rate-independent)
        if (not r.snap_fired and snap_condition and gate_ok
                and (now - r.entered_at) >= sustain_needed):
            return self._fire_snap(now)

        # Near-miss: arm is up but snap condition not met — log for tuning.
        if not r.snap_fired and r.up_frames > 1:
            if not snap_condition:
                reason = (f"elevation={reading.elevation:.2f} < "
                          f"{self.c.snap_elevation_min:.2f} (min),"
                          f" extension={reading.extension:.2f},"
                          f" snap_roll={reading.snap_roll:.3f}")
                self._near_miss("SNAP", reason, reading)
            elif not gate_ok:
                if self.c.snap_require_flourish:
                    if "no_flourish" not in r.gates_logged:
                        r.gates_logged.add("no_flourish")
                        self._near_miss(
                            "SNAP",
                            f"no_flourish: arm is up but did not sweep — "
                            f"climb={reading.sweep_climb:.2f} "
                            f"(need {self.c.flourish_min_climb:.2f}), "
                            f"rate={reading.sweep_rate:.2f}/s "
                            f"(need {self.c.flourish_min_rate:.2f}"
                            f"-{self.c.flourish_max_rate:.2f})",
                            reading)
                elif not rise_ok:
                    if "no_rise" not in r.gates_logged:
                        r.gates_logged.add("no_rise")
                        self._near_miss(
                            "SNAP",
                            f"no_rise: no lift seen — elevation climbed only "
                            f"{reading.rise_delta:.2f} and the arm was never low",
                            reading)
                elif "wrist_moving" not in r.gates_logged:
                    r.gates_logged.add("wrist_moving")
                    self._near_miss(
                        "SNAP", "wrist_moving: wrist not held still", reading)
            elif (now - r.entered_at) < sustain_needed:
                held = now - r.entered_at
                reason = f"sustain={held:.3f}s < {self.c.snap_sustain_s}s required"
                self._near_miss("SNAP", reason, reading)

        return None

    def _tick_sustained(self, state: ArmState, now: float) -> Optional[Gesture]:
        entered = self._sustain.entered_at.get(state)
        last = self._sustain.last_seen.get(state)
        # A gap longer than sustain_gap_s means the pose was actually
        # released and re-formed, not merely mis-measured for a frame or two.
        # The epsilon is not cosmetic: 4.2 - 3.6 is 0.6000000000000001, so a
        # gap exactly equal to the limit compares as greater and resets a
        # hold that never actually lapsed.
        if entered is None or (last is not None
                               and now - last > self.c.sustain_gap_s + 1e-6):
            self._sustain.entered_at = {state: now}   # reset others
            self._sustain.last_seen = {state: now}
            self._sustain.held_s = {state: 0.0}
            return None

        # Credit the interval since the previous match, capped so a forgiven
        # gap cannot stand in for the hold it interrupted. See _SustainState.
        held = self._sustain.held_s.get(state, 0.0) + min(
            now - last, self._credit_cap())
        self._sustain.last_seen[state] = now
        self._sustain.held_s[state] = held

        if held < self.c.sustain_s:
            return None

        # Held long enough — fire once and latch until the pose is released
        # for sustained_rearm_s (see tick()).
        gesture = POSE_GESTURE[state]
        self._sustain.entered_at.clear()
        self._sustain.last_seen.clear()
        self._sustain.held_s.clear()
        self._pose_latch[state] = now
        return self._fire(gesture, now)

    # ── SNAP + DOUBLE_SNAP logic ──────────────────────────────────────────

    def _fire_snap(self, now: float) -> Optional[Gesture]:
        r = self._raise
        r.snap_fired = True
        r.snap_fired_at = now

        # DOUBLE_SNAP: prior SNAP within window? Only when it is switched on
        # — otherwise a second snap must still register as a plain SNAP
        # rather than vanishing into a disabled gesture.
        self._snap_times[:] = [
            t for t in self._snap_times
            if now - t < self.c.double_snap_window_s
        ]
        if self._snap_times and self.is_enabled(Gesture.DOUBLE_SNAP):
            self._snap_times.clear()
            fired = self._fire(Gesture.DOUBLE_SNAP, now)
        else:
            self._snap_times.append(now)
            fired = self._fire(Gesture.SNAP, now)

        # A DOUBLE_SNAP counts: the arm is up and an event went out, so a
        # HOLD that follows is still extending a gesture the user performed.
        r.snap_emitted = fired is not None
        return fired

    # ── Fire helper ───────────────────────────────────────────────────────

    _ENABLE_ATTR = {
        Gesture.SNAP:        "enable_snap",
        Gesture.HOLD:        "enable_hold",
        Gesture.DOUBLE_SNAP: "enable_double_snap",
        Gesture.CROSS_ARMS:  "enable_cross_arms",
        Gesture.T_POSE:      "enable_t_pose",
        Gesture.RAISE_BOTH:  "enable_raise_both",
        Gesture.FOLDED_ARMS: "enable_folded_arms",
    }

    def is_enabled(self, gesture: Gesture) -> bool:
        return bool(getattr(self.c, self._ENABLE_ATTR[gesture], True))

    def _fire(self, gesture: Gesture, now: float) -> Optional[Gesture]:
        """Emit *gesture*, or None when that gesture is switched off.

        Returning None keeps the per-raise/per-pose bookkeeping of the
        caller intact — a disabled gesture is treated as having happened
        and been discarded, so it neither fires nor retries every frame.
        """
        if not self.is_enabled(gesture):
            log.debug("Gesture %s detected but disabled in settings", gesture)
            return None
        self._last_fired_any = now
        self.total_emitted += 1
        log.info("Gesture fired: %s", gesture)
        return gesture

    def _reset_raise(self) -> None:
        self._raise = _RaiseState()
