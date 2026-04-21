"""
tests/test_state_machine.py

Unit tests for GestureStateMachine. Runs without MediaPipe.

Run with: python -m pytest tests/test_state_machine.py -v
Or:       python tests/test_state_machine.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmReading, ArmState, Side
from core.state_machine import GestureStateMachine, StateMachineConfig, Gesture


def _machine(**overrides) -> GestureStateMachine:
    cfg = StateMachineConfig(**overrides)
    return GestureStateMachine(cfg)


def _down() -> ArmReading:
    return ArmReading(state=ArmState.DOWN)


def _vertical_up() -> ArmReading:
    return ArmReading(
        state=ArmState.SINGLE_UP,
        raised_side=Side.RIGHT,
        wrist_x=800, wrist_y=100,
        forearm_dy=0.15,
    )


def _horizontal_up() -> ArmReading:
    return ArmReading(
        state=ArmState.SINGLE_UP,
        raised_side=Side.RIGHT,
        wrist_x=1000, wrist_y=300,
        forearm_dy=0.02,
    )


def _state(s: ArmState) -> ArmReading:
    return ArmReading(state=s)


# ── SNAP ──────────────────────────────────────────────────────────────────────

def test_snap_fires_after_snap_sustain_frames():
    m = _machine(snap_sustain_frames=2)
    assert m.tick(_vertical_up(), now=0.0) is None     # frame 1
    assert m.tick(_vertical_up(), now=0.1) == Gesture.SNAP   # frame 2


def test_snap_does_not_fire_for_non_vertical_arm():
    m = _machine(snap_sustain_frames=2)
    for i in range(10):
        assert m.tick(_horizontal_up(), now=i * 0.1) is None


def test_snap_resets_on_arm_down():
    m = _machine(snap_sustain_frames=2)
    m.tick(_vertical_up(), now=0.0)  # frame 1 of 2
    m.tick(_down(),        now=0.1)  # reset
    assert m.tick(_vertical_up(), now=0.2) is None  # must start over


# ── HOLD ──────────────────────────────────────────────────────────────────────

def test_hold_fires_after_snap_then_arm_stays_up():
    m = _machine(snap_sustain_frames=2, hold_duration_s=1.0, cooldown_s=0.1)
    # Fire snap at t=0.1
    m.tick(_vertical_up(), now=0.0)
    assert m.tick(_vertical_up(), now=0.1) == Gesture.SNAP
    # Arm stays up. Cooldown is 0.1s, HOLD requires 1.0s past snap.
    for t in [0.3, 0.5, 0.8]:
        assert m.tick(_vertical_up(), now=t) is None
    # 1.0s past snap → HOLD fires
    assert m.tick(_vertical_up(), now=1.2) == Gesture.HOLD


def test_hold_does_not_fire_twice_per_raise():
    m = _machine(snap_sustain_frames=2, hold_duration_s=1.0, cooldown_s=0.1)
    m.tick(_vertical_up(), now=0.0)
    m.tick(_vertical_up(), now=0.1)  # SNAP
    m.tick(_vertical_up(), now=1.2)  # HOLD
    # Keep arm up, advance a long time — should NOT fire HOLD again
    for t in [1.5, 2.0, 3.0, 5.0]:
        assert m.tick(_vertical_up(), now=t) is None


# ── DOUBLE_SNAP ───────────────────────────────────────────────────────────────

def test_double_snap_fires_on_second_raise_within_window():
    m = _machine(snap_sustain_frames=1, cooldown_s=0.1, double_snap_window_s=3.0)
    # First snap
    assert m.tick(_vertical_up(), now=0.0) == Gesture.SNAP
    # Lower arm
    m.tick(_down(), now=0.5)
    # Raise again within 3s
    assert m.tick(_vertical_up(), now=2.0) == Gesture.DOUBLE_SNAP


def test_double_snap_does_not_fire_outside_window():
    m = _machine(snap_sustain_frames=1, cooldown_s=0.1, double_snap_window_s=2.0)
    assert m.tick(_vertical_up(), now=0.0) == Gesture.SNAP
    m.tick(_down(), now=0.5)
    # 3 seconds later > 2s window → just another SNAP
    assert m.tick(_vertical_up(), now=3.0) == Gesture.SNAP


# ── Sustained gestures ───────────────────────────────────────────────────────

def test_raise_both_needs_sustain():
    m = _machine(sustain_s=0.5, cooldown_s=0.1)
    assert m.tick(_state(ArmState.BOTH_UP), now=0.0) is None
    assert m.tick(_state(ArmState.BOTH_UP), now=0.3) is None  # not long enough
    assert m.tick(_state(ArmState.BOTH_UP), now=0.6) == Gesture.RAISE_BOTH


def test_t_pose_needs_sustain():
    m = _machine(sustain_s=0.5, cooldown_s=0.1)
    assert m.tick(_state(ArmState.T_POSE), now=0.0) is None
    assert m.tick(_state(ArmState.T_POSE), now=0.6) == Gesture.T_POSE


def test_cross_arms_needs_sustain():
    m = _machine(sustain_s=0.5, cooldown_s=0.1)
    assert m.tick(_state(ArmState.CROSS_ARMS), now=0.0) is None
    assert m.tick(_state(ArmState.CROSS_ARMS), now=0.6) == Gesture.CROSS_ARMS


def test_brief_t_pose_while_raising_both_does_not_fire():
    m = _machine(sustain_s=0.5, cooldown_s=0.1)
    # Briefly look like T_POSE as user raises arms through horizontal
    m.tick(_state(ArmState.T_POSE), now=0.0)
    m.tick(_state(ArmState.T_POSE), now=0.1)
    # Then switch to BOTH_UP
    m.tick(_state(ArmState.BOTH_UP), now=0.2)
    # Sustain timer resets — neither should fire yet
    assert m.tick(_state(ArmState.BOTH_UP), now=0.4) is None
    # BOTH_UP held long enough
    assert m.tick(_state(ArmState.BOTH_UP), now=0.8) == Gesture.RAISE_BOTH


# ── Cooldown ─────────────────────────────────────────────────────────────────

def test_cooldown_blocks_sustained_refire():
    """Cooldown prevents rapid re-fire of sustained gestures like RAISE_BOTH."""
    m = _machine(sustain_s=0.5, cooldown_s=2.0)
    assert m.tick(_state(ArmState.BOTH_UP), now=0.0) is None
    assert m.tick(_state(ArmState.BOTH_UP), now=0.6) == Gesture.RAISE_BOTH
    # Immediately after, cooldown blocks re-entry
    m.tick(_down(), now=0.7)
    assert m.tick(_state(ArmState.BOTH_UP), now=0.9) is None
    assert m.tick(_state(ArmState.BOTH_UP), now=1.5) is None


def test_cooldown_does_not_block_double_snap():
    """SINGLE_UP must not be cooldown-blocked — otherwise DOUBLE_SNAP can't fire."""
    m = _machine(snap_sustain_frames=1, cooldown_s=2.0, double_snap_window_s=3.0)
    assert m.tick(_vertical_up(), now=0.0) == Gesture.SNAP
    m.tick(_down(), now=0.5)
    # Only 1.0s after first snap — cooldown is 2s but SNAP bypasses it
    assert m.tick(_vertical_up(), now=1.0) == Gesture.DOUBLE_SNAP


# ── Missing/None reading ─────────────────────────────────────────────────────

def test_none_reading_returns_none():
    m = _machine()
    assert m.tick(None, now=0.0) is None


if __name__ == "__main__":
    import traceback
    failed = 0
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'PASSED' if failed == 0 else f'{failed} FAILED'} ({len(tests) - failed}/{len(tests)})")
    sys.exit(failed)
