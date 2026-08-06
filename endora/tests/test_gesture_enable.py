"""
tests/test_gesture_enable.py

Per-gesture switches. A gesture you never perform is not free: it fires HA
events you did not ask for, and — before the cooldown split in v1.9.130 —
an unwanted one could suppress a real gesture entirely.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmReading, ArmState, Side
from core.state_machine import Gesture, GestureStateMachine, StateMachineConfig


def _m(**over):
    cfg = dict(snap_sustain_s=0.0, sustain_s=0.5, cooldown_s=0.1)
    cfg.update(over)
    return GestureStateMachine(StateMachineConfig(**cfg))


def _up():
    return ArmReading(state=ArmState.SINGLE_UP, raised_side=Side.RIGHT,
                      elevation=0.95, extension=0.95,
                      sweep_climb=1.6, sweep_rate=2.4)


def _pose(s):
    return ArmReading(state=s)


def test_disabled_cross_arms_never_fires():
    m = _m(enable_cross_arms=False)
    fired = [m.tick(_pose(ArmState.CROSS_ARMS), t / 10.0) for t in range(60)]
    assert not any(fired)


def test_disabling_one_gesture_leaves_the_others_alone():
    m = _m(enable_cross_arms=False, enable_t_pose=False, enable_raise_both=False)
    assert m.tick(_up(), 0.0) is Gesture.SNAP


def test_disabled_gesture_does_not_retry_every_frame():
    """A disabled gesture is consumed, not endlessly re-attempted — otherwise
    it would spam the near-miss log and the pose bookkeeping."""
    m = _m(enable_cross_arms=False, sustained_rearm_s=2.0)
    for t in range(60):
        m.tick(_pose(ArmState.CROSS_ARMS), t / 10.0)
    assert m.total_emitted == 0


def test_second_snap_still_fires_when_double_snap_is_off():
    """With DOUBLE_SNAP disabled a second snap must register as a plain
    SNAP, not disappear into the disabled gesture."""
    m = _m(enable_double_snap=False, cooldown_s=0.0, double_snap_window_s=3.0)
    assert m.tick(_up(), 0.0) is Gesture.SNAP
    m.tick(_pose(ArmState.DOWN), 0.5)
    assert m.tick(_up(), 1.0) is Gesture.SNAP


def test_double_snap_still_works_when_enabled():
    m = _m(cooldown_s=0.0, double_snap_window_s=3.0)
    assert m.tick(_up(), 0.0) is Gesture.SNAP
    m.tick(_pose(ArmState.DOWN), 0.5)
    assert m.tick(_up(), 1.0) is Gesture.DOUBLE_SNAP


if __name__ == "__main__":
    import traceback
    failed = 0
    for fn in [v for k, v in dict(globals()).items()
               if k.startswith("test_") and callable(v)]:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception:
            failed += 1; print(f"  ERROR {fn.__name__}"); traceback.print_exc()
    sys.exit(failed)
