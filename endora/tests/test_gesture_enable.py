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


# ── HOLD extends an EMITTED raise, not a detected one ─────────────────────────

def test_hold_does_not_fire_when_snap_is_disabled():
    """Switching SNAP off used to leave HOLD firing on its own.

    _fire_snap marks the raise as snapped whether or not the gesture is
    enabled — it has to, or a disabled gesture would be re-detected every
    frame. HOLD then saw "a snap happened" and fired 1.5 s later, so turning
    SNAP off converted the raise into a HOLD event rather than silencing it.
    """
    m = _m(enable_snap=False, enable_hold=True, hold_duration_s=1.0)
    fired = [m.tick(_up(), t / 10.0) for t in range(40)]
    assert not any(fired), [g.name for g in fired if g]


def test_hold_still_follows_a_snap_that_did_fire():
    m = _m(hold_duration_s=1.0)
    assert m.tick(_up(), 0.0) is Gesture.SNAP
    fired = [m.tick(_up(), t / 10.0) for t in range(1, 30)]
    assert Gesture.HOLD in fired


def test_hold_follows_a_double_snap_too():
    """A DOUBLE_SNAP is still an emitted raise gesture, and the arm is still
    up — so holding it is still a HOLD."""
    m = _m(cooldown_s=0.0, double_snap_window_s=3.0, hold_duration_s=1.0)
    assert m.tick(_up(), 0.0) is Gesture.SNAP
    m.tick(_pose(ArmState.DOWN), 0.5)
    assert m.tick(_up(), 1.0) is Gesture.DOUBLE_SNAP
    fired = [m.tick(_up(), 1.0 + t / 10.0) for t in range(1, 30)]
    assert Gesture.HOLD in fired


def test_a_disabled_snap_is_still_consumed_once():
    """The other half of the contract: HOLD must not fire, but the snap must
    also not be re-detected on every subsequent frame."""
    m = _m(enable_snap=False, enable_hold=False)
    for t in range(40):
        m.tick(_up(), t / 10.0)
    assert m.total_emitted == 0


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


# ── Disabling a pose must improve RAISE recognition, not just silence it ──────

def _shadowing_pose():
    """A frame whose geometry satisfies CROSS_ARMS while an arm is also
    raised. Two-handed poses are tested first and return early, so this
    frame is consumed as CROSS_ARMS and the raise is never even considered.
    """
    from tests.fake_landmarks import Landmarks, Point
    from tests.fake_landmarks import (
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE)
    return Landmarks({
        NOSE:           Point(0.50, 0.30),
        LEFT_SHOULDER:  Point(0.40, 0.40), RIGHT_SHOULDER: Point(0.60, 0.40),
        # wrists crossed at chest height -> CROSS_ARMS geometry
        LEFT_ELBOW:     Point(0.50, 0.52), LEFT_WRIST:  Point(0.62, 0.50),
        RIGHT_ELBOW:    Point(0.50, 0.52), RIGHT_WRIST: Point(0.38, 0.50),
        LEFT_HIP:       Point(0.42, 0.65), RIGHT_HIP:   Point(0.58, 0.65),
        LEFT_KNEE:      Point(0.42, 0.80), RIGHT_KNEE:  Point(0.58, 0.80),
    })


def test_a_pose_shadows_the_frame_until_it_is_switched_off():
    from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
    lm = _shadowing_pose()

    on = ArmTracker(ArmTrackerConfig())._classify_raw(lm, 1280, 640)
    assert on.state is ArmState.CROSS_ARMS, f"fixture should shadow, got {on.state}"

    off = ArmTracker(ArmTrackerConfig(detect_cross_arms=False))._classify_raw(
        lm, 1280, 640)
    assert off.state is not ArmState.CROSS_ARMS, \
        "a switched-off pose must not consume the frame"


def test_switching_a_pose_off_leaves_the_raise_intact():
    """Turning off unused poses must not disturb the gesture you do use."""
    from cameras.arm_tracker import ArmTracker, ArmTrackerConfig
    from tests.fake_landmarks import right_arm_up_vertical
    cfg = ArmTrackerConfig(detect_cross_arms=False, detect_t_pose=False,
                           detect_both_up=False)
    r = ArmTracker(cfg)._classify_raw(right_arm_up_vertical(), 1280, 720)
    assert r.state is ArmState.SINGLE_UP and r.raised_side is Side.RIGHT
