"""
tests/test_hysteresis.py

Tests for ArmTracker's time-based hysteresis layer — the public classify() method.
Ensures phantom detections are filtered and mid-gesture dropouts don't
cause spurious state changes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState
from tests.fake_landmarks import arm_down, right_arm_up_vertical

CONFIRM_S = 0.20
RELEASE_S = 0.30


def _tracker(**overrides) -> ArmTracker:
    return ArmTracker(ArmTrackerConfig(**overrides))


def test_first_frame_not_yet_confirmed():
    """A single frame of SINGLE_UP at t=0 must not yet be confirmed."""
    t = _tracker(state_confirm_s=CONFIRM_S)
    r = t.classify(right_arm_up_vertical(), 1280, 720, now=0.0)
    assert r.state == ArmState.DOWN, f"expected DOWN on first frame, got {r.state}"


def test_state_confirmed_after_confirm_s():
    """After confirm_s seconds of continuous SINGLE_UP, state is accepted."""
    t = _tracker(state_confirm_s=CONFIRM_S)
    t.classify(right_arm_up_vertical(), 1280, 720, now=0.0)
    r = t.classify(right_arm_up_vertical(), 1280, 720, now=CONFIRM_S + 0.05)
    assert r.state == ArmState.SINGLE_UP, f"should be confirmed by now, got {r.state}"


def test_state_not_confirmed_just_before_confirm_s():
    """Just before confirm_s elapses the state is still pending."""
    t = _tracker(state_confirm_s=CONFIRM_S)
    t.classify(right_arm_up_vertical(), 1280, 720, now=0.0)
    r = t.classify(right_arm_up_vertical(), 1280, 720, now=CONFIRM_S - 0.05)
    assert r.state == ArmState.DOWN, f"should still be pending, got {r.state}"


def test_flicker_resets_pending_timer():
    """Alternating SINGLE_UP/DOWN faster than confirm_s never confirms."""
    t = _tracker(state_confirm_s=CONFIRM_S)
    for i in range(8):
        # Alternate every 10 ms — the pending timer resets on each direction change
        t.classify(right_arm_up_vertical(), 1280, 720, now=i * 0.02)
        r = t.classify(arm_down(), 1280, 720, now=i * 0.02 + 0.01)
        assert r.state == ArmState.DOWN


def test_mid_gesture_brief_dropout_does_not_release():
    """A single short dropout (< release_s) must not flip back to DOWN."""
    t = _tracker(state_confirm_s=CONFIRM_S, state_release_s=RELEASE_S)
    # Confirm SINGLE_UP
    t.classify(right_arm_up_vertical(), 1280, 720, now=0.0)
    t.classify(right_arm_up_vertical(), 1280, 720, now=CONFIRM_S + 0.05)
    assert t.classify(right_arm_up_vertical(), 1280, 720,
                      now=CONFIRM_S + 0.10).state == ArmState.SINGLE_UP
    # Brief dropout (less than release_s)
    r = t.classify(arm_down(), 1280, 720, now=CONFIRM_S + 0.15)
    assert r.state == ArmState.SINGLE_UP, \
        f"short dropout should be ignored, got {r.state}"


def test_sustained_arm_down_releases_state():
    """After release_s seconds of DOWN, stable SINGLE_UP is released."""
    t = _tracker(state_confirm_s=CONFIRM_S, state_release_s=RELEASE_S)
    # Confirm SINGLE_UP
    t0 = 0.0
    t.classify(right_arm_up_vertical(), 1280, 720, now=t0)
    t.classify(right_arm_up_vertical(), 1280, 720, now=t0 + CONFIRM_S + 0.05)
    # Drop arm and hold past release_s
    drop_t = t0 + CONFIRM_S + 0.10
    t.classify(arm_down(), 1280, 720, now=drop_t)
    r = t.classify(arm_down(), 1280, 720, now=drop_t + RELEASE_S + 0.05)
    assert r.state == ArmState.DOWN, f"should have released to DOWN, got {r.state}"


def test_immediate_down_never_needs_confirmation():
    """DOWN state is accepted immediately without waiting for confirm_s."""
    t = _tracker(state_confirm_s=CONFIRM_S)
    r = t.classify(arm_down(), 1280, 720, now=0.0)
    # After the very first call stable_reading is set; DOWN needs no confirm
    assert r is not None


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
