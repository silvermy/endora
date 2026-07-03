"""
tests/test_arm_tracker.py

Unit tests for ArmTracker. Runs without MediaPipe — uses fake landmarks.

Run with: python -m pytest tests/test_arm_tracker.py -v
Or:       python tests/test_arm_tracker.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState, Side
from tests.fake_landmarks import (
    arm_down, right_arm_up_vertical, right_arm_up_horizontal,
    both_arms_up, t_pose, cross_arms, lying_down, low_visibility,
)


def _tracker() -> ArmTracker:
    return ArmTracker(ArmTrackerConfig())


def test_arm_down_returns_down_state():
    r = _tracker()._classify_raw(arm_down(), 1280, 720)
    assert r.state == ArmState.DOWN


def test_right_arm_vertical_is_single_up_right():
    r = _tracker()._classify_raw(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.raised_side == Side.RIGHT
    assert r.forearm_dy > 0.10, f"forearm_dy should be vertical, got {r.forearm_dy}"


def test_right_arm_horizontal_is_not_single_up():
    # Wrist at same height as shoulder → not raised above head
    r = _tracker()._classify_raw(right_arm_up_horizontal(), 1280, 720)
    assert r.state == ArmState.DOWN


def test_both_arms_up_is_both_up():
    r = _tracker()._classify_raw(both_arms_up(), 1280, 720)
    assert r.state == ArmState.BOTH_UP


def test_t_pose_is_t_pose():
    r = _tracker()._classify_raw(t_pose(), 1280, 720)
    assert r.state == ArmState.T_POSE


def test_cross_arms_is_cross_arms():
    r = _tracker()._classify_raw(cross_arms(), 1280, 720)
    assert r.state == ArmState.CROSS_ARMS


def test_lying_down_casual_arm_rejected():
    # Reclined body + arm only slightly above shoulder → DOWN (not high enough)
    from tests.fake_landmarks import _build, Point
    lm = _build(
        # Hips well above shoulders (lying on back, feet toward camera)
        left_hip=Point(0.42, 0.20), right_hip=Point(0.58, 0.20),
        # Wrist at y=0.28, shoulder at y=0.40 → margin 0.12 < reclined threshold 0.28
        right_elbow=Point(0.65, 0.33), right_wrist=Point(0.65, 0.28),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.DOWN
    assert r.upright is False


def test_lying_down_straight_arm_accepted():
    # Reclined body + arm pointing straight up → SINGLE_UP fires
    from tests.fake_landmarks import _build, Point
    lm = _build(
        # Hips well above shoulders (lying on back)
        left_hip=Point(0.42, 0.20), right_hip=Point(0.58, 0.20),
        # Wrist at y=0.08, shoulder at y=0.40 → margin 0.32 > reclined threshold 0.30
        right_elbow=Point(0.65, 0.22), right_wrist=Point(0.65, 0.08),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.upright is False


def test_leg_raise_suppresses_gesture():
    # Knees above shoulders (legs-in-V while lying down) → DOWN even if arm raised
    from tests.fake_landmarks import _build, Point
    lm = _build(
        right_elbow=Point(0.65, 0.22), right_wrist=Point(0.65, 0.08),  # arm up
        # Knees raised well above shoulders (y=0.40) — upside-down V
        left_knee=Point(0.42, 0.20), right_knee=Point(0.58, 0.20),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.DOWN


def test_low_visibility_returns_none():
    r = _tracker()._classify_raw(low_visibility(), 1280, 720)
    assert r is None


def test_hands_on_chest_is_cross_arms():
    """Hands clasped on chest (realistic cross-arms pose) should fire."""
    from tests.fake_landmarks import _build, Point
    # Midline at ~0.50, shoulders at y=0.40, hips at y=0.65.
    # Wrists cross 0.10 past midline each side (above min_crossing=0.08).
    lm = _build(
        left_wrist=Point(0.60, 0.50),   # 0.10 past midline to the right
        right_wrist=Point(0.40, 0.50),  # 0.10 past midline to the left
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.CROSS_ARMS, f"got {r.state}"


def test_hands_in_lap_is_not_cross_arms():
    # Seated, hands resting in lap (around hip level, near midline)
    from tests.fake_landmarks import _build, Point
    lm = _build(
        left_wrist=Point(0.45, 0.70),   # near midline, at hip height
        right_wrist=Point(0.55, 0.70),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state != ArmState.CROSS_ARMS, f"got {r.state}"


def test_hands_uncrossed_at_chest_is_not_cross_arms():
    # Both hands on same side — not crossed
    from tests.fake_landmarks import _build, Point
    lm = _build(
        left_wrist=Point(0.35, 0.45),
        right_wrist=Point(0.42, 0.45),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state != ArmState.CROSS_ARMS, f"got {r.state}"


def test_none_landmarks_returns_none():
    r = _tracker()._classify_raw(None, 1280, 720)
    assert r is None


def test_both_up_prefers_both_over_single():
    # Ambiguity: both arms raised — must return BOTH_UP, not SINGLE_UP
    r = _tracker()._classify_raw(both_arms_up(), 1280, 720)
    assert r.state == ArmState.BOTH_UP


def test_t_pose_prefers_t_over_both():
    # T-pose wrists are at shoulder height, not above → not BOTH_UP
    r = _tracker()._classify_raw(t_pose(), 1280, 720)
    assert r.state == ArmState.T_POSE


def test_one_shoulder_occluded_still_detects_raise():
    # Side-on on the couch: left shoulder hidden by a cushion (low vis), right
    # arm raised.  Old averaging gate rejected this; per-side logic should fire.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        left_shoulder=Point(0.40, 0.40, visibility=0.1),
        right_elbow=Point(0.65, 0.25), right_wrist=Point(0.65, 0.10),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.raised_side == Side.RIGHT


def test_forearm_vertical_route_fires_when_wrist_near_shoulder():
    # Camera angle: raised arm whose wrist barely clears the shoulder (0.02),
    # well under arm_above_head_tolerance, but forearm is clearly vertical.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        right_elbow=Point(0.60, 0.55),  # elbow low → forearm_dy = 0.17
        right_wrist=Point(0.60, 0.38),  # wrist just above shoulder (y=0.40)
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.SINGLE_UP, f"got {r.state}"
    assert r.raised_side == Side.RIGHT


def test_low_visibility_wrist_does_not_create_false_raise():
    # A garbage/occluded wrist keypoint placed high must not trigger a raise.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        right_elbow=Point(0.65, 0.25),
        right_wrist=Point(0.65, 0.10, visibility=0.1),  # high but not visible
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.DOWN


def test_t_pose_needs_both_shoulders_visible():
    # With one shoulder occluded we can't confirm a two-handed pose — must not
    # misfire T_POSE (falls through to per-side SINGLE_UP, which also fails here).
    from tests.fake_landmarks import _build, Point
    lm = _build(
        left_shoulder=Point(0.40, 0.40, visibility=0.1),
        left_elbow=Point(0.25, 0.40),   left_wrist=Point(0.10, 0.40),
        right_elbow=Point(0.75, 0.40),  right_wrist=Point(0.90, 0.40),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state != ArmState.T_POSE


def test_hand_resting_near_face_does_not_fire_single_up():
    # Realistic false positive: seated, resting/adjusting a hand right next to
    # the face (glasses, phone, scratching, chin-on-hand). Wrist clears the
    # shoulder margin the same way a deliberate raise would, but sits right on
    # top of the nose keypoint — a real raise holds the hand up and away from
    # the head.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        # Nose default is (0.50, 0.30); this wrist is ~0.08 away — inside the
        # default wrist_head_exclude_dist of 0.09.
        right_elbow=Point(0.58, 0.35),
        right_wrist=Point(0.52, 0.22),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.DOWN, f"got {r.state}"


def test_hand_resting_near_face_fires_when_nose_not_visible():
    # Same near-face wrist position, but the face is occluded (nose not
    # confidently visible) — must not block a genuine raise just because we
    # can't confirm proximity to the head.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        nose=Point(0.50, 0.30, visibility=0.1),
        right_elbow=Point(0.58, 0.35),
        right_wrist=Point(0.52, 0.22),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.SINGLE_UP, f"got {r.state}"


def test_raise_well_away_from_face_still_fires():
    # Sanity check the exclusion is distance-gated, not a blanket rejection
    # near the top of the frame — a real raise well clear of the head fires.
    r = _tracker()._classify_raw(right_arm_up_vertical(), 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.raised_side == Side.RIGHT


def test_hips_hidden_forearm_up_fires():
    # Blanket to chest (hips not visible → upright unknown). Arm raised with a
    # visible, vertical forearm should fire via the lenient+forearm-up route.
    from tests.fake_landmarks import _build, Point
    lm = _build(
        left_hip=Point(0.42, 0.65, visibility=0.1),
        right_hip=Point(0.58, 0.65, visibility=0.1),
        right_elbow=Point(0.65, 0.25), right_wrist=Point(0.65, 0.10),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.state == ArmState.SINGLE_UP
    assert r.upright is False  # bool(None) → False; hips were unknown


if __name__ == "__main__":
    # Minimal runner without pytest
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
