"""
tests/test_body_scale.py

Body-scale normalization: every geometric threshold is tuned at
ArmTrackerConfig.body_scale_reference (torso length as a frame fraction) and
scales with each person's detected size. A small/distant reclined body must
not be asked to clear margins sized for a full-frame standing one, and a
large/close body must not trigger on casual movements that only look big
because the body fills the frame.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState, Side
from tests.fake_landmarks import _build, Point, Landmarks
from tests.fake_landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
)


def _tracker() -> ArmTracker:
    return ArmTracker(ArmTrackerConfig())


def test_default_fixture_is_scale_neutral():
    # The standard fixture torso (shoulders y=0.40, hips y=0.65) spans 0.25 —
    # exactly body_scale_reference — so existing threshold semantics are
    # unchanged for it.
    from tests.fake_landmarks import arm_down
    r = _tracker()._classify_raw(arm_down(), 1280, 720)
    assert abs(r.scale_factor - 1.0) < 0.01, f"got {r.scale_factor}"


def test_small_reclined_person_raise_accepted_at_scaled_margin():
    # Half-reference body (torso 0.125 → factor 0.5) lying horizontally.
    # The wrist clears the shoulder by 0.20 — under the unscaled reclined
    # margin of 0.38, but over the scaled one (0.19). Before body scaling
    # this raise was physically near-impossible for a small/distant body.
    lm = Landmarks({
        NOSE:           Point(0.35, 0.52),
        LEFT_SHOULDER:  Point(0.40, 0.49),
        RIGHT_SHOULDER: Point(0.44, 0.51),
        LEFT_HIP:       Point(0.525, 0.49),
        RIGHT_HIP:      Point(0.565, 0.51),
        LEFT_KNEE:      Point(0.62, 0.55),
        RIGHT_KNEE:     Point(0.64, 0.55),
        LEFT_ELBOW:     Point(0.46, 0.55),
        LEFT_WRIST:     Point(0.50, 0.58),
        RIGHT_ELBOW:    Point(0.42, 0.40),
        RIGHT_WRIST:    Point(0.42, 0.30),   # 0.20 above shoulder level
    })
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.upright is False
    assert abs(r.scale_factor - 0.5) < 0.05, f"got {r.scale_factor}"
    assert r.state == ArmState.SINGLE_UP, f"got {r.state}"
    assert r.raised_side == Side.RIGHT


def test_large_person_casual_arm_not_raised():
    # Double-reference body (torso 0.5 → factor 2.0), upright. Wrist clears
    # the shoulder by 0.20 — over the unscaled margin of 0.15, but under the
    # scaled one (0.30): for a body this large in frame, 0.20 is a casual
    # hand movement, not a raise. Forearm is deliberately not vertical
    # enough for the (also scaled) secondary route.
    lm = Landmarks({
        NOSE:           Point(0.50, 0.10),
        LEFT_SHOULDER:  Point(0.30, 0.25),
        RIGHT_SHOULDER: Point(0.70, 0.25),
        LEFT_HIP:       Point(0.35, 0.75),
        RIGHT_HIP:      Point(0.65, 0.75),
        LEFT_KNEE:      Point(0.35, 0.95),
        RIGHT_KNEE:     Point(0.65, 0.95),
        LEFT_ELBOW:     Point(0.25, 0.45),
        LEFT_WRIST:     Point(0.22, 0.60),
        RIGHT_ELBOW:    Point(0.75, 0.20),
        RIGHT_WRIST:    Point(0.78, 0.05),   # 0.20 above shoulder; dy=0.15 < 0.20
    })
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert abs(r.scale_factor - 2.0) < 0.05, f"got {r.scale_factor}"
    assert r.state == ArmState.DOWN, f"got {r.state}"


def test_shoulder_width_fallback_when_hips_hidden():
    # Blanket scenario: hips invisible → torso estimated from shoulder width.
    # Shoulder width 0.40 → torso estimate 0.50 → factor 2.0.
    lm = _build(
        left_shoulder=Point(0.30, 0.40),
        right_shoulder=Point(0.70, 0.40),
        left_hip=Point(0.42, 0.65, visibility=0.1),
        right_hip=Point(0.58, 0.65, visibility=0.1),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert abs(r.scale_factor - 2.0) < 0.05, f"got {r.scale_factor}"


def test_foreshortened_reclined_body_uses_shoulder_width():
    # Reclining feet-toward-the-camera foreshortens the torso to a sliver in
    # image space while the shoulders stay lateral. The size estimate must
    # take the LARGER of torso and shoulder-width estimates, or every margin
    # collapses and a resting arm fires all day (live data: scale 0.5–0.65
    # for a normal-sized person on the couch).
    lm = Landmarks({
        NOSE:           Point(0.46, 0.36),
        LEFT_SHOULDER:  Point(0.30, 0.45),
        RIGHT_SHOULDER: Point(0.62, 0.45),   # width 0.32 → estimate 0.40
        LEFT_HIP:       Point(0.44, 0.52),
        RIGHT_HIP:      Point(0.52, 0.52),   # torso only ~0.07 (foreshortened)
        LEFT_KNEE:      Point(0.44, 0.60),
        RIGHT_KNEE:     Point(0.52, 0.60),
        LEFT_ELBOW:     Point(0.26, 0.60),
        LEFT_WRIST:     Point(0.24, 0.66),
        RIGHT_ELBOW:    Point(0.66, 0.60),
        RIGHT_WRIST:    Point(0.68, 0.66),
    })
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.scale_factor > 1.0, \
        f"foreshortened torso must not shrink the scale, got {r.scale_factor}"


def test_scale_factor_clamped_on_degenerate_shoulders():
    # Side-on: shoulders nearly coincide and hips are hidden — the size
    # estimate collapses, but the factor must clamp at 0.5, not zero the
    # margins out.
    lm = _build(
        left_shoulder=Point(0.49, 0.40),
        right_shoulder=Point(0.51, 0.40),
        left_hip=Point(0.42, 0.65, visibility=0.1),
        right_hip=Point(0.58, 0.65, visibility=0.1),
    )
    r = _tracker()._classify_raw(lm, 1280, 720)
    assert r.scale_factor == 0.5, f"got {r.scale_factor}"


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
        except Exception:
            failed += 1
            print(f"  ERROR {t.__name__}")
            traceback.print_exc()
    sys.exit(failed)
