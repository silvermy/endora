"""
tests/test_invariance.py

The properties the geometry rewrite exists to guarantee. Each of these was a
real, logged failure under the old frame-fraction thresholds:

  * the same gesture must classify the same whether the person is close to
    the camera or across the room (previously needed body_scale_reference,
    which had to be re-calibrated by hand every time the camera moved);
  * it must classify the same on a non-square frame — the dewarp emits
    1280x640, and landmarks are normalised per-axis, so any measurement
    taken directly in normalised coordinates silently mixes two units;
  * it must classify the same standing, sitting or lying down (previously
    needed a separate, much stricter reclined threshold).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState, Side
from tests.fake_landmarks import Landmarks, Point
from tests.fake_landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
)


def _tracker() -> ArmTracker:
    return ArmTracker(ArmTrackerConfig())


def _person(cx: float, cy: float, size: float, raised: bool) -> Landmarks:
    """An upright person centred at (cx, cy) whose torso spans *size*, with
    the right arm either hanging down or extended straight up.

    Coordinates are normalised; callers pick the frame dimensions, so the
    same body can be rendered at any apparent size or aspect ratio.
    """
    sh_y = cy - size * 0.5
    hip_y = cy + size * 0.5
    half_w = size * 0.22
    if raised:
        r_elbow = Point(cx + half_w, sh_y - size * 0.42)
        r_wrist = Point(cx + half_w, sh_y - size * 0.85)
    else:
        r_elbow = Point(cx + half_w * 1.1, sh_y + size * 0.42)
        r_wrist = Point(cx + half_w * 1.2, sh_y + size * 0.85)
    return Landmarks({
        NOSE:           Point(cx, sh_y - size * 0.28),
        LEFT_SHOULDER:  Point(cx - half_w, sh_y),
        RIGHT_SHOULDER: Point(cx + half_w, sh_y),
        LEFT_ELBOW:     Point(cx - half_w * 1.1, sh_y + size * 0.42),
        LEFT_WRIST:     Point(cx - half_w * 1.2, sh_y + size * 0.85),
        RIGHT_ELBOW:    r_elbow,
        RIGHT_WRIST:    r_wrist,
        LEFT_HIP:       Point(cx - half_w * 0.8, hip_y),
        RIGHT_HIP:      Point(cx + half_w * 0.8, hip_y),
        LEFT_KNEE:      Point(cx - half_w * 0.8, hip_y + size * 0.8),
        RIGHT_KNEE:     Point(cx + half_w * 0.8, hip_y + size * 0.8),
    })


def _reclined(cx: float, cy: float, size: float, raised: bool) -> Landmarks:
    """A person lying with their body axis horizontal in the frame (head to
    the left), arm either resting along the body or raised toward the
    ceiling — which the camera sees as up the frame.
    """
    sh_x = cx - size * 0.5
    hip_x = cx + size * 0.5
    half = size * 0.22
    if raised:
        r_elbow = Point(sh_x + size * 0.05, cy - size * 0.42)
        r_wrist = Point(sh_x + size * 0.05, cy - size * 0.85)
    else:
        r_elbow = Point(sh_x + size * 0.42, cy + half * 0.6)
        r_wrist = Point(sh_x + size * 0.85, cy + half * 0.7)
    return Landmarks({
        NOSE:           Point(sh_x - size * 0.28, cy),
        LEFT_SHOULDER:  Point(sh_x, cy - half),
        RIGHT_SHOULDER: Point(sh_x, cy + half),
        LEFT_ELBOW:     Point(sh_x + size * 0.42, cy - half * 0.6),
        LEFT_WRIST:     Point(sh_x + size * 0.85, cy - half * 0.7),
        RIGHT_ELBOW:    r_elbow,
        RIGHT_WRIST:    r_wrist,
        LEFT_HIP:       Point(hip_x, cy - half * 0.8),
        RIGHT_HIP:      Point(hip_x, cy + half * 0.8),
        LEFT_KNEE:      Point(hip_x + size * 0.8, cy - half * 0.8),
        RIGHT_KNEE:     Point(hip_x + size * 0.8, cy + half * 0.8),
    })


# ── Distance invariance ───────────────────────────────────────────────────────

def test_raise_detected_at_every_apparent_size():
    # Same gesture, body spanning 12% to 60% of the frame — i.e. across the
    # room versus right in front of the camera.
    for size in (0.12, 0.20, 0.35, 0.60):
        r = _tracker()._classify_raw(_person(0.5, 0.5, size, raised=True), 1280, 720)
        assert r.state == ArmState.SINGLE_UP, f"size={size} gave {r.state}"
        assert r.raised_side == Side.RIGHT


def test_arm_down_rejected_at_every_apparent_size():
    for size in (0.12, 0.20, 0.35, 0.60):
        r = _tracker()._classify_raw(_person(0.5, 0.5, size, raised=False), 1280, 720)
        assert r.state == ArmState.DOWN, f"size={size} gave {r.state}"


def test_elevation_is_stable_across_apparent_size():
    # The measured elevation of the same pose must barely move with size —
    # this is the property that removes the need for body_scale_reference.
    elevs = []
    for size in (0.12, 0.20, 0.35, 0.60):
        r = _tracker()._classify_raw(_person(0.5, 0.5, size, raised=True), 1280, 720)
        elevs.append(r.elevation)
    assert max(elevs) - min(elevs) < 0.02, f"elevation drifted with size: {elevs}"


# ── Frame-shape invariance ────────────────────────────────────────────────────

def _same_body_on_frame(w: int, h: int, raised: bool) -> Landmarks:
    """The SAME physical person (fixed pixel geometry: 160px torso, centred)
    expressed in the per-axis normalised coordinates of a w x h frame.

    Comparing across frame shapes only means something if the underlying
    body is identical in pixels — a fixed normalised fixture would describe
    a differently-proportioned person on every aspect ratio.
    """
    torso_px, half_w_px = 160.0, 36.0
    cx_px, cy_px = w / 2.0, h / 2.0
    sh_y, hip_y = cy_px - torso_px / 2, cy_px + torso_px / 2
    if raised:
        r_elbow = (cx_px + half_w_px, sh_y - torso_px * 0.42)
        r_wrist = (cx_px + half_w_px, sh_y - torso_px * 0.85)
    else:
        r_elbow = (cx_px + half_w_px * 1.1, sh_y + torso_px * 0.42)
        r_wrist = (cx_px + half_w_px * 1.2, sh_y + torso_px * 0.85)

    def P(x_px, y_px, v=1.0):
        return Point(x_px / w, y_px / h, v)

    return Landmarks({
        NOSE:           P(cx_px, sh_y - torso_px * 0.28),
        LEFT_SHOULDER:  P(cx_px - half_w_px, sh_y),
        RIGHT_SHOULDER: P(cx_px + half_w_px, sh_y),
        LEFT_ELBOW:     P(cx_px - half_w_px * 1.1, sh_y + torso_px * 0.42),
        LEFT_WRIST:     P(cx_px - half_w_px * 1.2, sh_y + torso_px * 0.85),
        RIGHT_ELBOW:    P(*r_elbow),
        RIGHT_WRIST:    P(*r_wrist),
        LEFT_HIP:       P(cx_px - half_w_px * 0.8, hip_y),
        RIGHT_HIP:      P(cx_px + half_w_px * 0.8, hip_y),
        LEFT_KNEE:      P(cx_px - half_w_px * 0.8, hip_y + torso_px * 0.8),
        RIGHT_KNEE:     P(cx_px + half_w_px * 0.8, hip_y + torso_px * 0.8),
    })


def test_same_body_classifies_identically_on_any_frame_aspect():
    # 1280x640 is what the fisheye dewarp actually emits. Under the old
    # normalised-coordinate maths a 2:1 frame silently halved every
    # horizontal measurement against every vertical one, so the same person
    # could classify differently purely because of the frame's shape.
    results = {}
    for w, h in ((640, 640), (1280, 720), (1280, 640), (960, 540)):
        r = _tracker()._classify_raw(_same_body_on_frame(w, h, raised=True), w, h)
        results[(w, h)] = (r.state, round(r.elevation, 3), round(r.extension, 3))
    assert {v[0] for v in results.values()} == {ArmState.SINGLE_UP}, \
        f"frame shape changed the verdict: {results}"
    elevs = [v[1] for v in results.values()]
    assert max(elevs) - min(elevs) < 1e-6, f"elevation depended on frame shape: {results}"


def test_arm_down_stays_down_on_any_frame_aspect():
    for w, h in ((640, 640), (1280, 720), (1280, 640), (960, 540)):
        r = _tracker()._classify_raw(_same_body_on_frame(w, h, raised=False), w, h)
        assert r.state == ArmState.DOWN, f"{w}x{h} gave {r.state}"


# ── Posture invariance ────────────────────────────────────────────────────────

def test_reclined_raise_uses_the_same_threshold_as_standing():
    r = _tracker()._classify_raw(_reclined(0.5, 0.5, 0.25, raised=True), 1280, 720)
    assert r.state == ArmState.SINGLE_UP, f"got {r.state}"
    assert r.upright is False, "fixture should read as reclined"


def test_reclined_resting_arm_is_not_a_raise():
    r = _tracker()._classify_raw(_reclined(0.5, 0.5, 0.25, raised=False), 1280, 720)
    assert r.state == ArmState.DOWN, f"got {r.state}"


def test_hips_hidden_does_not_change_the_verdict():
    # Blanket over the lower body: the raise must still read the same. The
    # old code fell back to a much stricter threshold here and missed it.
    lm = _person(0.5, 0.5, 0.25, raised=True)
    visible = _tracker()._classify_raw(lm, 1280, 640)
    hidden_pts = dict(lm._points)
    for idx in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE):
        p = hidden_pts[idx]
        hidden_pts[idx] = Point(p.x, p.y, visibility=0.05)
    hidden = _tracker()._classify_raw(Landmarks(hidden_pts), 1280, 640)
    assert visible.state == hidden.state == ArmState.SINGLE_UP


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


# ── Left/right orientation ────────────────────────────────────────────────────

def _mirror(lm: Landmarks) -> Landmarks:
    """Flip a pose horizontally AND swap the left/right keypoint labels —
    i.e. the same physical person seen from the other side, which is the
    difference between facing the camera and facing away.
    """
    swap = {LEFT_SHOULDER: RIGHT_SHOULDER, RIGHT_SHOULDER: LEFT_SHOULDER,
            LEFT_ELBOW: RIGHT_ELBOW, RIGHT_ELBOW: LEFT_ELBOW,
            LEFT_WRIST: RIGHT_WRIST, RIGHT_WRIST: LEFT_WRIST,
            LEFT_HIP: RIGHT_HIP, RIGHT_HIP: LEFT_HIP,
            LEFT_KNEE: RIGHT_KNEE, RIGHT_KNEE: LEFT_KNEE}
    out = {}
    for idx, p in lm._points.items():
        out[swap.get(idx, idx)] = Point(1.0 - p.x, p.y, p.visibility)
    return Landmarks(out)


def test_arms_at_rest_are_not_crossed_arms_in_either_orientation():
    """Regression: the crossing test used raw image x, which assumes an
    orientation. A person facing the camera is mirrored — their right wrist
    sits left of the midline just by hanging at their side — so both arms at
    rest classified as CROSS_ARMS and stole gestures from an obvious raise.
    """
    lm = _person(0.5, 0.5, 0.25, raised=False)
    for label, pose in (("as-is", lm), ("mirrored", _mirror(lm))):
        r = _tracker()._classify_raw(pose, 1280, 640)
        assert r.state != ArmState.CROSS_ARMS, f"{label} gave {r.state}"


def test_a_raise_survives_mirroring():
    lm = _person(0.5, 0.5, 0.25, raised=True)
    for label, pose in (("as-is", lm), ("mirrored", _mirror(lm))):
        r = _tracker()._classify_raw(pose, 1280, 640)
        assert r.state == ArmState.SINGLE_UP, f"{label} gave {r.state}"
