"""
tests/test_folded_arms.py

FOLDED_ARMS — hands folded together at the chest, facing the camera, sitting
or standing. The "I Dream of Jeannie" pose, minus the nod: a head nod moves
the nose about 9 px at this camera distance (4.5 px in the model's own input
space), which is at or below keypoint noise, so it is not detectable here.
The arm pose carries the whole gesture.

Two exclusions are requirements, not tuning:

* **Facing the camera.** In profile the two wrists overlap in the image
  whatever the hands are doing, so "hands together at the chest" becomes
  trivially true for anyone turned side-on.
* **Upright only.** Lying on a couch with forearms resting on the chest is
  geometrically this pose, and a couch-facing camera sees that for hours —
  it is the same class of false positive that fired T_POSE at 00:56 while
  the user was asleep.

Geometry is built in pixels on the user's real 1280x640 dewarped frame, at
the scale measured from a live capture: torso ~108 px, shoulders ~87 px.
"""
import math

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState
from tests.fake_landmarks import Landmarks, Point
from tests.fake_landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
)

W, H = 1280, 640
TORSO, SHOULDER_W = 108.0, 87.0      # measured from a real frame


def _pose(*, wrist_gap=30.0, wrist_dy=45.0, elbow_out=70.0,
          shoulder_w=SHOULDER_W, torso=TORSO, reclined=False,
          midline_shift=0.0, straight=False,
          cx=400.0, cy=350.0) -> Landmarks:
    """Hands folded at the chest, with knobs for each thing that should
    disqualify the pose.

    wrist_dy      how far below the shoulder line the hands sit
    elbow_out     how far the elbows stick out sideways (arm bend)
    shoulder_w    narrower = turned away from the camera
    torso         shoulder-to-hip length; scales WITH shoulder_w for a
                  distance change, and independently to alter the facing
                  ratio the two of them define
    reclined      lay the torso down instead of standing it up
    midline_shift push both wrists off-centre together
    straight      put the elbow on the shoulder-wrist line (unbent arm)
    """
    sh_y = cy
    ls = (cx + shoulder_w / 2, sh_y)
    rs = (cx - shoulder_w / 2, sh_y)
    if reclined:
        # Torso horizontal: hips beside the shoulders rather than below.
        lh = (cx + torso, sh_y + 10)
        rh = (cx + torso, sh_y - 10)
    else:
        lh = (cx + shoulder_w * 0.4, sh_y + torso)
        rh = (cx - shoulder_w * 0.4, sh_y + torso)

    wy = sh_y + wrist_dy
    lw = (cx + wrist_gap / 2 + midline_shift, wy)
    rw = (cx - wrist_gap / 2 + midline_shift, wy)
    if straight:
        # Elbow exactly halfway along shoulder->wrist: extension ~1.0.
        le = ((ls[0] + lw[0]) / 2, (ls[1] + lw[1]) / 2)
        re = ((rs[0] + rw[0]) / 2, (rs[1] + rw[1]) / 2)
    else:
        le = (cx + elbow_out, sh_y + wrist_dy * 0.6)
        re = (cx - elbow_out, sh_y + wrist_dy * 0.6)

    P = lambda p: Point(p[0] / W, p[1] / H)
    return Landmarks({
        NOSE: P((cx, sh_y - 34)),
        LEFT_SHOULDER: P(ls), RIGHT_SHOULDER: P(rs),
        LEFT_ELBOW: P(le), RIGHT_ELBOW: P(re),
        LEFT_WRIST: P(lw), RIGHT_WRIST: P(rw),
        LEFT_HIP: P(lh), RIGHT_HIP: P(rh),
        LEFT_KNEE: P((cx + 20, sh_y + torso * 1.8)),
        RIGHT_KNEE: P((cx - 20, sh_y + torso * 1.8)),
    })


def _state(lm, **cfg):
    """Hold the pose long enough for the tracker to confirm it.

    classify() debounces by state_confirm_s, so a single call can never
    report a newly-entered state — it answers DOWN whatever the geometry
    says. Feeding one frame and asserting on the result tests the debounce,
    not the pose.
    """
    tr = ArmTracker(ArmTrackerConfig(**cfg))
    state = None
    for i in range(8):                       # 0.8 s at 10 fps
        r = tr.classify(lm, W, H, None, now=i / 10.0)
        state = r.state if r is not None else None
    return state


# ── the pose itself ───────────────────────────────────────────────────────────

def test_hands_folded_at_the_chest_is_detected():
    assert _state(_pose()) is ArmState.FOLDED_ARMS


def test_detected_at_any_distance():
    # Every threshold is a multiple of shoulder width or torso length, so a
    # person further away must classify identically.
    for scale in (0.5, 1.0, 1.8):
        lm = _pose(wrist_gap=30 * scale, wrist_dy=45 * scale,
                   elbow_out=70 * scale, shoulder_w=SHOULDER_W * scale,
                   torso=TORSO * scale)
        assert _state(lm) is ArmState.FOLDED_ARMS, f"failed at scale {scale}"


def test_can_be_switched_off():
    assert _state(_pose(), detect_folded_arms=False) is not ArmState.FOLDED_ARMS


# ── the exclusions the user asked for ─────────────────────────────────────────

def test_reclining_is_excluded():
    # Lying on a couch with forearms on the chest is this pose geometrically.
    # A couch-facing camera sees that for hours at a time.
    assert _state(_pose(reclined=True)) is not ArmState.FOLDED_ARMS


def test_turned_away_from_the_camera_is_excluded():
    # In profile the wrists overlap in the image whatever the hands are
    # doing, so the pose would otherwise be trivially satisfied. Rotation
    # foreshortens the shoulders while the torso keeps its length, which is
    # exactly the ratio facing_shoulder_min measures.
    assert _state(_pose(shoulder_w=SHOULDER_W * 0.35)) is not ArmState.FOLDED_ARMS


def test_facing_threshold_admits_a_slight_turn():
    # Sitting at an angle to the camera is normal and must still work.
    assert _state(_pose(shoulder_w=SHOULDER_W * 0.75)) is ArmState.FOLDED_ARMS


# ── things that must NOT be read as the gesture ───────────────────────────────

def test_hands_apart_is_not_folded():
    assert _state(_pose(wrist_gap=SHOULDER_W * 1.2)) is not ArmState.FOLDED_ARMS


def test_hands_in_the_lap_is_not_folded():
    # Below the hip line — the resting position of anyone sitting down.
    assert _state(_pose(wrist_dy=TORSO * 1.6)) is not ArmState.FOLDED_ARMS


def test_straight_arms_are_not_folded():
    # Folded arms are bent arms. An unbent pair reaching to the same spot is
    # someone holding something, not this gesture.
    assert _state(_pose(straight=True)) is not ArmState.FOLDED_ARMS


def test_both_hands_off_to_one_side_is_not_folded():
    # Hands together but carried away from the centre — reaching, holding a
    # mug, resting on an armrest.
    assert _state(_pose(midline_shift=SHOULDER_W * 0.8)) is not ArmState.FOLDED_ARMS


def test_does_not_shadow_a_raised_arm():
    # Two-handed poses are tested BEFORE the single-arm raise and return
    # early, so a new pose that over-matches steals snaps. This is the exact
    # failure CROSS_ARMS caused before the body-axis fix.
    lm = _pose()
    lm._points[RIGHT_WRIST] = Point(400 / W, (350 - 150) / H)   # arm straight up
    lm._points[RIGHT_ELBOW] = Point(400 / W, (350 - 75) / H)
    assert _state(lm) is not ArmState.FOLDED_ARMS


def test_stays_distinct_from_cross_arms():
    """The two poses must not both match the same body.

    FOLDED_ARMS keeps the wrists NEAR the midline; CROSS_ARMS requires each
    to pass it onto the other side. They are mutually exclusive only while
    folded_midline_max stays below cross_arms_min_crossing, so pin that.
    """
    c = ArmTrackerConfig()
    assert c.folded_midline_max < c.cross_arms_min_crossing

    # A genuinely crossed pose: each wrist well past the midline.
    crossed = _pose()
    crossed._points[LEFT_WRIST] = Point((400 - 45) / W, 395 / H)
    crossed._points[RIGHT_WRIST] = Point((400 + 45) / W, 395 / H)
    assert _state(crossed) is not ArmState.FOLDED_ARMS
