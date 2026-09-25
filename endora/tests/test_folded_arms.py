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

import pytest

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
          midline_shift=0.0, straight=False, lean_deg=0.0, knee_rise=None,
          hip_vis=1.0, knee_vis=1.0,
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
    lean_deg      tip the torso off vertical — a partial recline, which is
                  what a couch actually produces
    knee_rise     put the knees this many shoulder widths ABOVE the hips
                  (feet up); None leaves them below, as when seated
    hip_vis       confidence on the hip keypoints
    knee_vis      confidence on the knee keypoints
    """
    sh_y = cy
    ls = (cx + shoulder_w / 2, sh_y)
    rs = (cx - shoulder_w / 2, sh_y)
    if reclined:
        # Torso horizontal: hips beside the shoulders rather than below.
        lh = (cx + torso, sh_y + 10)
        rh = (cx + torso, sh_y - 10)
        hx, hy = cx + torso, sh_y
    else:
        hx = cx + torso * math.sin(math.radians(lean_deg))
        hy = sh_y + torso * math.cos(math.radians(lean_deg))
        lh = (hx + shoulder_w * 0.4, hy)
        rh = (hx - shoulder_w * 0.4, hy)

    knee_y = (hy - knee_rise * shoulder_w) if knee_rise is not None \
        else hy + torso * 0.8

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
    V = lambda p, v: Point(p[0] / W, p[1] / H, v)
    return Landmarks({
        NOSE: P((cx, sh_y - shoulder_w * 0.39)),
        LEFT_SHOULDER: P(ls), RIGHT_SHOULDER: P(rs),
        LEFT_ELBOW: P(le), RIGHT_ELBOW: P(re),
        LEFT_WRIST: P(lw), RIGHT_WRIST: P(rw),
        LEFT_HIP: V(lh, hip_vis), RIGHT_HIP: V(rh, hip_vis),
        LEFT_KNEE: V((hx + 20, knee_y), knee_vis),
        RIGHT_KNEE: V((hx - 20, knee_y), knee_vis),
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


# ── posture: sitting or standing, not reclining ───────────────────────────────
#
# Fourteen false fires in one evening, all of them the user on the couch. The
# original guard was the shared `upright` flag, which asks only whether the
# torso is more vertical than horizontal — so it passes anything short of a 45
# degree lean, and a couch is a 25-45 degree lean. Nobody lies flat on a sofa.

@pytest.mark.parametrize("deg", [25, 30, 40, 44])
def test_a_partial_recline_is_excluded(deg):
    """The shape a couch actually produces. `upright` says True for all of
    these, which is exactly why it was not enough on its own."""
    assert _state(_pose(lean_deg=deg)) is not ArmState.FOLDED_ARMS


@pytest.mark.parametrize("deg", [0, 8, 15])
def test_sitting_at_a_normal_slouch_still_works(deg):
    # Measured on recorded captures of the real gesture: 1.8-13.7 degrees of
    # lean, never above 16.6. Nobody sits perfectly straight.
    assert _state(_pose(lean_deg=deg)) is ArmState.FOLDED_ARMS


def test_the_lean_limit_sits_between_the_two_populations():
    c = ArmTrackerConfig()
    assert 17.0 < c.folded_lean_max_deg < 25.0, (
        "genuine captures reached 16.6 degrees and reclining started at 25 — "
        "a threshold outside that gap drops real gestures or keeps false ones")


def test_knees_above_the_hips_is_excluded():
    """Feet up on the couch. The single cleanest signal in the recorded data:
    every genuine capture has the knees at or below the hips (+0.02 to +1.51
    shoulder widths below), and thirteen of the fourteen false fires have
    them above (0.06 to 2.90 above)."""
    assert _state(_pose(knee_rise=1.0)) is not ArmState.FOLDED_ARMS


def test_knees_level_with_the_hips_is_still_seated():
    # A normal chair posture, and one genuine capture measured +0.02 — so the
    # threshold is deliberately slack rather than sitting at zero.
    assert _state(_pose(knee_rise=0.0)) is ArmState.FOLDED_ARMS


def test_invisible_knees_do_not_block_the_gesture():
    # Legs out of frame is ordinary for someone seated close to the camera:
    # four of the eight genuine captures have knee confidence below 0.4.
    assert _state(_pose(knee_vis=0.05)) is ArmState.FOLDED_ARMS


def test_invisible_hips_do_block_it():
    """Posture is REQUIRED here, not merely "not contradicted".

    A frame whose hips cannot be seen cannot be shown to be sitting, and this
    is the one gesture whose false-positive mode is a posture. The confirm
    accumulator already tolerates a few unreadable frames without discarding
    a hold, so this costs nothing on a real gesture.
    """
    assert _state(_pose(hip_vis=0.05)) is not ArmState.FOLDED_ARMS


# ── posture memory ────────────────────────────────────────────────────────────

def _feed(tr, frames, fps=10.0, t0=0.0):
    """Run frames through a tracker, returning the last state."""
    state = None
    for i, lm in enumerate(frames):
        r = tr.classify(lm, W, H, None, now=t0 + i / fps)
        state = r.state if r is not None else None
    return state


def test_a_recent_sighting_of_reclining_suppresses_the_gesture():
    """Why a per-frame posture test is not enough on its own.

    On the recorded false positives the model alternates, frame to frame,
    between two skeletons of the same reclining body: one correct, and one
    hallucinated — a short upright torso with the hips guessed partway up the
    body and the knees essentially invisible. The correct frames are rejected
    and the hallucinated ones sail through, so the gesture still fired. A
    body does not sit up and lie down eight times a second.
    """
    tr = ArmTracker(ArmTrackerConfig())
    lying = _pose(lean_deg=45, knee_rise=2.0)
    upright_looking = _pose(knee_vis=0.05)
    # Half a second of clearly lying down, then the hallucinated upright read.
    assert _feed(tr, [lying] * 5 + [upright_looking] * 8) \
        is not ArmState.FOLDED_ARMS


def test_the_memory_expires():
    """It suppresses, so it must not suppress forever — getting up off the
    couch and performing the gesture has to work."""
    tr = ArmTracker(ArmTrackerConfig())
    c = ArmTrackerConfig()
    _feed(tr, [_pose(lean_deg=45, knee_rise=2.0)] * 5)
    later = c.folded_recline_memory_s + 0.5
    assert _feed(tr, [_pose()] * 10, t0=later) is ArmState.FOLDED_ARMS


def test_the_veto_outranks_the_hysteresis_hold():
    """state_release_s lets a confirmed pose coast through half a second of
    contradicting frames. On a recorded capture that was enough for the
    sustain timer to complete on frames that plainly showed a reclining body
    — 51 degrees of lean, knees two shoulder widths above the hips. Seeing
    the person lying down is not the kind of contradiction to sit out.
    """
    tr = ArmTracker(ArmTrackerConfig())
    assert _feed(tr, [_pose()] * 8) is ArmState.FOLDED_ARMS      # confirmed
    lying = _pose(lean_deg=45, knee_rise=2.0)
    r = tr.classify(lying, W, H, None, now=0.9)                  # one frame
    assert r.state is not ArmState.FOLDED_ARMS, \
        "the hold coasted through a frame showing the person lying down"


def test_the_veto_needs_confident_keypoints():
    """It can only suppress, so a guessed hip that blocks a real gesture is
    worse than one that fails to block a false one. The hallucinated
    skeletons carry hip confidence around 0.50 with the knees at 0.05-0.11.
    """
    c = ArmTrackerConfig()
    assert c.folded_posture_visibility_min >= c.folded_visibility_min
    tr = ArmTracker(ArmTrackerConfig())
    # A low-confidence "lying down" reading must not veto what follows.
    _feed(tr, [_pose(lean_deg=45, knee_rise=2.0, hip_vis=0.15, knee_vis=0.15)] * 5)
    assert _feed(tr, [_pose()] * 10, t0=0.5) is ArmState.FOLDED_ARMS


def test_no_other_gesture_gained_a_posture_requirement():
    """A raised arm from a couch is still a raised arm — this whole guard is
    scoped to the one gesture whose false-positive mode is a posture."""
    tr = ArmTracker(ArmTrackerConfig())
    # Lean only: knees high enough to clear the shoulder line would trip the
    # unrelated leg-raise guard, which is a different question.
    _feed(tr, [_pose(lean_deg=45)] * 5)          # arm the recline memory
    raised = _pose(lean_deg=45)
    raised._points[RIGHT_WRIST] = Point(400 / W, (350 - 150) / H)
    raised._points[RIGHT_ELBOW] = Point(400 / W, (350 - 75) / H)
    assert _feed(tr, [raised] * 10, t0=0.5) is ArmState.SINGLE_UP


def test_turned_away_from_the_camera_is_excluded():
    # In profile the wrists overlap in the image whatever the hands are
    # doing, so the pose would otherwise be trivially satisfied. Rotation
    # foreshortens the shoulders while the torso keeps its length, which is
    # exactly the ratio facing_shoulder_min measures.
    assert _state(_pose(shoulder_w=SHOULDER_W * 0.35)) is not ArmState.FOLDED_ARMS


def test_facing_threshold_admits_a_slight_turn():
    # Sitting at an angle to the camera is normal and must still work. The
    # hands come in with the shoulders, since every threshold is now scaled
    # by shoulder width rather than torso length.
    narrow = SHOULDER_W * 0.75
    assert _state(_pose(shoulder_w=narrow, wrist_dy=45 * 0.75,
                        wrist_gap=30 * 0.75)) is ArmState.FOLDED_ARMS


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


def test_hands_on_a_laptop_are_not_folded_arms():
    """The gesture's first false positive, 13 seconds after it went live.

    Sitting on the couch with a laptop: facing the camera, upright, hands
    together in front of the body — every test the first version applied.
    What it did not check was HEIGHT. "At the chest" accepted anywhere
    between the shoulder and hip lines, which is the whole torso, and a
    laptop sits near the bottom of it. Folding your hands at your chest puts
    them near the sternum.
    """
    laptop = _pose(wrist_dy=TORSO * 0.75)          # hands low, over a lap
    assert _state(laptop) is not ArmState.FOLDED_ARMS


def test_the_chest_band_is_the_upper_torso_only():
    c = ArmTrackerConfig()
    assert c.folded_chest_depth < 1.0, \
        "a band reaching the hip line admits hands resting in the lap"
    # Hands at the sternum qualify; hands most of the way to the hips do not.
    assert _state(_pose(wrist_dy=TORSO * 0.30)) is ArmState.FOLDED_ARMS
    assert _state(_pose(wrist_dy=TORSO * 0.70)) is not ArmState.FOLDED_ARMS


def test_a_flickering_hold_still_confirms():
    """Recorded from the deployed camera, the reason this gesture felt broken.

    Two hands pressed together look like one blob, so the model guesses
    which wrist is where: the measured gap swings between 0.06 and 0.85
    shoulder widths in adjacent frames while the person has not moved. Only
    4 frames in 12 of a real hold matched.

    Three things each independently prevented a fire, and all three had to
    go. The tracker discarded its whole confirm accumulator on a single
    contradicting frame; the state machine wiped the sustain timer whenever
    the state touched DOWN; and the release window expired 0.02 s before the
    sustain completed. Loosening the geometry instead would have re-admitted
    the laptop pose, which sits inside the flicker.
    """
    tr = ArmTracker(ArmTrackerConfig())
    from core.state_machine import GestureStateMachine, StateMachineConfig, Gesture
    sm = GestureStateMachine(StateMachineConfig())
    held, apart = _pose(), _pose(wrist_gap=SHOULDER_W * 1.3)
    fired = []
    for i in range(30):                        # 3 s at 10 fps
        # Two good frames in three, which is what the recording shows.
        lm = held if i % 3 != 1 else apart
        g = sm.tick(tr.classify(lm, W, H, None, now=i / 10.0), i / 10.0)
        if g is Gesture.FOLDED_ARMS:
            fired.append(i / 10.0)
    assert fired, "a hold measured two frames in three never confirmed"
    assert fired[0] < 1.5, f"took {fired[0]:.1f}s to recognise a steady hold"
