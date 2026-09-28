"""
tests/test_body_scale_stability.py

A person cannot shrink by half in a seventh of a second.

Recorded live 2026-09-28: somebody sitting with their arm out roughly
horizontal, while the detector intermittently substituted a skeleton a fifth
of the size — shoulders 107 px, then 20 px in the next frame — and in the
collapsed version the short "arm" pointed straight up. Two false snaps, and
the arm never rose at all.

What makes this worth a dedicated guard is that every scale-relative test is
blind to it by construction. The arm was the right length FOR that tiny body
(min_arm_len_frac passed at 0.85-1.35), the torso the right shape, the pose
upright. Confidence gave no warning. The only thing that betrays it is
comparing the body against its OWN recent history.

And the damage is not the one bad frame. It is that sweep_climb and
sweep_rate are then measured ACROSS the substitution — real elevation -0.65
against garbage elevation +1.00, an apparent climb of 1.65 at a plausible
rate — so a frame that is never classified as anything still manufactures a
gesture. Hence the rejection happens before the frame reaches the history.
"""
import math

import numpy as np
import pytest

from cameras.arm_tracker import ArmTracker, ArmTrackerConfig, ArmState
from tests.fake_landmarks import Landmarks, Point
from tests.fake_landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
)

W, H = 1280, 640


def _body(shoulder_w=100.0, torso=120.0, cx=400.0, cy=300.0) -> Landmarks:
    P = lambda x, y: Point(x / W, y / H)
    return Landmarks({
        NOSE: P(cx, cy - shoulder_w * 0.4),
        LEFT_SHOULDER: P(cx + shoulder_w / 2, cy),
        RIGHT_SHOULDER: P(cx - shoulder_w / 2, cy),
        LEFT_ELBOW: P(cx + shoulder_w * 0.6, cy + 40),
        RIGHT_ELBOW: P(cx - shoulder_w * 0.6, cy + 40),
        LEFT_WRIST: P(cx + shoulder_w * 0.5, cy + 80),
        RIGHT_WRIST: P(cx - shoulder_w * 0.5, cy + 80),
        LEFT_HIP: P(cx + shoulder_w * 0.4, cy + torso),
        RIGHT_HIP: P(cx - shoulder_w * 0.4, cy + torso),
        LEFT_KNEE: P(cx + 20, cy + torso * 1.8),
        RIGHT_KNEE: P(cx - 20, cy + torso * 1.8),
    })


def _feed(tr, frames, fps=10.0):
    """Returns the RAW classification per frame; None means rejected."""
    out = []
    for i, lm in enumerate(frames):
        out.append(tr._classify_raw(lm, W, H, now=i / fps))
    return out


# ── the collapse ──────────────────────────────────────────────────────────────

def test_a_collapsed_skeleton_is_rejected():
    tr = ArmTracker(ArmTrackerConfig())
    frames = [_body(shoulder_w=100)] * 8 + [_body(shoulder_w=20)]
    assert _feed(tr, frames)[-1] is None


def test_it_measures_shoulders_not_body_size():
    """body_size is max(shoulder_w, torso_len), and a collapsed skeleton
    frequently keeps a spurious long torso — shoulders 28 px against a torso
    of 118 in one recorded frame. The max stays large and hides exactly the
    collapse being looked for.
    """
    tr = ArmTracker(ArmTrackerConfig())
    frames = [_body(shoulder_w=100, torso=120)] * 8 + [
        _body(shoulder_w=20, torso=120)]        # shoulders gone, torso intact
    assert _feed(tr, frames)[-1] is None, \
        "a long torso must not mask collapsed shoulders"


def test_a_normal_frame_is_untouched():
    tr = ArmTracker(ArmTrackerConfig())
    assert all(r is not None for r in _feed(tr, [_body()] * 12))


def test_small_variation_is_tolerated():
    """Keypoint noise moves shoulder width around by a few per cent every
    frame; only a collapse is a collapse."""
    tr = ArmTracker(ArmTrackerConfig())
    frames = [_body(shoulder_w=100 + 8 * math.sin(i)) for i in range(20)]
    assert all(r is not None for r in _feed(tr, frames))


# ── it must not become a trap ─────────────────────────────────────────────────

def test_a_real_size_change_is_accepted_after_a_short_run():
    """The failure mode this guard could easily introduce. A rejected frame
    does not update the median, so somebody who genuinely changes size —
    walking away from the camera, standing up — would be rejected once and
    then forever, because the median could never follow them.
    """
    c = ArmTrackerConfig()
    tr = ArmTracker(ArmTrackerConfig())
    frames = [_body(shoulder_w=100)] * 8 + [_body(shoulder_w=40)] * 12
    out = _feed(tr, frames)
    assert out[8] is None, "the first small frame should be treated as a glitch"
    assert out[-1] is not None, "a persistent new size must be accepted"
    rejected = sum(1 for r in out[8:] if r is None)
    assert rejected <= c.body_scale_reject_run_max + 1, \
        f"{rejected} frames rejected — the median never followed the person"


def test_walking_away_gradually_is_never_rejected():
    tr = ArmTracker(ArmTrackerConfig())
    frames = [_body(shoulder_w=120 * (0.97 ** i)) for i in range(40)]
    assert all(r is not None for r in _feed(tr, frames))


def test_the_check_abstains_until_it_has_history():
    """A newly acquired track has nothing to be inconsistent with."""
    c = ArmTrackerConfig()
    tr = ArmTracker(ArmTrackerConfig())
    out = _feed(tr, [_body(shoulder_w=20)] * c.body_scale_min_samples)
    assert all(r is not None for r in out)


def test_thresholds_are_sane():
    c = ArmTrackerConfig()
    assert 0.0 < c.body_scale_min_ratio < 1.0
    assert c.body_scale_min_samples >= 2
    assert c.body_scale_reject_run_max >= 1


# ── the point of it: the sweep must not be measured across the substitution ──

def test_a_rejected_frame_does_not_reach_the_sweep_history():
    """The whole reason the rejection happens early. A collapsed frame that
    merely classified as DOWN would still have poisoned sweep_climb, which
    is how two false snaps were manufactured from an arm that never rose.
    """
    tr = ArmTracker(ArmTrackerConfig())
    low = _body(shoulder_w=100)
    _feed(tr, [low] * 10)

    spike = _body(shoulder_w=18)
    spike._points[LEFT_WRIST] = Point(400 / W, (300 - 200) / H)   # "arm up"
    for i in range(3):
        tr._classify_raw(spike, W, H, now=1.0 + i / 10.0)

    after = tr._classify_raw(low, W, H, now=1.5)
    assert after is not None
    assert after.sweep_climb < 0.60, (
        f"climb {after.sweep_climb:.2f} — a collapsed skeleton reached the "
        f"sweep history and manufactured a flourish")
