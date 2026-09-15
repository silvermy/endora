"""
tests/test_collapsed_elbow.py

The false positive from the Jetson (2026-09-15).

A reclining person produced a tangled skeleton and fired a snap with
`extension: 1.00` — while every genuine raise ever captured measured
0.86-0.92. extension is arm_len / (upper + fore), which the triangle
inequality caps at exactly 1.0, reached when the elbow lies on the
shoulder-wrist line and in particular when it coincides with either end. A
pose model that cannot find the elbow puts it on top of one of them, so the
*least* informative reading scored as the straightest possible arm, and the
gate was a floor with no ceiling.

Also covers the pruning rule that let the junk detection in: person churn
kept the confidence hysteresis on its permissive "maintain" threshold, which
is how a 0.25-confidence box was admitted at all.
"""
import math

from cameras.arm_tracker import _elbow_is_articulated, _arm_metrics


# ── the degenerate readings ───────────────────────────────────────────────────

def test_elbow_on_the_wrist_scores_a_perfect_extension():
    # Documents the trap rather than the fix: this is why a floor alone is
    # not enough. The most broken reading possible looks the most convincing.
    shoulder, wrist = (100.0, 300.0), (100.0, 100.0)
    _elev, ext, _len = _arm_metrics(shoulder, wrist, wrist)
    assert ext == 1.0


def test_elbow_on_the_wrist_is_rejected():
    shoulder, wrist = (100.0, 300.0), (100.0, 100.0)
    assert not _elbow_is_articulated(shoulder, wrist, wrist)


def test_elbow_on_the_shoulder_is_rejected():
    shoulder, wrist = (100.0, 300.0), (100.0, 100.0)
    assert not _elbow_is_articulated(shoulder, shoulder, wrist)


def test_elbow_almost_on_the_wrist_is_rejected():
    # Not just exact coincidence: a keypoint 5% of the way along carries no
    # more information than one at 0%.
    shoulder, wrist = (100.0, 300.0), (100.0, 100.0)
    elbow = (100.0, 110.0)
    assert not _elbow_is_articulated(shoulder, elbow, wrist)


# ── the real arms it must not reject ──────────────────────────────────────────

def test_a_normally_bent_arm_is_accepted():
    shoulder, elbow, wrist = (100.0, 300.0), (140.0, 200.0), (100.0, 100.0)
    assert _elbow_is_articulated(shoulder, elbow, wrist)


def test_a_genuinely_straight_arm_is_accepted():
    # The case a hard extension ceiling would have broken: a real arm held
    # straight, elbow correctly placed mid-arm and dead on the line. Its
    # extension is ~1.0 and it must still pass.
    shoulder, elbow, wrist = (100.0, 300.0), (100.0, 200.0), (100.0, 100.0)
    _elev, ext, _len = _arm_metrics(shoulder, elbow, wrist)
    assert ext > 0.999
    assert _elbow_is_articulated(shoulder, elbow, wrist)


def test_the_captured_real_snaps_are_accepted():
    """Elbow positions reproducing the extension values of genuine fires."""
    shoulder, wrist = (100.0, 300.0), (100.0, 100.0)
    for offset in (20.0, 40.0, 60.0):          # increasing bend
        elbow = (100.0 + offset, 200.0)
        _elev, ext, _len = _arm_metrics(shoulder, elbow, wrist)
        assert 0.80 <= ext <= 0.99, ext        # the band real raises land in
        assert _elbow_is_articulated(shoulder, elbow, wrist), offset


def test_an_arm_raised_at_an_angle_is_accepted():
    # Articulation must not depend on the arm pointing straight up.
    shoulder = (100.0, 300.0)
    for deg in (30.0, 60.0, 90.0, 135.0):
        a = math.radians(deg)
        elbow = (shoulder[0] + 55 * math.sin(a), shoulder[1] - 55 * math.cos(a))
        wrist = (shoulder[0] + 110 * math.sin(a), shoulder[1] - 110 * math.cos(a))
        assert _elbow_is_articulated(shoulder, elbow, wrist), deg


def test_degenerate_zero_length_arm_is_rejected_without_dividing_by_zero():
    p = (100.0, 100.0)
    assert not _elbow_is_articulated(p, p, p)
