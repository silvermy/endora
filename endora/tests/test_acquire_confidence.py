"""
tests/test_acquire_confidence.py

A weak detection may CONTINUE a track; only a strong one may START one.

The detector runs at two confidences: strict while nobody is tracked, and
permissive once somebody is, so that the keypoint dropouts of an arm raise
don't lose the person mid-gesture. That relaxation was applied to the whole
frame, so one tracked person lowered the bar for every detection anywhere in
it — including brand-new ones with nothing to do with them.

On this camera that is self-sustaining, which is what makes it serious rather
than untidy: a ghost accepted at the low threshold becomes a tracked person,
which keeps the roster non-empty, which holds the threshold down, which
admits the next ghost.

Measured live on 2026-09-26 — 1234 pids in 40 hours, new tracks appearing on
wall art, in a kitchen doorway and in an empty corner. One of them was a
painting of a figure: accepted at confidence 0.39 against a configured
yolo_conf of 0.45, confirmed human 0.56 s later, and firing FOLDED_ARMS in an
empty room eleven seconds after that.

The same note in memory records this failing once before, when person-id
churn "pinned the confidence hysteresis on its permissive threshold, which
let a 0.25-conf junk detection fire".
"""
import numpy as np
import pytest

from cameras.analyser import _acquire_ok, _all_valid_landmarks

NEAR = (100.0, 100.0)
FAR = (900.0, 400.0)
MATCH = 80.0
ACQUIRE = 0.585            # yolo_conf 0.45 * 1.3


# ── the rule ──────────────────────────────────────────────────────────────────

def test_a_strong_detection_may_start_a_track_anywhere():
    assert _acquire_ok(0.9, FAR, [], MATCH, ACQUIRE) is True


def test_a_weak_detection_may_not_start_a_track():
    """0.39 on a painting, with a real person tracked elsewhere in the room."""
    assert _acquire_ok(0.39, FAR, [NEAR], MATCH, ACQUIRE) is False


def test_a_weak_detection_may_continue_an_existing_track():
    """The whole point of the permissive threshold: an arm raise drops
    keypoint confidence, and losing the person mid-gesture is the failure it
    was added to prevent."""
    assert _acquire_ok(0.39, (110.0, 105.0), [NEAR], MATCH, ACQUIRE) is True


def test_nobody_tracked_means_weak_detections_are_refused():
    assert _acquire_ok(0.39, NEAR, [], MATCH, ACQUIRE) is False


def test_a_detection_exactly_at_the_threshold_is_allowed():
    assert _acquire_ok(ACQUIRE, FAR, [], MATCH, ACQUIRE) is True


def test_a_missing_confidence_is_not_treated_as_weak():
    """Boxes are optional — a caller with no box array must not have every
    detection silently dropped."""
    assert _acquire_ok(None, FAR, [], MATCH, ACQUIRE) is True


def test_proximity_is_measured_against_the_match_radius():
    just_outside = (NEAR[0] + MATCH + 1, NEAR[1])
    just_inside = (NEAR[0] + MATCH - 1, NEAR[1])
    assert _acquire_ok(0.39, just_outside, [NEAR], MATCH, ACQUIRE) is False
    assert _acquire_ok(0.39, just_inside, [NEAR], MATCH, ACQUIRE) is True


# ── wired into the detection gate ─────────────────────────────────────────────

def _kps(cx: float, cy: float) -> np.ndarray:
    """Enough visible keypoints to pass the other gates."""
    k = np.zeros((17, 3), dtype=np.float32)
    for i in (0, 5, 6, 7, 8, 9, 10, 11, 12):
        k[i] = (cx, cy, 0.9)
    k[5] = (cx + 40, cy, 0.9)
    k[6] = (cx - 40, cy, 0.9)
    return k


def _run(conf, tracked):
    kps = np.stack([_kps(*FAR)])
    boxes = np.array([[0, 0, 10, 10, conf]], dtype=np.float32)
    return _all_valid_landmarks(
        kps, 1280, 640,
        fg_mask=None,                       # no mask → liveness not applied
        boxes=boxes,
        tracked_centroids=tracked,
        acquire_conf=ACQUIRE,
    )


def test_the_gate_drops_a_weak_unmatched_detection():
    assert _run(0.39, [NEAR]) == []


def test_the_gate_keeps_a_strong_unmatched_detection():
    assert len(_run(0.9, [NEAR])) == 1


def test_the_gate_is_inert_when_not_configured():
    """acquire_conf defaults to 0, so existing callers and tests that pass
    no boxes behave exactly as before."""
    kps = np.stack([_kps(*FAR)])
    assert len(_all_valid_landmarks(kps, 1280, 640, fg_mask=None)) == 1


# ── the analyser uses the strict value, not the permissive one ────────────────

def test_the_analyser_gates_acquisition_on_the_strict_threshold():
    import inspect
    from cameras import analyser
    src = inspect.getsource(analyser.CameraAnalyser._run)
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "_acquire_conf = base_conf * 1.3" in code
    assert "_maintain_conf = base_conf * 0.65" in code
    assert "acquire_conf=_acquire_conf" in code, \
        "the gate must use the STRICT threshold, not the permissive one"


def test_the_overlay_names_the_rejection():
    """A dropped detection must be visible on the debug overlay, or a ghost
    and a wrongly-dropped person both render as empty frame."""
    import inspect
    from cameras import analyser
    src = inspect.getsource(analyser.CameraAnalyser._run)
    assert "too weak to acquire" in src
