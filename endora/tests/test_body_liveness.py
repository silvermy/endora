"""
tests/test_body_liveness.py

Is this thing made of pixels that ever change?

The wrist-patch liveness test asks a narrower question — did a hand move —
and a framed painting of a person answers it exactly the way a sleeping
person does. But a painting differs from the learned background NOWHERE, and
never: not at the wrists, not at the head, not once in twenty-nine hours. A
living body differs from it somewhere, continuously.

Live evidence, 2026-09-27/28: a Picasso-ish figure on the living-room wall
was detected at confidence 0.71 — above the strict acquire threshold, so no
confidence gate could touch it — confirmed human, and fired FOLDED_ARMS 17
times in 20 minutes from one long-lived pid, in an empty room.

Note what these tests can and cannot show. The captures hold keypoints, not
foreground masks, so the NEGATIVE side (a painting scores exactly zero) is
provable from real recorded data and is asserted below against the actual
painting keypoints. The POSITIVE side is not measurable offline and is
covered only by synthetic masks; that is why the threshold ships near zero
and why the measured fraction is logged on every rejection.
"""
import numpy as np
import pytest

from cameras.analyser import (
    _body_shows_life, _body_foreground_frac, _wrist_shows_motion,
    _passes_liveness_gate, _BODY_FOREGROUND_MIN, _BODY_FOREGROUND_MIN_PX,
)

W, H, MW, MH = 1280, 640, 160, 120


def _person(cx=400.0, cy=300.0, w=80.0, h=200.0) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[0] = (cx, cy - h / 2, 0.9)                 # nose
    k[5], k[6] = (cx + w / 2, cy - h / 4, 0.9), (cx - w / 2, cy - h / 4, 0.9)
    k[7], k[8] = (cx + w / 2, cy, 0.9), (cx - w / 2, cy, 0.9)
    k[9], k[10] = (cx + w / 3, cy + h / 5, 0.9), (cx - w / 3, cy + h / 5, 0.9)
    k[11], k[12] = (cx + w / 3, cy + h / 3, 0.9), (cx - w / 3, cy + h / 3, 0.9)
    return k


def _mask_over(kps, fill=1.0):
    """Foreground filling `fill` of the candidate's own sampled region."""
    m = np.zeros((MH, MW), dtype=np.uint8)
    vis = kps[kps[:, 2] > 0.3]
    xs, ys = vis[:, 0] * MW / W, vis[:, 1] * MH / H
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    n = max(1, int((y1 - y0) * fill))
    m[y0:y0 + n, x0:x1] = 255
    return m


# ── the negative side: proven against the real painting ──────────────────────

def test_a_perfectly_static_region_shows_no_life():
    assert _body_shows_life(_person(), np.zeros((MH, MW), np.uint8), W, H) is False


def test_the_real_painting_keypoints_score_zero():
    """Replayed from the captures that fired 17 times. Against a settled
    background model the painting's own body region is entirely background,
    which is the signal no confidence threshold could provide."""
    import glob
    from pathlib import Path
    caps = sorted(glob.glob(str(Path(__file__).parent / "ghosts" / "*.npz")))
    assert caps, "the painting traces must stay committed — see tests/ghosts/"
    static = np.zeros((MH, MW), np.uint8)
    for path in caps:
        for f in np.load(path, allow_pickle=True)["keypoints"]:
            if (f[:, 2] > 0.3).sum() >= 6:
                assert _body_shows_life(f, static, W, H) is False


def test_single_pixel_noise_is_not_life():
    """MOG2 flips the odd pixel on a settled region; one is not evidence."""
    m = np.zeros((MH, MW), np.uint8)
    k = _person()
    m[int(300 * MH / H), int(400 * MW / W)] = 255
    assert _body_shows_life(k, m, W, H) is False
    assert _BODY_FOREGROUND_MIN_PX >= 2


# ── the positive side: synthetic only, and deliberately easy to satisfy ──────

def test_a_body_region_full_of_foreground_shows_life():
    k = _person()
    assert _body_shows_life(k, _mask_over(k, 1.0), W, H) is True


def test_a_modest_fraction_is_enough():
    """The threshold's job is to separate zero from non-zero, not to measure
    motion. A person who barely stirs must still pass."""
    k = _person()
    assert _body_shows_life(k, _mask_over(k, 0.25), W, H) is True


def test_the_threshold_stays_near_zero():
    assert _BODY_FOREGROUND_MIN <= 0.01, (
        "this test cannot be validated against real foreground masks — the "
        "captures hold keypoints only — so it must stay conservative until "
        "the logged distribution says otherwise")


# ── abstention: never reject for want of something to measure ────────────────

def test_no_background_model_yet_abstains():
    assert _body_shows_life(_person(), None, W, H) is True
    assert _body_foreground_frac(_person(), None, W, H) is None


def test_too_few_keypoints_abstains():
    sparse = np.zeros((17, 3), dtype=np.float32)
    sparse[0] = (400, 300, 0.9)
    assert _body_shows_life(sparse, np.zeros((MH, MW), np.uint8), W, H) is True


# ── how it combines with the wrist test ──────────────────────────────────────

def test_either_signal_admits_a_candidate():
    """A hand-waver shows life at the wrists and barely across the body; a
    still person the reverse. Requiring BOTH dropped the hand-waver, which
    is how this landed wrong the first time."""
    k = _person()
    wrist_only = np.zeros((MH, MW), np.uint8)
    wx, wy = int(k[9][0] * MW / W), int(k[9][1] * MH / H)
    wrist_only[wy - 3:wy + 4, wx - 3:wx + 4] = 255

    assert _wrist_shows_motion(k, wrist_only, W, H, 0.12) is True
    assert _passes_liveness_gate(k, (400.0, 300.0), wrist_only, W, H, 0.12) is True


def test_neither_signal_rejects():
    k = _person()
    static = np.zeros((MH, MW), np.uint8)
    assert _passes_liveness_gate(k, (400.0, 300.0), static, W, H, 0.12) is False


def test_the_measured_fraction_is_reported_for_tuning():
    k = _person()
    assert _body_foreground_frac(k, _mask_over(k, 1.0), W, H) == pytest.approx(1.0, abs=0.2)
    assert _body_foreground_frac(k, np.zeros((MH, MW), np.uint8), W, H) == 0.0


def test_the_overlay_shows_the_measured_value():
    import inspect
    from cameras import analyser
    src = inspect.getsource(analyser.CameraAnalyser._run)
    assert "not live bf=" in src, \
        "a rejection must report what it measured, or it cannot be retuned"
