"""
tests/test_surround_contrast.py

Is the change the candidate's own, or the whole scene's?

Both liveness tests ask whether pixels differ from the learned background,
and a change in the light answers yes everywhere at once. Live evidence,
2026-09-28: afternoon sun lit up the painting's box, the painting was
confirmed human fourteen times and fired FOLDED_ARMS eight times with nobody
in the room. A light change marks the wall around a painting as much as the
painting; a person differs from what surrounds them.

As with test_body_liveness, the captures hold keypoints and not masks, so the
masks here are synthetic. The painting keypoints are the real recorded ones.
"""
import glob
from pathlib import Path

import numpy as np

from cameras.analyser import (
    _SURROUND_MIN_RATIO, _SURROUND_QUIET, _all_valid_landmarks,
    _passes_liveness_gate, _stands_out, _surround_contrast,
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


def _box(kps):
    vis = kps[kps[:, 2] > 0.3]
    xs, ys = vis[:, 0] * MW / W, vis[:, 1] * MH / H
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def _own_foreground(kps) -> np.ndarray:
    m = np.zeros((MH, MW), np.uint8)
    x0, x1, y0, y1 = _box(kps)
    m[y0:y1, x0:x1] = 255
    return m


def _painting_frames():
    caps = sorted(glob.glob(str(Path(__file__).parent / "ghosts" / "*.npz")))
    assert caps, "the painting traces must stay committed — see tests/ghosts/"
    for path in caps:
        for f in np.load(path, allow_pickle=True)["keypoints"]:
            if (f[:, 2] > 0.3).sum() >= 6:
                yield f


# ── the failure this exists for ───────────────────────────────────────────────

def test_a_light_change_across_the_painting_is_not_life():
    """The whole scene reads as changed: the painting's box and the wall
    around it alike. Before this check, that passed the liveness gate."""
    lit = np.full((MH, MW), 255, np.uint8)
    for f in _painting_frames():
        assert _stands_out(f, lit, W, H) is False
        assert _passes_liveness_gate(f, (40.0, 120.0), lit, W, H, 0.12) is False


def test_a_lit_painting_is_not_admitted_as_a_person():
    lit = np.full((MH, MW), 255, np.uint8)
    for f in _painting_frames():
        assert _all_valid_landmarks(f[None], W, H, fg_mask=lit) == []


def test_a_patchy_light_change_is_not_life_either():
    """Exposure noise is not uniform. Half the pixels flickering, inside the
    box and out, is still the scene and not the candidate."""
    rng = np.random.default_rng(0)
    noisy = (rng.random((MH, MW)) < 0.5).astype(np.uint8) * 255
    for f in _painting_frames():
        assert _stands_out(f, noisy, W, H) is False


# ── what it must leave alone ──────────────────────────────────────────────────

def test_a_person_changing_against_a_quiet_room_stands_out():
    k = _person()
    assert _stands_out(k, _own_foreground(k), W, H) is True
    assert _passes_liveness_gate(k, (400.0, 300.0), _own_foreground(k), W, H, 0.12) is True


def test_a_person_moving_while_the_room_is_a_little_busy_stands_out():
    """Some change around a real person is normal — hair and feet beyond the
    keypoints, a shadow. They stand out as long as they are clearly more
    changed than their surroundings."""
    k = _person()
    m = _own_foreground(k)
    x0, x1, y0, y1 = _box(k)
    ring_rows = slice(max(0, y0 - 10), y0)
    m[ring_rows, x0:x1] = 255          # a strip of spill just above the box
    inside, ring = _surround_contrast(k, m, W, H)
    assert ring >= _SURROUND_QUIET
    assert inside >= _SURROUND_MIN_RATIO * ring
    assert _stands_out(k, m, W, H) is True


def test_a_still_person_in_a_quiet_room_is_not_affected():
    """With the surroundings quiet this changes nothing: the existing tests
    decide, exactly as before."""
    k = _person()
    quiet = np.zeros((MH, MW), np.uint8)
    assert _stands_out(k, quiet, W, H) is True


def test_a_confirmed_person_survives_a_light_change():
    """A known, confirmed person is exempt before any foreground test runs,
    so a light change cannot drop someone who is sitting still."""
    k = _person()
    lit = np.full((MH, MW), 255, np.uint8)
    assert _passes_liveness_gate(
        k, (400.0, 300.0), lit, W, H, 0.12,
        known_centroids=[(402.0, 301.0)], match_dist=50.0,
    ) is True


def test_nothing_to_measure_abstains():
    k = _person()
    assert _stands_out(k, None, W, H) is True
    sparse = np.zeros((17, 3), np.float32)
    sparse[0] = (400, 300, 0.9)
    assert _stands_out(sparse, np.full((MH, MW), 255, np.uint8), W, H) is True
