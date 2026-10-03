"""
tests/test_short_term_motion.py

Confirmation asks for movement, not difference from the background.

Live evidence, 2026-09-30 7 am: early sun lit the painting alone, so the
background model reported its box as changed on body and wrist, twice, two
seconds apart, which is what a person looks like to it. The painting was
confirmed human and fired FOLDED_ARMS five times into an empty room. Light
drifts over seconds to minutes; a body shifts from moment to moment.

As elsewhere in the liveness tests, the frames and masks here are synthetic:
the captures hold keypoints, not images. The painting keypoints are real.
"""
import glob
from pathlib import Path

import numpy as np

from cameras.analyser import (
    _MOTION_GAP_S, _ShortTermMotion, _all_valid_landmarks, _body_moved,
)

W, H, MW, MH = 1280, 640, 320, 160


def _person(cx=400.0, cy=300.0, w=80.0, h=200.0) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[0] = (cx, cy - h / 2, 0.9)
    k[5], k[6] = (cx + w / 2, cy - h / 4, 0.9), (cx - w / 2, cy - h / 4, 0.9)
    k[7], k[8] = (cx + w / 2, cy, 0.9), (cx - w / 2, cy, 0.9)
    k[9], k[10] = (cx + w / 3, cy + h / 5, 0.9), (cx - w / 3, cy + h / 5, 0.9)
    k[11], k[12] = (cx + w / 3, cy + h / 3, 0.9), (cx - w / 3, cy + h / 3, 0.9)
    return k


def _box(kps):
    vis = kps[kps[:, 2] > 0.3]
    xs, ys = vis[:, 0] * MW / W, vis[:, 1] * MH / H
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def _room(level=100) -> np.ndarray:
    return np.full((MH, MW, 3), level, np.uint8)


def _painting_frames():
    caps = sorted(glob.glob(str(Path(__file__).parent / "ghosts" / "*.npz")))
    assert caps, "the painting traces must stay committed — see tests/ghosts/"
    for path in caps:
        for f in np.load(path, allow_pickle=True)["keypoints"]:
            if (f[:, 2] > 0.3).sum() >= 6:
                yield f


def _mask_after(first, second, gap=_MOTION_GAP_S):
    m = _ShortTermMotion()
    assert m.update(first, 0.0) is None
    return m.update(second, gap)


# ── the tracker itself ───────────────────────────────────────────────────────

def test_no_answer_until_there_is_enough_history():
    m = _ShortTermMotion()
    assert m.update(_room(), 0.0) is None
    assert m.update(_room(), _MOTION_GAP_S / 2) is None
    assert m.update(_room(), _MOTION_GAP_S) is not None


def test_compares_against_a_frame_about_the_gap_ago():
    """A change that happened long before the gap is history, not motion."""
    m = _ShortTermMotion()
    m.update(_room(0), 0.0)
    m.update(_room(200), 1.0)
    mask = m.update(_room(200), 1.0 + _MOTION_GAP_S)
    assert not mask.any()


def test_a_static_scene_shows_no_motion():
    assert not _mask_after(_room(), _room()).any()


# ── what confirms and what does not ──────────────────────────────────────────

def test_sunlight_on_the_painting_alone_is_not_movement():
    """The 7 am case. Over half a second the light on the painting barely
    changes, so nothing clears the per-pixel threshold."""
    for f in _painting_frames():
        before, after = _room(), _room()
        x0, x1, y0, y1 = _box(f)
        after[y0:y1, x0:x1] += 3            # a slow brightening, sub-threshold
        assert _body_moved(f, _mask_after(before, after), W, H) is False


def test_the_whole_frame_changing_is_not_movement():
    """An exposure step changes everything at once; _stands_out vetoes it."""
    k = _person()
    assert _body_moved(k, _mask_after(_room(80), _room(160)), W, H) is False


def test_a_person_shifting_is_movement():
    k = _person()
    before, after = _room(), _room()
    x0, x1, y0, y1 = _box(k)
    after[y0:y0 + (y1 - y0) // 3, x0:x1] = 180     # head and shoulders moved
    assert _body_moved(k, _mask_after(before, after), W, H) is True


def test_no_motion_history_does_not_confirm():
    """Unlike admission, confirmation does not abstain: 'cannot tell' is no."""
    assert _body_moved(_person(), None, W, H) is False


# ── how it feeds confirmation ─────────────────────────────────────────────────

def _lit_everywhere():
    return np.full((MH, MW), 255, np.uint8)


def _fg_on(kps):
    m = np.zeros((MH, MW), np.uint8)
    x0, x1, y0, y1 = _box(kps)
    m[y0:y1, x0:x1] = 255
    return m


def test_background_difference_alone_no_longer_confirms():
    """The painting's box differs from the background (the sun) but did not
    move: admitted, so tracking is unchanged, but not a confirming pass."""
    k = _person()
    still = np.zeros((MH, MW), np.uint8)
    [(lm, c, raw_live, moved)] = _all_valid_landmarks(
        k[None], W, H, fg_mask=_fg_on(k), motion_mask=still)
    assert raw_live is True      # still tracked through the grace period
    assert moved is False        # but not a confirming pass


def test_a_moving_person_confirms():
    k = _person()
    [(lm, c, raw_live, moved)] = _all_valid_landmarks(
        k[None], W, H, fg_mask=_fg_on(k), motion_mask=_fg_on(k))
    assert moved is True


def test_without_a_motion_mask_the_old_evidence_still_applies():
    k = _person()
    [(lm, c, raw_live, moved)] = _all_valid_landmarks(k[None], W, H, fg_mask=_fg_on(k))
    assert raw_live is True and moved is True


# ── tracking and confirmation are separate jobs ──────────────────────────────
# The first version made movement do both, and someone sitting still on the
# couch lost their track every few seconds: 27 new tracks in 3.5 minutes,
# against 52 in the 20 minutes before. Each new track then had to be
# confirmed again before it could fire.

def _analyser():
    from unittest.mock import MagicMock
    from cameras.analyser import CameraAnalyser
    from config.settings import Settings
    return CameraAnalyser(camera=MagicMock(), settings=Settings(),
                          on_candidate=MagicMock(), label="test")


def test_a_still_person_stays_tracked_but_is_not_confirmed():
    a = _analyser()
    a._match_persons([(None, (400.0, 300.0), True, False)], 640, 480, now=1000.0)
    e = next(iter(a._persons.values()))
    a._match_persons([(None, (400.0, 300.0), True, False)], 640, 480, now=1010.0)
    assert e.last_genuine_live_at == 1010.0     # grace period keeps running
    assert e.confirmed_human is False


def test_a_track_born_still_needs_two_moving_passes():
    """Birth counts as a confirming pass only if the body moved, so the
    painting cannot be confirmed by a single flicker after it appears."""
    a = _analyser()
    a._match_persons([(None, (400.0, 300.0), True, False)], 640, 480, now=1000.0)
    e = next(iter(a._persons.values()))
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=1003.0)
    assert e.confirmed_human is False
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=1006.0)
    assert e.confirmed_human is True


def test_a_person_who_moves_is_confirmed_as_before():
    a = _analyser()
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=1000.0)
    e = next(iter(a._persons.values()))
    a._match_persons([(None, (400.0, 300.0), True, True)], 640, 480, now=1002.5)
    assert e.confirmed_human is True
