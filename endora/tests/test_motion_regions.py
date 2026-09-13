"""
tests/test_motion_regions.py

Waking the pose model for a gesture.

Both motion statistics divide by the area they are measured over, so an arm
sweep that is decisive evidence about the person it belongs to is a rounding
error against a whole living room. That denominator is why the gate stayed
shut through real gestures (v1.9.141 fixed one scale of it; this is the
next). Measuring inside the last known person boxes fixes it without
lowering the threshold for the rest of the frame, which is what would let
noise back in.

Also covers the background mask now running downscaled — it cost 68 ms a
frame at full size on the deployed host, a third of the entire loop.
"""
import numpy as np

from cameras.analyser import (
    _frame_has_motion, _boxes_to_thumb, _wrist_shows_motion,
    _MOTION_THUMB_W, _MOTION_THUMB_H, _COCO_RIGHT_WRIST,
)

THRESH, AREA_MIN = 0.015, 0.002          # the shipped defaults
FRAME_W, FRAME_H = 1280, 640


def _thumbs(changed_cells, box=(30, 10, 42, 38)):
    """A before/after thumbnail pair whose only change is *changed_cells*
    cells inside *box* — an arm moving within one seated person.
    """
    prev = np.full((_MOTION_THUMB_H, _MOTION_THUMB_W), 100, np.uint8)
    cur = prev.copy()
    x1, y1, x2, y2 = box
    placed = 0
    for y in range(y1, y2):
        for x in range(x1, x2):
            if placed >= changed_cells:
                break
            cur[y, x] = 160          # a 60-level change, well over the delta
            placed += 1
        if placed >= changed_cells:
            break
    return prev, cur


def test_limb_sized_motion_is_missed_by_the_whole_frame_test():
    # The failure being fixed: eight changed cells out of 4800 is 0.0017,
    # under the 0.002 area threshold, so the frame-wide test sleeps through
    # a moving arm.
    prev, cur = _thumbs(8)
    assert not _frame_has_motion(prev, cur, THRESH, AREA_MIN)


def test_same_motion_wakes_the_gate_inside_a_person_box():
    prev, cur = _thumbs(8)
    regions = [(30, 10, 42, 38)]
    assert _frame_has_motion(prev, cur, THRESH, AREA_MIN, regions=regions)


def test_regions_do_not_make_a_still_scene_move():
    # Sensor noise must not clear the bar just because the denominator shrank.
    rng = np.random.default_rng(0)
    prev = np.full((_MOTION_THUMB_H, _MOTION_THUMB_W), 100, np.uint8)
    cur = np.clip(prev.astype(np.int16) + rng.integers(-6, 7, prev.shape), 0, 255).astype(np.uint8)
    assert not _frame_has_motion(prev, cur, THRESH, AREA_MIN,
                                 regions=[(30, 10, 42, 38)])


def test_motion_outside_every_region_still_wakes_the_gate():
    # Somebody walking in where nobody is tracked yet: the full-frame test
    # runs first and must still fire.
    prev, cur = _thumbs(400, box=(2, 2, 40, 40))
    assert _frame_has_motion(prev, cur, THRESH, AREA_MIN, regions=[(60, 40, 70, 50)])


def test_first_frame_always_counts_as_motion():
    _, cur = _thumbs(8)
    assert _frame_has_motion(None, cur, THRESH, AREA_MIN, regions=[])


# ── box → thumbnail mapping ───────────────────────────────────────────────────

def test_boxes_are_scaled_into_thumbnail_space():
    box = np.array([640, 320, 1280, 640, 0.9], np.float32)   # bottom-right quarter
    (x1, y1, x2, y2), = _boxes_to_thumb([box], FRAME_W, FRAME_H)
    assert (x1, y1) == (_MOTION_THUMB_W // 2, _MOTION_THUMB_H // 2)
    assert (x2, y2) == (_MOTION_THUMB_W, _MOTION_THUMB_H)


def test_tiny_boxes_are_dropped():
    # A box a few cells across carries no usable statistics, only noise.
    box = np.array([0, 0, 8, 8, 0.9], np.float32)
    assert _boxes_to_thumb([box], FRAME_W, FRAME_H) == []


def test_box_coordinates_are_clamped_to_the_thumbnail():
    box = np.array([-50, -50, FRAME_W + 500, FRAME_H + 500, 0.9], np.float32)
    (x1, y1, x2, y2), = _boxes_to_thumb([box], FRAME_W, FRAME_H)
    assert (x1, y1, x2, y2) == (0, 0, _MOTION_THUMB_W, _MOTION_THUMB_H)


# ── downscaled background mask ────────────────────────────────────────────────

def _row_with_wrist(xy):
    row = np.zeros((17, 3), np.float32)
    row[_COCO_RIGHT_WRIST] = (*xy, 0.9)
    return row


def _mask_with_blob(w, h, centre_frac, half_frac=0.06):
    m = np.zeros((h, w), np.uint8)
    cx, cy = int(centre_frac[0] * w), int(centre_frac[1] * h)
    hx, hy = int(half_frac * w), int(half_frac * h)
    m[max(0, cy - hy): cy + hy, max(0, cx - hx): cx + hx] = 255
    return m


def test_downscaled_mask_gives_the_same_verdict_as_a_full_size_one():
    # The point of the optimisation: a 4x smaller mask must not change who
    # counts as alive. Wrist over a moving blob → live, in both.
    wrist = (0.5 * FRAME_W, 0.5 * FRAME_H)
    row = _row_with_wrist(wrist)
    full = _mask_with_blob(FRAME_W, FRAME_H, (0.5, 0.5))
    small = _mask_with_blob(320, 160, (0.5, 0.5))
    assert _wrist_shows_motion(row, full, FRAME_W, FRAME_H, 0.12)
    assert _wrist_shows_motion(row, small, FRAME_W, FRAME_H, 0.12)


def test_downscaled_mask_still_rejects_a_wrist_over_static_pixels():
    # Regression guard for the scaling itself: before keypoints were scaled
    # into mask space, a frame-pixel x of 640 indexed column 640 of a
    # 320-wide mask — clamped to the edge, so every wrist was judged against
    # the wrong part of the room.
    row = _row_with_wrist((0.5 * FRAME_W, 0.5 * FRAME_H))
    small = _mask_with_blob(320, 160, (0.9, 0.9))      # movement far away
    assert not _wrist_shows_motion(row, small, FRAME_W, FRAME_H, 0.12)
