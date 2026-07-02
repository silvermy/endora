"""
tests/test_analyser_ghost_rejection.py

Unit tests for the background-subtraction liveness filter in
cameras/analyser.py — rejects YOLO "person" detections (e.g. a framed
picture on the wall) whose wrist sits over pixels the background model
considers static, since a real arm-raise always disturbs the pixels at
the wrist.

Runs without a real camera or YOLO model — builds synthetic COCO keypoint
rows and foreground masks by hand.
"""
import numpy as np

from cameras.analyser import (
    _all_valid_landmarks, _wrist_shows_motion,
    _COCO_LEFT_WRIST, _COCO_RIGHT_WRIST,
)

FRAME_W, FRAME_H = 640, 480


def _coco_row(right_wrist_xy=(500, 100), left_wrist_xy=None) -> np.ndarray:
    """A plausible single-person COCO [17, 3] row: shoulders/elbows/wrists/hips
    visible, everything else low-confidence. Right wrist raised by default —
    same shape as a SNAP-candidate detection.
    """
    row = np.zeros((17, 3), dtype=np.float32)
    row[5]  = (220, 200, 0.9)   # left shoulder
    row[6]  = (420, 200, 0.9)   # right shoulder
    row[7]  = (210, 300, 0.9)   # left elbow
    row[8]  = (460, 150, 0.9)   # right elbow
    row[9]  = (200, 380, 0.9) if left_wrist_xy is None else (*left_wrist_xy, 0.9)
    row[10] = (*right_wrist_xy, 0.9)  # right wrist
    row[11] = (240, 420, 0.9)  # left hip
    row[12] = (400, 420, 0.9)  # right hip
    return row


def _fg_mask_with_blob(cx, cy, radius=40) -> np.ndarray:
    """A foreground mask that's all background (0) except a bright circle."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    y, x = np.ogrid[:FRAME_H, :FRAME_W]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2] = 255
    return mask


def test_wrist_over_foreground_counts_as_live():
    fg = _fg_mask_with_blob(500, 100)
    row = _coco_row(right_wrist_xy=(500, 100))
    assert _wrist_shows_motion(row, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12)


def test_wrist_over_pure_background_is_rejected():
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)  # nothing moving anywhere
    row = _coco_row(right_wrist_xy=(500, 100))
    assert not _wrist_shows_motion(row, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12)


def test_no_background_model_passes_through():
    row = _coco_row(right_wrist_xy=(500, 100))
    assert _wrist_shows_motion(row, None, FRAME_W, FRAME_H, min_foreground_frac=0.12)


def test_no_visible_wrist_is_not_rejected():
    row = _coco_row()
    row[_COCO_LEFT_WRIST, 2] = 0.0
    row[_COCO_RIGHT_WRIST, 2] = 0.0
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    assert _wrist_shows_motion(row, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12)


def test_all_valid_landmarks_drops_static_ghost():
    """A framed-picture-shaped detection with a permanently static wrist
    (e.g. re-detected every frame at the exact same pixels) must not survive
    once a background model shows that region has never changed.
    """
    kps = np.stack([_coco_row(right_wrist_xy=(500, 100))])
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)  # painting never moves
    detected = _all_valid_landmarks(kps, FRAME_W, FRAME_H, fg_mask=fg, min_foreground_frac=0.12)
    assert detected == []


def test_all_valid_landmarks_keeps_moving_person():
    kps = np.stack([_coco_row(right_wrist_xy=(500, 100))])
    fg = _fg_mask_with_blob(500, 100)  # wrist region freshly changed
    detected = _all_valid_landmarks(kps, FRAME_W, FRAME_H, fg_mask=fg, min_foreground_frac=0.12)
    assert len(detected) == 1


def test_all_valid_landmarks_unaffected_when_bg_subtract_disabled():
    kps = np.stack([_coco_row(right_wrist_xy=(500, 100))])
    detected = _all_valid_landmarks(kps, FRAME_W, FRAME_H, fg_mask=None)
    assert len(detected) == 1
