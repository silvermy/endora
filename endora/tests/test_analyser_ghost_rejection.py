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
    _all_valid_landmarks, _wrist_shows_motion, _passes_liveness_gate,
    _person_centroid, _COCO_LEFT_WRIST, _COCO_RIGHT_WRIST,
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


def _shifted_coco_row(dx: float, dy: float) -> np.ndarray:
    """A whole second person-shaped row (all keypoints, not just the wrist),
    offset by (dx, dy) from _coco_row()'s default position — for controlling
    centroid distance between two candidates in the same frame, e.g. a ghost
    in a different part of the room from a real tracked person.
    """
    row = _coco_row()
    for idx in (5, 6, 7, 8, 9, 10, 11, 12):
        row[idx, 0] += dx
        row[idx, 1] += dy
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


# ── Regression: a real, already-tracked person holding still (e.g. typing)
# must not be dropped just because their resting wrist gets absorbed into
# the background model — that is a worse outcome than the ghost detections
# the liveness check exists to filter. See _passes_liveness_gate.

def _match_dist_for(frame_w, frame_h):
    from cameras.analyser import _LIVENESS_EXEMPT_DIST
    return _LIVENESS_EXEMPT_DIST * ((frame_w ** 2 + frame_h ** 2) ** 0.5)


def test_still_person_near_known_centroid_is_exempt_from_liveness_check():
    row = _coco_row(right_wrist_xy=(500, 100))
    centroid = _person_centroid(row)
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)  # wrist reads as pure background
    assert _passes_liveness_gate(
        row, centroid, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12,
        known_centroids=[centroid], match_dist=_match_dist_for(FRAME_W, FRAME_H),
    )


def test_unknown_static_object_is_still_rejected_despite_known_centroids_list():
    row = _coco_row(right_wrist_xy=(500, 100))
    centroid = _person_centroid(row)
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    far_away_known = (5.0, 5.0)  # nowhere near this candidate
    assert not _passes_liveness_gate(
        row, centroid, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12,
        known_centroids=[far_away_known], match_dist=_match_dist_for(FRAME_W, FRAME_H),
    )


def test_ghost_elsewhere_in_room_not_exempted_by_a_real_tracked_person():
    """Regression for a real bug: a ghost (e.g. a framed picture) in one
    corner of the room sitting within _PERSON_MATCH_DIST (30% of the frame
    diagonal — tuned for tracking a person walking across the room between
    frames) of a real tracked person elsewhere was wrongly inheriting that
    person's liveness exemption. The exemption radius must be tight enough
    that only the *same* detection, not merely "somewhere in the same
    room," qualifies.
    """
    real_person = _coco_row(right_wrist_xy=(500, 100))
    real_centroid = _person_centroid(real_person)

    # A separate, permanently static "ghost" whose own centroid is 150px
    # from the real person — inside the old 30%-of-diagonal (240px) radius,
    # outside the correct tight exemption radius (48px for this frame size).
    ghost = _shifted_coco_row(dx=150, dy=0)
    ghost_centroid = _person_centroid(ghost)
    dist = ((ghost_centroid[0] - real_centroid[0]) ** 2
            + (ghost_centroid[1] - real_centroid[1]) ** 2) ** 0.5
    assert 48 < dist < 240, f"test setup assumption violated: dist={dist}"

    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)  # ghost's wrist never moves
    assert not _passes_liveness_gate(
        ghost, ghost_centroid, fg, FRAME_W, FRAME_H, min_foreground_frac=0.12,
        known_centroids=[real_centroid], match_dist=_match_dist_for(FRAME_W, FRAME_H),
    )


def test_all_valid_landmarks_keeps_still_tracked_person():
    """The exact regression: a real person YOLO still detects fine, but who
    has been still long enough that fg_mask reads pure background at their
    wrist, must stay in the result once their centroid is already tracked.
    """
    row = _coco_row(right_wrist_xy=(500, 100))
    centroid = _person_centroid(row)
    kps = np.stack([row])
    fg = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    detected = _all_valid_landmarks(
        kps, FRAME_W, FRAME_H, fg_mask=fg, min_foreground_frac=0.12,
        known_centroids=[centroid], match_dist=_match_dist_for(FRAME_W, FRAME_H),
    )
    assert len(detected) == 1
