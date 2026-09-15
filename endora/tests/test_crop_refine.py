"""
tests/test_crop_refine.py

Covers refine_by_crop — the policy layer of crop-then-pose refinement.

The point of the feature is that a person's on-model resolution is set by
how much of the *frame* they occupy, not by the camera: a full-frame pass
scales everything into one square input, so someone at the far end of the
room lands on a few dozen rows. That is the regime where a pose model stops
finding an elbow and starts placing it on the shoulder-wrist line, which the
arm tracker then has to reject as a collapsed segment (v1.9.151).

refine_by_crop takes the inference function as an argument rather than
reaching for a model, so every decision it makes — who gets refined, in what
order, how many, and how crop coordinates come back to frame space — is
testable here with no onnxruntime, no GPU and no camera.
"""
import numpy as np

from cameras.pose_model import _expand_box, _should_refine_box, refine_by_crop


def _box(x1, y1, x2, y2, conf=0.9):
    return np.array([x1, y1, x2, y2, conf], dtype=np.float32)


def _kps(value=0.0):
    """17 keypoints all at (value, value) with full visibility."""
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 0] = value
    k[:, 1] = value
    k[:, 2] = 1.0
    return k


def _frame(w=1280, h=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── _expand_box ───────────────────────────────────────────────────────────────

def test_expand_box_grows_by_margin():
    x1, y1, x2, y2 = _expand_box(_box(100, 100, 200, 300), 0.10, 1280, 640)
    # 100 wide → 10px each side; 200 tall → 20px each side.
    assert (x1, y1, x2, y2) == (90, 80, 210, 320)


def test_expand_box_clamps_to_frame():
    """A person at the frame edge must not produce a negative-origin crop."""
    x1, y1, x2, y2 = _expand_box(_box(5, 5, 105, 105), 0.5, 1280, 640)
    assert x1 == 0 and y1 == 0
    assert x2 <= 1280 and y2 <= 640


def test_expand_box_clamps_at_far_edge():
    x1, y1, x2, y2 = _expand_box(_box(1200, 600, 1275, 638), 0.5, 1280, 640)
    assert x2 == 1280 and y2 == 640


# ── _should_refine_box ────────────────────────────────────────────────────────

def test_small_person_is_refined():
    assert _should_refine_box(_box(0, 0, 60, 120), 1280, 640, 0.55) is True


def test_large_person_is_not_refined():
    """Someone filling the frame already has every model pixel available."""
    assert _should_refine_box(_box(0, 0, 900, 640), 1280, 640, 0.55) is False


def test_refine_threshold_uses_the_long_edge():
    # 700px tall in a 1280-long-edge frame: 700 >= 0.55 * 1280 (704) is false,
    # so it still refines — the boundary is on the long edge, not the height.
    assert _should_refine_box(_box(0, 0, 100, 700), 1280, 640, 0.55) is True
    assert _should_refine_box(_box(0, 0, 100, 710), 1280, 640, 0.55) is False


# ── refine_by_crop ────────────────────────────────────────────────────────────

def test_none_input_passes_through():
    kps, boxes, n = refine_by_crop(
        _frame(), None, None, lambda f: (None, None),
        margin=0.15, max_persons=2, min_box_frac=0.55)
    assert kps is None and boxes is None and n == 0


def test_refined_keypoints_return_in_frame_coordinates():
    """The crop's local coordinates must be offset back by the crop origin."""
    frame = _frame()
    boxes = np.stack([_box(200, 100, 260, 220)])
    kps = np.stack([_kps(0.0)])

    # The crop pass reports a keypoint 5px inside its own top-left corner.
    def fake_infer(crop):
        return np.stack([_kps(5.0)]), np.stack([_box(0, 0, crop.shape[1], crop.shape[0])])

    out_kps, out_boxes, n = refine_by_crop(
        frame, kps, boxes, fake_infer,
        margin=0.0, max_persons=2, min_box_frac=0.55)

    assert n == 1
    # Crop origin is the box corner (margin 0), so 5 → 205 / 105.
    assert out_kps[0, 0, 0] == 205.0
    assert out_kps[0, 0, 1] == 105.0
    # Boxes are the detector's own answer and must survive untouched.
    assert np.array_equal(out_boxes, boxes)


def test_large_person_is_skipped_entirely():
    calls = []

    def fake_infer(crop):
        calls.append(crop.shape)
        return np.stack([_kps(1.0)]), np.stack([_box(0, 0, 10, 10)])

    boxes = np.stack([_box(0, 0, 1000, 640)])
    _, _, n = refine_by_crop(
        _frame(), np.stack([_kps()]), boxes, fake_infer,
        margin=0.15, max_persons=2, min_box_frac=0.55)

    assert n == 0
    assert calls == [], "a well-resolved person must not cost a second inference"


def test_max_persons_caps_the_work():
    """The budget is a latency guarantee: four small people, two inferences."""
    calls = []

    def fake_infer(crop):
        calls.append(1)
        return np.stack([_kps(2.0)]), np.stack([_box(0, 0, 10, 10)])

    boxes = np.stack([_box(i * 100, 0, i * 100 + 50, 100) for i in range(4)])
    kps = np.stack([_kps() for _ in range(4)])

    _, _, n = refine_by_crop(
        _frame(), kps, boxes, fake_infer,
        margin=0.15, max_persons=2, min_box_frac=0.55)

    assert n == 2
    assert len(calls) == 2


def test_smallest_person_is_refined_first():
    """The budget should go to whoever pass 1 resolved worst."""
    seen = []

    def fake_infer(crop):
        seen.append(crop.shape[1])          # crop width
        return np.stack([_kps(1.0)]), np.stack([_box(0, 0, 10, 10)])

    # Widths 200, 40, 120 — smallest (40) must be cropped first.
    boxes = np.stack([
        _box(0, 0, 200, 300),
        _box(400, 0, 440, 80),
        _box(700, 0, 820, 200),
    ])
    kps = np.stack([_kps() for _ in range(3)])

    refine_by_crop(_frame(), kps, boxes, fake_infer,
                   margin=0.0, max_persons=1, min_box_frac=0.55)

    assert seen == [40]


def test_empty_crop_detection_keeps_pass_one_keypoints():
    """No detection in the crop must not delete the person."""
    boxes = np.stack([_box(100, 100, 160, 220)])
    kps = np.stack([_kps(7.0)])

    out_kps, _, n = refine_by_crop(
        _frame(), kps, boxes, lambda crop: (None, None),
        margin=0.15, max_persons=2, min_box_frac=0.55)

    assert n == 0
    assert np.array_equal(out_kps, kps)


def test_degenerate_box_is_not_cropped():
    calls = []

    def fake_infer(crop):
        calls.append(1)
        return np.stack([_kps()]), np.stack([_box(0, 0, 1, 1)])

    boxes = np.stack([_box(10, 10, 14, 14)])       # 4px — below _MIN_CROP_PX
    refine_by_crop(_frame(), np.stack([_kps()]), boxes, fake_infer,
                   margin=0.0, max_persons=2, min_box_frac=0.55)

    assert calls == []


def test_biggest_detection_in_the_crop_wins():
    """A bystander clipped at the crop edge must not hijack the refinement."""
    frame = _frame()
    boxes = np.stack([_box(300, 100, 360, 220)])

    def fake_infer(crop):
        small = _kps(1.0)        # bystander slice at the edge
        big = _kps(9.0)          # the person the crop was cut around
        return (
            np.stack([small, big]),
            np.stack([_box(0, 0, 5, 5), _box(0, 0, 60, 120)]),
        )

    out_kps, _, n = refine_by_crop(
        frame, np.stack([_kps()]), boxes, fake_infer,
        margin=0.0, max_persons=2, min_box_frac=0.55)

    assert n == 1
    assert out_kps[0, 0, 0] == 300.0 + 9.0
