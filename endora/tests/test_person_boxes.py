"""
tests/test_person_boxes.py

The debug overlay's person bounding boxes.

Two things have to hold for the box to be worth looking at. It must land in
the same pixel space as the skeleton — a box decoded without undoing the
letterbox padding is offset by a fixed amount and looks like a tracking bug
that isn't there. And it must be drawn for detections the gates *rejected*,
because the question the box exists to answer is "did YOLO see nobody, or did
it see someone I then threw away?" — two situations that are pixel-identical
in the skeleton-only view and need opposite fixes.

Runs without a YOLO model: PoseModel.infer is driven with a stub session
returning a hand-built raw tensor.
"""
import numpy as np

from cameras.pose_model import PoseModel
from cameras.analyser import _draw_person_boxes, _draw_debug

FRAME_W, FRAME_H = 1280, 640      # the user's dewarped 2:1 frame
IMGSZ = 640
# letterbox of a 1280x640 frame into 640x640: half scale, 160px top/bottom bars
RATIO, PAD_W, PAD_H = 0.5, 0, 160


class _StubSession:
    """Stands in for an onnxruntime InferenceSession."""

    def __init__(self, raw: np.ndarray) -> None:
        self._raw = raw

    def run(self, _outputs, _feed):
        return [self._raw]


def _model_returning(box_xyxy, conf, kp_xy) -> PoseModel:
    """A PoseModel whose session reports one person at *box_xyxy* (frame
    pixels) with a single visible keypoint at *kp_xy* (frame pixels).
    """
    def to_letterbox(x, y):
        return x * RATIO + PAD_W, y * RATIO + PAD_H

    lx1, ly1 = to_letterbox(box_xyxy[0], box_xyxy[1])
    lx2, ly2 = to_letterbox(box_xyxy[2], box_xyxy[3])

    pred = np.zeros(56, dtype=np.float32)
    pred[0] = (lx1 + lx2) / 2          # cx
    pred[1] = (ly1 + ly2) / 2          # cy
    pred[2] = lx2 - lx1                # w
    pred[3] = ly2 - ly1                # h
    pred[4] = conf
    kx, ky = to_letterbox(*kp_xy)
    pred[5], pred[6], pred[7] = kx, ky, 0.9

    raw = pred.reshape(1, 56, 1)       # [1, 56, num_anchors]

    m = object.__new__(PoseModel)      # bypass __init__: no ONNX file needed
    m._sess = _StubSession(raw)
    m._input_name = "images"
    m.imgsz = IMGSZ
    m.conf = 0.25
    return m


def _blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


# ── decode ────────────────────────────────────────────────────────────────────

def test_box_is_returned_in_frame_pixels():
    model = _model_returning((200, 100, 600, 500), 0.87, (300, 200))
    kps, boxes = model.infer(_blank_frame())

    assert boxes is not None and boxes.shape == (1, 5)
    np.testing.assert_allclose(boxes[0, :4], (200, 100, 600, 500), atol=1.0)
    assert abs(float(boxes[0, 4]) - 0.87) < 1e-5
    # Same space as the keypoints, which is the whole point of drawing both.
    np.testing.assert_allclose(kps[0, 0, :2], (300, 200), atol=1.0)


def test_box_row_matches_its_keypoint_row():
    # The box must describe the person whose skeleton shares its index: the
    # keypoint falls inside the box it is paired with.
    model = _model_returning((200, 100, 600, 500), 0.5, (300, 200))
    kps, boxes = model.infer(_blank_frame())
    x, y = kps[0, 0, :2]
    assert boxes[0, 0] <= x <= boxes[0, 2]
    assert boxes[0, 1] <= y <= boxes[0, 3]


def test_no_detection_returns_no_boxes():
    model = _model_returning((200, 100, 600, 500), 0.87, (300, 200))
    model.conf = 0.99                  # above the stub's confidence
    kps, boxes = model.infer(_blank_frame())
    assert kps is None and boxes is None


def test_call_still_returns_keypoints_only():
    # Every gesture-path caller uses model(frame) and must be unaffected.
    model = _model_returning((200, 100, 600, 500), 0.87, (300, 200))
    out = model(_blank_frame())
    assert isinstance(out, np.ndarray) and out.shape == (1, 17, 3)


# ── overlay ───────────────────────────────────────────────────────────────────

def _box(x1=200, y1=100, x2=600, y2=500, conf=0.8):
    return np.array([x1, y1, x2, y2, conf], dtype=np.float32)


def test_accepted_and_rejected_boxes_are_drawn_differently():
    accepted = _blank_frame()
    _draw_person_boxes(accepted, [(_box(), None)])
    rejected = _blank_frame()
    _draw_person_boxes(rejected, [(_box(), "not live")])

    assert accepted.any(), "accepted box drew nothing"
    assert rejected.any(), "rejected box drew nothing"
    assert not np.array_equal(accepted, rejected), \
        "a rejected detection must not look like an accepted one"


def test_box_at_the_top_edge_still_draws_its_label():
    # Regression guard: the label sits above the box, which is off-frame for
    # anyone detected at the top edge — a standing person in a ceiling-ish
    # view. It must flip inside instead of being silently clipped away.
    img = _blank_frame()
    _draw_person_boxes(img, [(_box(y1=0, y2=300), None)])
    assert img[0:40, 200:400].any(), "label lost off the top of the frame"


def test_rejected_detection_reports_itself_instead_of_no_pose():
    # With no skeletons but a rejected box, the overlay must not claim
    # nothing was detected — that sends debugging in the wrong direction.
    img = _draw_debug(_blank_frame(), [], None, None, None,
                      person_boxes=[(_box(), "not live")])
    assert img.shape == (FRAME_H, FRAME_W, 3)
    # The banner is drawn on a red plate near the top centre.
    banner = img[20:80, FRAME_W // 3: 2 * FRAME_W // 3]
    assert banner.any(), "no banner drawn for a rejected detection"


def test_overlay_without_boxes_is_unchanged():
    # person_boxes is optional; older callers (and the capture path before a
    # YOLO run) pass nothing and must still render.
    img = _draw_debug(_blank_frame(), [], None, None, None)
    assert img.shape == (FRAME_H, FRAME_W, 3)
