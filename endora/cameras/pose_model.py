"""
cameras/pose_model.py

YOLO11n-pose inference via direct ONNX Runtime.

Why not ultralytics?
--------------------
When you call ``YOLO(frame)``, ultralytics wraps the actual ONNX Runtime
session with several Python layers: tensor conversion, letterbox via PIL,
Results objects, per-class NMS in Python loops, etc.  On a Pi 5 this
overhead can exceed the inference itself.

Here we own every step:
  1. Letterbox resize (one cv2.resize + numpy fill, ~0.5 ms)
  2. CHW normalisation (numpy, ~0.2 ms)
  3. ONNX Runtime inference (multi-threaded, on all available cores)
  4. Confidence mask + numpy NMS (~0.3 ms)
  5. Coordinate unscale back to original frame pixels

The result is a plain numpy array — no .cpu() calls, no Results wrappers.

Typical timing on Pi 5 (Cortex-A76, 4 cores):
  ultralytics YOLO @ 640×640  ~250 ms
  PoseModel         @ 320×320  ~25–40 ms  (6–10× faster)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _letterbox(
    img: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *img* to *size*×*size* with grey padding (no stretch).

    Returns *(padded_img, scale_ratio, (pad_left, pad_top))*.
    """
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pw, ph = (size - nw) // 2, (size - nh) // 2
    canvas[ph: ph + nh, pw: pw + nw] = resized
    return canvas, ratio, (pw, ph)


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45
) -> list[int]:
    """Vectorised greedy NMS.  *boxes*: [N, 4] xyxy; *scores*: [N]."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


# ── main class ────────────────────────────────────────────────────────────────

class PoseModel:
    """YOLO11n-pose via direct ONNX Runtime.

    Parameters
    ----------
    model_path:
        Absolute path to ``yolo11n-pose.onnx``.
    imgsz:
        Square inference resolution.  320 uses one-quarter the FLOPs of 640
        and is sufficient for whole-body pose at typical room-camera distances.
        Must be a multiple of 32.
    conf:
        Minimum person confidence to keep a detection (can be changed at
        runtime via ``model.conf = ...``).
    num_threads:
        ONNX Runtime intra-op thread count.  0 = use ``os.cpu_count()``.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        conf: float = 0.45,
        num_threads: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for PoseModel but is not installed. "
                "It should be present as a transitive dependency of ultralytics."
            ) from exc

        if num_threads <= 0:
            num_threads = os.cpu_count() or 4

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress onnxruntime INFO spam

        self._sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        log.info(
            "PoseModel: %s  imgsz=%d  threads=%d  conf=%.2f",
            os.path.basename(model_path), imgsz, num_threads, conf,
        )

    # ── inference ─────────────────────────────────────────────────────────

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run pose inference on *frame* (BGR uint8, any resolution).

        Returns an ndarray of shape **[N, 17, 3]** — N detected persons,
        17 COCO keypoints, each ``[x_px, y_px, visibility]`` in *frame*
        pixel space — or **None** when no person exceeds the confidence
        threshold.

        YOLO11n-pose ONNX output layout (per grid anchor):
            [cx, cy, w, h, person_conf, kp0_x, kp0_y, kp0_v, kp1_x, …]
        All bbox / keypoint coordinates are in *letterboxed-image* pixels.
        """
        fh, fw = frame.shape[:2]
        sz = self.imgsz

        # 1. Letterbox resize
        img, ratio, (pad_w, pad_h) = _letterbox(frame, sz)

        # 2. BGR→RGB, HWC→CHW, normalise to [0, 1]
        x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x[np.newaxis])   # [1, 3, sz, sz]

        # 3. Inference — [1, 56, num_anchors]
        raw = self._sess.run(None, {self._input_name: x})[0]

        # 4. Transpose → [num_anchors, 56], confidence filter
        preds = raw[0].T                           # [num_anchors, 56]
        mask = preds[:, 4] >= self.conf
        preds = preds[mask]
        if preds.shape[0] == 0:
            return None

        # 5. Decode boxes: cx,cy,w,h → x1,y1,x2,y2 (letterboxed pixels)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
        )

        # 6. NMS
        keep = _nms(boxes, preds[:, 4])
        if not keep:
            return None
        preds = preds[keep]

        # 7. Keypoints: [N, 51] → [N, 17, 3], unpad & unscale to frame pixels
        kps = preds[:, 5:].reshape(-1, 17, 3).copy()
        kps[:, :, 0] = np.clip((kps[:, :, 0] - pad_w) / ratio, 0, fw)
        kps[:, :, 1] = np.clip((kps[:, :, 1] - pad_h) / ratio, 0, fh)

        return kps   # [N, 17, 3]
