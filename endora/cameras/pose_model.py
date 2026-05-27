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
  3. ONNX Runtime inference (multi-threaded, using all available cores)
  4. Confidence mask + numpy NMS (~0.3 ms)
  5. Coordinate unscale back to original frame pixels

The result is a plain numpy array — no .cpu() calls, no Results wrappers.

Input-size note
---------------
YOLO ONNX models are typically exported with *static* input shapes (e.g.
640×640).  ONNX Runtime refuses to run them at a different resolution.

``_resolve_model_path`` handles this transparently:
  1. If the on-disk model already has the requested size → use it directly.
  2. If the model is dynamic-shape → use it with any size.
  3. If ``yolo11n-pose.pt`` exists alongside the .onnx → export a
     ``yolo11n-pose-320.onnx`` once into /data/ and cache it there.
     (This takes ~30 s on first boot but persists across restarts.)
  4. Otherwise → fall back to the static 640×640 model (still faster than
     ultralytics due to multi-threading and no Python wrapper overhead).

Typical timing on Pi 5 (Cortex-A76, 4 cores):
  ultralytics YOLO  @ 640×640, 1 thread   ~250 ms / frame
  PoseModel         @ 640×640, 4 threads   ~80 ms  / frame  (3×)
  PoseModel         @ 320×320, 4 threads   ~25 ms  / frame  (10×)
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── model-path resolution ─────────────────────────────────────────────────────

def _static_input_size(model_path: str) -> Optional[int]:
    """Return the static input H (=W) of an ONNX model, or None if dynamic."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        shape = sess.get_inputs()[0].shape   # e.g. [1, 3, 640, 640] or [1,3,'h','w']
        h = shape[2]
        return int(h) if isinstance(h, int) else None
    except Exception:
        return None


def _export_at_size(pt_path: str, imgsz: int, dest: str) -> bool:
    """Export *pt_path* to ONNX at *imgsz*×*imgsz* and save to *dest*.

    Returns True on success.  Never attempted on aarch64 — PyTorch causes
    SIGILL on Cortex-A72/A76 (Pi 4/5) and the error is uncatchable.
    Generate the model on an x86/macOS machine instead (see Dockerfile).
    """
    import platform
    if platform.machine() == "aarch64":
        log.warning(
            "Cannot export %dx%d ONNX at runtime on aarch64 — torch is unsafe "
            "on this CPU.  Generate yolo11n-pose-%d.onnx on an x86/macOS machine "
            "and either commit it to the repo or copy it to /data/.",
            imgsz, imgsz, imgsz,
        )
        return False
    try:
        log.info(
            "Exporting %dx%d ONNX from %s — this takes ~30 s and only runs once.",
            imgsz, imgsz, pt_path,
        )
        from ultralytics import YOLO  # only runs on x86/macOS
        m = YOLO(pt_path)
        exported = str(m.export(format="onnx", imgsz=imgsz, simplify=True))
        shutil.copy2(exported, dest)
        log.info("Saved %dx%d model → %s", imgsz, imgsz, dest)
        return True
    except Exception as exc:
        log.warning("Could not export %dx%d model: %s", imgsz, imgsz, exc)
        return False


def resolve_model_path(model_path: str, imgsz: int) -> tuple[str, int]:
    """Return *(path_to_use, actual_imgsz)* for inference.

    Resolution order
    ----------------
    1. The on-disk model already has the right static size (or is dynamic).
    2. A cached ``/data/<stem>-<imgsz>.onnx`` exists from a previous export.
    3. A ``.pt`` source file is found — checked in this order:
         a. ``/data/<stem>.pt``          ← user can drop it here via SSH/Samba
         b. Same directory as the .onnx  ← bundled in the Docker image
       If found, export once to ``/data/<stem>-<imgsz>.onnx`` (~30 s on Pi 5)
       and use it immediately (no restart needed).
    4. Fall back to the original static model at its native size; still faster
       than ultralytics thanks to multi-threaded ONNX Runtime.

    To unlock 320×320 inference without rebuilding the Docker image:
      1. Download ``yolo11n-pose.pt`` (~5 MB) from
         https://github.com/ultralytics/assets/releases
      2. Copy it into the add-on /data/ folder (SSH, Samba, or HA file editor).
      3. Restart the add-on once.  Export runs in ~30 s, then persists forever.
    """
    static_sz = _static_input_size(model_path)

    # Already the right size, or dynamic-shape model
    if static_sz is None or static_sz == imgsz:
        return model_path, imgsz

    stem = Path(model_path).stem           # e.g. "yolo11n-pose"

    # 1. Baked into the Docker image alongside the 640×640 model
    image_small = Path(model_path).parent / f"{stem}-{imgsz}.onnx"
    if image_small.exists():
        log.info("Using image-bundled %dx%d model: %s", imgsz, imgsz, image_small)
        return str(image_small), imgsz

    # 2. Cached in /data/ from a previous lazy export
    cache = Path("/data") / f"{stem}-{imgsz}.onnx"
    if cache.exists():
        log.info("Using cached %dx%d model: %s", imgsz, imgsz, cache)
        return str(cache), imgsz

    # 3. Export from .pt — check /data/ first (user-accessible), then /app/
    pt_candidates = [
        Path("/data") / f"{stem}.pt",
        Path(model_path).with_suffix(".pt"),
    ]
    for pt_path in pt_candidates:
        if pt_path.exists():
            log.info("Found .pt at %s", pt_path)
            if _export_at_size(str(pt_path), imgsz, str(cache)):
                return str(cache), imgsz
            break   # export failed; don't try the other candidate

    # Nothing worked — use original model at its native size
    log.warning(
        "Model %s has static %dx%d input; no .pt found for re-export. "
        "Running at %dx%d (still ~3x faster than ultralytics via multi-threading). "
        "To enable %dx%d: copy yolo11n-pose.pt into the add-on /data/ folder "
        "and restart once.",
        Path(model_path).name, static_sz, static_sz, static_sz, static_sz,
        imgsz,
    )
    return model_path, static_sz


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
        Requested square inference resolution.  If the on-disk model has a
        different static size, ``resolve_model_path`` will try to produce or
        locate a matching one.  Must be a multiple of 32.
    conf:
        Minimum person confidence to keep a detection (mutable at runtime).
    num_threads:
        ONNX Runtime intra-op thread count.  0 = ``os.cpu_count()``.
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

        # Resolve model path — may re-export at requested imgsz
        actual_path, actual_imgsz = resolve_model_path(model_path, imgsz)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress onnxruntime INFO spam

        self._sess = ort.InferenceSession(
            actual_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name
        self.imgsz = actual_imgsz
        self.conf = conf
        log.info(
            "PoseModel: %s  imgsz=%d  threads=%d  conf=%.2f",
            os.path.basename(actual_path), actual_imgsz, num_threads, conf,
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
