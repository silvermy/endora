"""
cameras/dewarp.py

Fisheye-to-perspective dewarping for equidistant fisheye lenses.

Equidistant projection model:  r = f * θ
  r   = pixel distance from fisheye image centre
  f   = focal length in pixels  = image_radius / (fov_rad / 2)
  θ   = angle from the optical axis (radians)

Usage
-----
Build the remap tables once at startup (lazy-init on first frame):

    from cameras.dewarp import build_dewarp_maps, apply_dewarp

    map_x, map_y = build_dewarp_maps(
        in_w=1280, in_h=960,
        out_w=1280,  out_h=480,
        fisheye_fov_deg=180.0,
        pan_deg=0.0,    # + = right,  - = left
        tilt_deg=30.0,  # + = DOWN toward floor,  - = upward
        roll_deg=0.0,   # + = clockwise,  - = counter-clockwise (levels horizon)
        vfov_deg=50.0,  # vertical FOV of output; horizontal FOV grows with out_w
    )

    flat = apply_dewarp(fisheye_frame, map_x, map_y)

Per-frame cost is a single cv2.remap() — effectively free.

Sign conventions
----------------
  tilt_deg  + = camera looks DOWN toward the floor
            - = camera looks UP toward the ceiling
  pan_deg   + = camera looks RIGHT
            - = camera looks LEFT
  roll_deg  + = image rotates clockwise (scene appears to lean left)
            - = image rotates counter-clockwise (use this to level a CCW-leaning scene)

Horizontal FOV
--------------
Horizontal FOV is derived from vfov_deg and the output aspect ratio:
    hfov/2 = arctan( (out_w/out_h) * tan(vfov/2) )
So to widen the view without changing vertical FOV, increase out_w.
  out_w=640,  out_h=480, vfov=50° → hfov ≈ 64°
  out_w=1280, out_h=480, vfov=50° → hfov ≈ 102°
  out_w=1920, out_h=480, vfov=50° → hfov ≈ 124°

RTSP note
---------
Use the RAW fisheye RTSP stream (no in-camera dewarping).
Set the Reolink camera to "Fisheye" mode (not "Defisheye") in the
Reolink app so the RTSP output is the untouched fisheye circle.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def build_dewarp_maps(
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    fisheye_fov_deg: float = 180.0,
    pan_deg: float = 0.0,
    tilt_deg: float = 0.0,
    roll_deg: float = 0.0,
    vfov_deg: float = 70.0,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cv2.remap() source maps for a virtual perspective camera pointed
    into a fisheye image.

    Parameters
    ----------
    in_w, in_h        Input (fisheye) frame dimensions in pixels.
    out_w, out_h      Output (flat perspective) frame dimensions.
                      Wider out_w = wider horizontal FOV with same vfov.
    fisheye_fov_deg   Total FOV of the fisheye lens (180 = hemisphere).
    pan_deg           Horizontal pan: + = right, - = left.
    tilt_deg          Vertical tilt:  + = DOWN toward floor, - = upward.
    roll_deg          Image roll to level a tilted horizon.
                      + = clockwise, - = counter-clockwise.
                      If the scene leans to the right, use a negative value.
    vfov_deg          Vertical FOV of the output image.
    cx, cy            Fisheye circle centre (-1 / None = frame geometric centre).

    Returns
    -------
    (map_x, map_y)  float32 arrays shape (out_h, out_w) for cv2.remap().
    """
    if cx is None:
        cx = in_w / 2.0
    if cy is None:
        cy = in_h / 2.0

    # ── Fisheye focal length (equidistant model) ──────────────────────────
    fisheye_radius = min(in_w, in_h) / 2.0
    fov_rad = np.deg2rad(fisheye_fov_deg)
    f_fish = fisheye_radius / (fov_rad / 2.0)

    # ── Virtual camera focal length (from vertical FOV + output height) ───
    f_virt = (out_h / 2.0) / np.tan(np.deg2rad(vfov_deg) / 2.0)

    # ── Rotation matrices ─────────────────────────────────────────────────
    # Convention: X right, Y down (image), Z forward.
    # Positive tilt = rotate around X so Z tips downward = camera looks down.
    # Positive pan  = rotate around Y so Z tips right    = camera looks right.
    # Positive roll = rotate around Z clockwise.
    pan_r  = np.deg2rad(pan_deg)
    tilt_r = np.deg2rad(-tilt_deg)   # negate: positive tilt_deg = downward
    roll_r = np.deg2rad(roll_deg)

    Ry = np.array([                         # pan around Y
        [ np.cos(pan_r),  0.0, np.sin(pan_r)],
        [ 0.0,            1.0, 0.0          ],
        [-np.sin(pan_r),  0.0, np.cos(pan_r)],
    ], dtype=np.float64)

    Rx = np.array([                         # tilt around X
        [1.0,  0.0,             0.0            ],
        [0.0,  np.cos(tilt_r), -np.sin(tilt_r)],
        [0.0,  np.sin(tilt_r),  np.cos(tilt_r)],
    ], dtype=np.float64)

    Rz = np.array([                         # roll around Z
        [ np.cos(roll_r), -np.sin(roll_r), 0.0],
        [ np.sin(roll_r),  np.cos(roll_r), 0.0],
        [ 0.0,             0.0,            1.0],
    ], dtype=np.float64)

    R = Rx @ Ry @ Rz    # tilt, then pan, then roll

    # ── Output pixel grid ─────────────────────────────────────────────────
    us = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    vs = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    ug, vg = np.meshgrid(us, vs)            # shape (out_h, out_w)

    # Virtual camera ray directions (normalised)
    rays  = np.stack([ug, vg, np.full_like(ug, f_virt)], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays  = rays / norms

    # ── Rotate into the fisheye frame ─────────────────────────────────────
    flat    = rays.reshape(-1, 3)
    rotated = (R @ flat.T).T.reshape(out_h, out_w, 3)

    dx = rotated[..., 0]
    dy = rotated[..., 1]
    dz = rotated[..., 2]

    # ── Equidistant back-projection → fisheye pixel coords ───────────────
    theta = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    phi   = np.arctan2(dy, dx)

    r     = f_fish * theta
    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)

    hfov = np.degrees(2 * np.arctan((out_w / out_h) * np.tan(np.deg2rad(vfov_deg) / 2.0)))
    log.info(
        "Dewarp maps: %dx%d → %dx%d  "
        "pan=%.1f° tilt=%.1f° roll=%.1f° vfov=%.0f° hfov=%.0f°",
        in_w, in_h, out_w, out_h,
        pan_deg, tilt_deg, roll_deg, vfov_deg, hfov,
    )

    return map_x, map_y


def apply_dewarp(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Apply precomputed dewarp maps to a frame (single cv2.remap call)."""
    return cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
