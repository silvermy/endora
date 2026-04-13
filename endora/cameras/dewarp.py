"""
cameras/dewarp.py

Fisheye-to-perspective dewarping for equidistant fisheye lenses.

Equidistant projection model:  r = f * θ
  r   = pixel distance from fisheye image centre
  f   = focal length in pixels  = image_radius / (fov_rad / 2)
  θ   = angle from the optical axis (radians)

Usage
-----
Build the remap tables once at startup (or on first frame):

    from cameras.dewarp import build_dewarp_maps, apply_dewarp

    map_x, map_y = build_dewarp_maps(
        in_w=1280, in_h=960,
        out_w=640,  out_h=480,
        fisheye_fov_deg=180.0,
        pan_deg=0.0,    # + = right,  - = left
        tilt_deg=30.0,  # + = down toward floor
        vfov_deg=75.0,  # virtual camera FOV (wider = more room)
    )

Then per frame:

    flat = apply_dewarp(fisheye_frame, map_x, map_y)

The per-frame cost is a single cv2.remap() call — effectively free.

RTSP note
---------
Use the RAW fisheye stream from your camera (no in-camera dewarping).
For Reolink FE cameras the main stream is typically:
    rtsp://<user>:<pass>@<ip>:554/h264Preview_01_main
Set the camera's "video format" to "Fisheye" (not "Defisheye") in the
Reolink app or web UI so the RTSP output is the untouched circle.
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
    vfov_deg: float = 70.0,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cv2.remap() source maps for a virtual perspective camera pointed
    into a fisheye image.

    Parameters
    ----------
    in_w, in_h        Input (fisheye) frame size in pixels.
    out_w, out_h      Output (flat perspective) frame size in pixels.
    fisheye_fov_deg   Total field of view of the fisheye lens.
                      180 = hemisphere; 360 = full sphere (rare).
    pan_deg           Virtual camera horizontal pan.
                      0 = straight ahead; + = right; - = left.
    tilt_deg          Virtual camera vertical tilt.
                      0 = horizontal; + = downward; - = upward.
    vfov_deg          Vertical field of view of the output image.
                      70–90 is typical. Larger = more room visible
                      but more perspective distortion at the edges.
    cx, cy            Fisheye circle centre in the input image.
                      Defaults to the geometric centre of the frame.

    Returns
    -------
    (map_x, map_y)  float32 arrays of shape (out_h, out_w) for cv2.remap().
    """
    if cx is None:
        cx = in_w / 2.0
    if cy is None:
        cy = in_h / 2.0

    # ── Fisheye focal length (equidistant model) ──────────────────────────
    # The fisheye "circle" has radius = min(in_w, in_h) / 2.
    # That radius corresponds to half the total FOV angle.
    fisheye_radius = min(in_w, in_h) / 2.0
    fov_rad = np.deg2rad(fisheye_fov_deg)
    f_fish = fisheye_radius / (fov_rad / 2.0)

    # ── Virtual camera focal length ────────────────────────────────────────
    # Derived from the desired output vertical FOV.
    f_virt = (out_h / 2.0) / np.tan(np.deg2rad(vfov_deg) / 2.0)

    # ── Rotation matrix: pan around Y, then tilt around X ─────────────────
    pan_r  = np.deg2rad(pan_deg)
    tilt_r = np.deg2rad(tilt_deg)

    Ry = np.array([
        [ np.cos(pan_r),  0.0, np.sin(pan_r)],
        [ 0.0,            1.0, 0.0          ],
        [-np.sin(pan_r),  0.0, np.cos(pan_r)],
    ], dtype=np.float64)

    Rx = np.array([
        [1.0,  0.0,             0.0            ],
        [0.0,  np.cos(tilt_r), -np.sin(tilt_r)],
        [0.0,  np.sin(tilt_r),  np.cos(tilt_r)],
    ], dtype=np.float64)

    R = Rx @ Ry   # combined rotation

    # ── Build output pixel grid ────────────────────────────────────────────
    us = np.arange(out_w, dtype=np.float64) - out_w / 2.0
    vs = np.arange(out_h, dtype=np.float64) - out_h / 2.0
    ug, vg = np.meshgrid(us, vs)   # shape (out_h, out_w)

    # Virtual camera rays (normalised 3-D direction vectors)
    zg   = np.full_like(ug, f_virt)
    rays = np.stack([ug, vg, zg], axis=-1)          # (out_h, out_w, 3)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays  = rays / norms

    # ── Rotate rays into the fisheye frame ───────────────────────────────
    flat    = rays.reshape(-1, 3)           # (N, 3)
    rotated = (R @ flat.T).T.reshape(out_h, out_w, 3)

    dx = rotated[..., 0]
    dy = rotated[..., 1]
    dz = rotated[..., 2]

    # ── Equidistant back-projection ───────────────────────────────────────
    # θ = angle from the optical axis
    theta = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    phi   = np.arctan2(dy, dx)             # azimuth

    r     = f_fish * theta                 # pixel radius in fisheye image
    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)

    log.debug(
        "Dewarp maps: input=%dx%d fisheye_f=%.1fpx "
        "virtual_f=%.1fpx pan=%.1f° tilt=%.1f° vfov=%.1f°",
        in_w, in_h, f_fish, f_virt, pan_deg, tilt_deg, vfov_deg,
    )

    return map_x, map_y


def apply_dewarp(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """
    Apply precomputed dewarp maps to a single frame.

    Parameters
    ----------
    frame           Raw fisheye frame (BGR, any dtype).
    map_x, map_y   Source coordinate maps from build_dewarp_maps().

    Returns
    -------
    Flat perspective frame (same dtype as input).
    """
    return cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
