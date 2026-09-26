"""
cameras/analyser.py

Thin orchestration layer. Per frame:
  1. Preprocess (dewarp / crop / CLAHE)
  2. Run YOLO Pose → body keypoints for all persons in frame
  3. Run grlib Pipeline → hand landmarks (optional; NoHandDetectedException → None)
  4. Per-person: ArmTracker.classify() → ArmReading
  5. Per-person: GestureStateMachine.tick() → Gesture or None
  6. Debug overlay render

Multiple people are tracked simultaneously: each detected person gets their own
ArmTracker and GestureStateMachine, matched across frames by centroid proximity.
Any person can trigger a gesture.
"""
from __future__ import annotations

import collections
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from version import __version__
from cameras.arm_tracker import (
    ArmState, ArmTracker, ArmTrackerConfig, ReclineWitness, Side,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
)
from cameras.frame_capture import FrameCapture
from core.state_machine import (
    Gesture, GestureStateMachine, StateMachineConfig,
)

log = logging.getLogger(__name__)

# ── COCO → MediaPipe index remap ──────────────────────────────────────────────
# YOLO Pose outputs 17 COCO keypoints; ArmTracker uses MediaPipe PoseLandmark
# indices.  This map translates at read-time so ArmTracker needs no changes.
_COCO_TO_MP: dict[int, int] = {
    0:  0,   # nose
    5:  11,  # left shoulder
    6:  12,  # right shoulder
    7:  13,  # left elbow
    8:  14,  # right elbow
    9:  15,  # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
}

# COCO upper-body skeleton connections (used for debug overlay)
_COCO_UPPER_BODY = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
]


@dataclass
class _KP:
    x: float          # normalised 0-1
    y: float          # normalised 0-1
    visibility: float # keypoint confidence


class _YOLOLandmarks:
    """YOLO COCO keypoints wrapped to match ArmTracker's _Landmarks protocol."""

    def __init__(self, kps: np.ndarray, frame_w: int, frame_h: int) -> None:
        # kps: shape [17, 3] — (x_px, y_px, conf). Kept as-is so the test
        # recorder can save exactly what this person's tracker was given.
        self.kps = kps
        self._pts: dict[int, _KP] = {
            mp_idx: _KP(
                x=float(kps[coco_idx, 0]) / frame_w,
                y=float(kps[coco_idx, 1]) / frame_h,
                visibility=float(kps[coco_idx, 2]),
            )
            for coco_idx, mp_idx in _COCO_TO_MP.items()
        }

    def __getitem__(self, idx: int) -> _KP:
        return self._pts[idx]


def _person_centroid(kps_row: np.ndarray) -> Optional[tuple]:
    """Mean (x, y) pixel position of visible keypoints for one person."""
    vis = kps_row[kps_row[:, 2] > 0.3]
    if len(vis) == 0:
        return None
    return float(vis[:, 0].mean()), float(vis[:, 1].mean())


def _person_visible_kp_count(kps_row: np.ndarray) -> int:
    """Number of keypoints with confidence > 0.3 — proxy for detection quality."""
    return int((kps_row[:, 2] > 0.3).sum())


_MIN_VISIBLE_KPS = 6  # fewer than this → almost certainly not a real person

# Per-cell brightness change (0-255, on the 80x60 motion thumbnail) that counts
# as that cell having genuinely changed rather than having drifted with sensor
# noise. Used by the area half of the motion gate.
_MOTION_CELL_DELTA = 25

# Motion-thumbnail geometry, and the smallest person box worth testing on its
# own (below this the region is a handful of cells and its statistics are
# noise rather than a limb).
_MOTION_THUMB_W, _MOTION_THUMB_H = 80, 60
_MOTION_REGION_MIN_CELLS = 12

# Target width for the background-subtraction frame. MOG2 on the full 1280x640
# canvas measured 68 ms per frame on the deployed host — over a third of the
# entire loop, and spent every iteration for a mask only the pose path reads.
# The mask feeds a wrist patch sized as a fraction of the diagonal and a
# coverage fraction, both scale-free, so a smaller mask answers the same
# question for ~1/16 of the cost.
_BG_SUBTRACT_WIDTH = 320

# Person-pool constants
_PERSON_MATCH_DIST = 0.30  # max centroid displacement (fraction of frame diagonal)
                             # to link a detection to an existing tracked person
                             # ACROSS FRAMES — generous on purpose, so a person
                             # walking across the room between motion-gated
                             # frames is still the same tracked pid. Too wide
                             # to also use for the liveness-check exemption
                             # below (see _LIVENESS_EXEMPT_DIST) — a ghost in
                             # one corner of the room can sit well within 30%
                             # of the frame diagonal from a real person on the
                             # couch, which would wrongly exempt it too.
# A person is dropped after being absent from this many CONSECUTIVE YOLO
# runs — not after a wall-clock timeout. The motion gate makes YOLO's cadence
# wildly variable: on a live install the idle heartbeat ran it every ~2.1 s,
# just over the old 2.0 s timeout, so a single missed detection pruned the
# person. Re-acquiring builds a fresh ArmTracker with empty history, which
# throws away the sweep evidence a flourish depends on — the gesture was then
# unrecognisable through no fault of the geometry.
_PERSON_PRUNE_MISSES = 3

# ...and the mirror-image failure at the other end of the rate range. Counting
# only missed runs assumes runs are slow. On a Jetson sampling at 17.6/s,
# three misses is 0.17 s, so any momentary detection dropout destroyed the
# person and rebuilt them as a new pid — ~0.6 new pids per second, each with
# the empty history a sweep is measured against. Worse, the confidence
# hysteresis drops to the permissive "maintain" threshold whenever anyone is
# tracked, so constant churn kept the door open for junk detections: a 0.25
# box on a reclining person fired a false snap. Require both conditions, so
# the rule is right whether the model runs twice a second or twenty times.
_PERSON_PRUNE_MIN_S = 1.5

# How close a detection must be to an already-tracked person's last position
# to be trusted as "probably that same real person, just briefly still" and
# skip the wrist-liveness check. Deliberately much tighter than
# _PERSON_MATCH_DIST — this is "is this the same detection," not "could this
# plausibly be the same person after they moved."
_LIVENESS_EXEMPT_DIST = 0.06

# A tracked pid only becomes exempt from the wrist-liveness check (i.e. only
# becomes a "known centroid" other candidates can match against) after TWO
# genuine, non-exempted passes land within this many seconds of each other.
# Real motion (walking in, sitting down, gesturing) naturally produces
# clustered genuine passes; a static ghost's rare noise-driven fluke passes
# are isolated and unlikely to land twice within this window. Confirmation
# is sticky once earned — a confirmed person can then rest indefinitely
# without losing it. Without this, a ghost that earned a single lucky pass
# (e.g. once, ever) would exempt itself forever afterward just by matching
# its own unchanging position on every subsequent frame — the bug this
# constant exists to close.
#
# On its own this window still was not enough: real-world lighting noise
# (a light flickering, auto-exposure adjusting, a shadow shifting) does not
# produce independent single-frame flukes — it produces a BURST of several
# consecutive frames that all read as changed together, at whatever cadence
# the underlying light/shadow event lasts. Two such correlated flukes
# landing within any 60s window turned out to be far likelier than two
# independent ones would be, and a real ghost got fully confirmed this way.
# See _CENTROID_MOVED_MIN_FRAC for the fix: a genuine confirming pass now
# also requires the tracked keypoints to have actually moved, which no
# amount of lighting-driven foreground-mask noise can fake, since it comes
# from the pose model's own coordinate output, not the background model.
#
# This same window also bounds the pre-confirmation grace period (see
# _known_centroids): a pid gets to self-exempt for up to this long after
# its last genuine pass even before earning full confirmation, so a real
# person who's still settling in isn't pruned the instant they hold still.
# A static ghost gets the identical window, not a longer one — it's the
# same clock either way.
_LIVENESS_CONFIRM_WINDOW_S = 60.0
# …and no less than this far apart. The window was only ever an UPPER bound,
# so two adjacent frames 0.06 s apart satisfied "two genuine passes within
# 60 s" — which is not the sustained evidence the name implies. A painting of
# a figure was confirmed human 0.56 s after its track was born, and fired
# FOLDED_ARMS eleven seconds later (2026-09-26). A real person is in view for
# many seconds, so this costs them nothing.
_LIVENESS_CONFIRM_MIN_GAP_S = 2.0

# Minimum centroid displacement (fraction of frame diagonal) between this
# detection and this pid's previous one to count as "actually moved" for
# confirmation purposes (see _LIVENESS_CONFIRM_WINDOW_S). A static object's
# keypoints come out at virtually the same pixel coordinates on every
# detection — small residual jitter is sensor/inference noise, not motion.
# Real human movement (a raise, shifting position, walking) displaces the
# centroid by far more than this. Deliberately smaller than
# _LIVENESS_EXEMPT_DIST — that constant asks "is this the same detection,"
# this one asks "did that same detection actually move."
_CENTROID_MOVED_MIN_FRAC = 0.02

# Raw COCO keypoint indices (before the MediaPipe remap below) for the wrists —
# used by the background-subtraction liveness check.
_COCO_LEFT_WRIST  = 9
_COCO_RIGHT_WRIST = 10
_WRIST_PATCH_FRAC = 0.04  # half-width of the wrist foreground-check patch,
                           # as a fraction of the frame diagonal


def _wrist_shows_motion(
    kps_row: np.ndarray,
    fg_mask: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    min_foreground_frac: float,
) -> bool:
    """True if at least one visible wrist sits over recently-changed (foreground)
    pixels — used to reject static objects (e.g. a framed picture containing a
    person) that YOLO mis-detects as a permanently "raised arm" but which never
    actually moves. Returns True (don't reject) when there's nothing to check
    against — no background model yet, or no confidently-visible wrist.

    The mask may be smaller than the frame (the background model runs
    downscaled — it cost 68 ms a frame at full size, more than a third of the
    whole loop), so keypoints are scaled into mask space rather than indexed
    directly. The patch is a fraction of the diagonal either way, so what it
    covers of the person is unchanged.
    """
    if fg_mask is None:
        return True
    mh, mw = fg_mask.shape[:2]
    sx, sy = mw / max(1, frame_w), mh / max(1, frame_h)
    diag = ((mw) ** 2 + (mh) ** 2) ** 0.5
    half = max(2, int(_WRIST_PATCH_FRAC * diag))
    seen_wrist = False
    for idx in (_COCO_LEFT_WRIST, _COCO_RIGHT_WRIST):
        if kps_row[idx, 2] <= 0.3:
            continue
        seen_wrist = True
        x, y = kps_row[idx, 0] * sx, kps_row[idx, 1] * sy
        x0, x1 = max(0, int(x - half)), min(mw, int(x + half))
        y0, y1 = max(0, int(y - half)), min(mh, int(y + half))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = fg_mask[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if float((patch > 0).mean()) >= min_foreground_frac:
            return True
    return not seen_wrist  # no visible wrist → can't judge, don't reject


def _passes_liveness_gate(
    kps_row: np.ndarray,
    centroid: tuple,
    fg_mask: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    min_foreground_frac: float,
    known_centroids: Optional[list] = None,
    match_dist: float = 0.0,
) -> bool:
    """True if this candidate may be treated as a real person this frame.

    known_centroids (each an (x, y) pixel tuple) must come from
    _known_centroids — confirmed-human persons' last positions
    (_PersonEntry.confirmed_human — see _LIVENESS_CONFIRM_WINDOW_S) plus
    pids still inside their own confirmation grace period — not every
    currently-tracked pid unconditionally. A candidate near one is exempt
    from the wrist-liveness check — the same acquire-strict/maintain-lenient
    asymmetry already used for YOLO confidence in _run(). Without this, a
    real person who holds still for a while (e.g. typing, or just having
    sat down and not yet earned their second genuine pass) gets silently
    dropped — both from tracking and from the debug overlay — the moment
    their resting wrist gets absorbed into the background model, which is
    a worse outcome than the ghost detections this check exists to filter.
    Passing every tracked pid's centroid here unconditionally (skipping
    _known_centroids' grace-period bound) is a bug, not a stricter variant:
    an unconfirmed ghost would then trivially exempt itself indefinitely by
    matching its own unchanging position every frame.
    """
    near_known = known_centroids is not None and any(
        (centroid[0] - kc[0]) ** 2 + (centroid[1] - kc[1]) ** 2 <= match_dist ** 2
        for kc in known_centroids
    )
    if near_known:
        return True
    return _wrist_shows_motion(kps_row, fg_mask, frame_w, frame_h, min_foreground_frac)


def _acquire_ok(conf: Optional[float], centroid: tuple,
                tracked_centroids: Optional[list], match_dist: float,
                acquire_conf: float) -> bool:
    """May a detection this weak start a NEW person track?

    The detector runs at the permissive "maintain" confidence so an already
    tracked person survives the dropouts of an arm raise. That relaxation is
    meant for keeping a track you already have — but it was applied to the
    whole frame, so one tracked person lowered the bar for every detection
    anywhere in it, including brand-new ones with nothing to do with them.

    On this camera that is self-sustaining, which is what makes it serious
    rather than untidy: a ghost accepted at the low threshold becomes a
    tracked person, which keeps the roster non-empty, which holds the
    threshold down, which admits the next ghost. Measured live — 1234 pids
    in 40 hours, new tracks appearing on wall art, in the kitchen doorway
    and in an empty corner, one of them confirmed human 0.56 s after birth
    and firing FOLDED_ARMS off a painting of a figure at conf 0.39, against
    a configured yolo_conf of 0.45.

    So a detection that matches something already tracked may be weak; one
    that does not must clear the strict acquire threshold on its own.
    """
    if conf is None or conf >= acquire_conf:
        return True
    if not tracked_centroids:
        return False
    return any((centroid[0] - tc[0]) ** 2 + (centroid[1] - tc[1]) ** 2
               <= match_dist ** 2 for tc in tracked_centroids)


def _all_valid_landmarks(
    kps: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    fg_mask: Optional[np.ndarray] = None,
    min_foreground_frac: float = 0.12,
    known_centroids: Optional[list] = None,
    match_dist: float = 0.0,
    boxes: Optional[np.ndarray] = None,
    tracked_centroids: Optional[list] = None,
    acquire_conf: float = 0.0,
) -> list[tuple]:
    """Return list of (_YOLOLandmarks, centroid_px, raw_live) for every real
    person detected. known_centroids should come from _known_centroids (see
    _passes_liveness_gate), not an unconditional list of every tracked pid.
    raw_live is this candidate's own wrist-liveness result, independent of
    any known-centroid exemption — the caller (_match_persons) uses it to
    decide whether a tracked pid has earned confirmed-human status, which
    feeds back into _known_centroids on the next frame.

    tracked_centroids/acquire_conf gate which detections may start a new
    track rather than merely continue one — see _acquire_ok. They are
    deliberately separate from known_centroids, which is about liveness:
    every tracked pid counts for keeping a weak detection, but only a
    liveness-confirmed one exempts a still body from the wrist check.
    """
    if kps is None or kps.shape[0] == 0:
        return []
    result = []
    for i in range(kps.shape[0]):
        if _person_visible_kp_count(kps[i]) < _MIN_VISIBLE_KPS:
            continue
        c = _person_centroid(kps[i])
        if c is None:
            continue
        conf = (float(boxes[i][4]) if boxes is not None
                and i < len(boxes) else None)
        if not _acquire_ok(conf, c, tracked_centroids, match_dist, acquire_conf):
            continue
        raw_live = _wrist_shows_motion(kps[i], fg_mask, frame_w, frame_h, min_foreground_frac)
        if not raw_live:
            near_known = known_centroids is not None and any(
                (c[0] - kc[0]) ** 2 + (c[1] - kc[1]) ** 2 <= match_dist ** 2
                for kc in known_centroids
            )
            if not near_known:
                continue
        result.append((_YOLOLandmarks(kps[i], frame_w, frame_h), c, raw_live))
    return result


@dataclass
class _PersonEntry:
    """Per-person tracking state: own ArmTracker + GestureStateMachine."""
    arm_tracker:       ArmTracker
    state_machine:     GestureStateMachine
    centroid:          tuple         # last seen pixel centroid (x, y)
    last_seen:         float         # monotonic time of last YOLO detection
    last_seen_run:     int           # YOLO-run index of the last detection
    last_lm:           object        # cached _YOLOLandmarks from last YOLO frame
    last_arm_state:    ArmState      = ArmState.DOWN
    last_logged_state: object        = None
    # Last non-None ArmReading — used to centre the hand-detection crop on
    # the raised wrist (see _crop_around_wrist).
    last_reading:      object        = None
    # True once the chime has sounded for the sweep currently in progress;
    # cleared when the arm settles, so one flourish makes one sound.
    chimed_this_sweep: bool          = False
    # Liveness confirmation — see _LIVENESS_CONFIRM_WINDOW_S. Sticky once
    # True; confirmed_human pids (and pids still inside their own grace
    # period since last_genuine_live_at — see _known_centroids) contribute
    # to known_centroids, which is what exempts a matching candidate from
    # the wrist-liveness check.
    last_genuine_live_at: Optional[float] = None
    # Anchor of the current confirmation window: the FIRST genuine+moved pass
    # of the run in progress. Separate from last_genuine_live_at, which keeps
    # advancing because _known_centroids uses it as a freshness grace period
    # — anchoring on it would make the gap below unreachable.
    first_genuine_live_at: Optional[float] = None
    confirmed_human:       bool          = False


def _known_centroids(
    persons: "dict[int, _PersonEntry]", now: float
) -> list[tuple]:
    """Centroids exempt from the wrist-liveness check this frame.

    Includes confirmed_human pids (sticky, see _LIVENESS_CONFIRM_WINDOW_S)
    plus pids still inside their own confirmation grace period — within
    _LIVENESS_CONFIRM_WINDOW_S of their last genuine pass. Without the
    grace period, a real person who hasn't yet earned their second genuine
    pass (e.g. someone who just sat down and hasn't moved their wrist
    since) fails the liveness check on the very next frame, gets pruned by
    _PERSON_PRUNE_S a couple of seconds later, and has to start over as a
    brand-new pid — resetting the confirmation clock — the next time they
    move. The grace period is bounded, not indefinite: a static ghost gets
    the same window as a real person, not a free pass forever — if it
    never produces a genuine, MOVED pass before the window elapses, it
    drops back out of this list and is pruned exactly as before.
    """
    return [
        e.centroid for e in persons.values()
        if e.confirmed_human or (
            e.last_genuine_live_at is not None
            and now - e.last_genuine_live_at <= _LIVENESS_CONFIRM_WINDOW_S
        )
    ]


# ── Hand-crop helpers ─────────────────────────────────────────────────────────
# MediaPipe Hands (via grlib) detects almost nothing when the hand is a
# couch-distance speck in the full frame — historically snap_roll came back
# nonzero on only ~1 in 15 fires. Cropping a box around the known raised
# wrist and upscaling it gives the hand model a hand-sized hand.

_HAND_CROP_MIN_HALF_PX = 40    # never crop tighter than this half-size
_HAND_CROP_FOREARM_FACTOR = 1.6  # crop half-size as a multiple of forearm length
_HAND_CROP_UPSCALE_PX = 256    # upscale small crops to this square size


def _crop_around_wrist(frame: np.ndarray, reading, lm) -> Optional[np.ndarray]:
    """Square crop of *frame* centred just above the raised wrist (the hand
    sits above the wrist when the arm is up). Sized from the forearm length
    so it tracks person scale/distance. Returns None when the crop cannot be
    built (missing side, degenerate box) — caller falls back to the full frame.
    """
    h, w = frame.shape[:2]
    side = getattr(reading, 'raised_side', None)
    if side is None or lm is None:
        return None
    try:
        e_idx, w_idx = ((LEFT_ELBOW, LEFT_WRIST) if side is Side.LEFT
                        else (RIGHT_ELBOW, RIGHT_WRIST))
        el, wr = lm[e_idx], lm[w_idx]
        forearm_px = (((el.x - wr.x) * w) ** 2 + ((el.y - wr.y) * h) ** 2) ** 0.5
    except Exception:
        return None
    half = int(max(_HAND_CROP_MIN_HALF_PX, _HAND_CROP_FOREARM_FACTOR * forearm_px))
    half = min(half, int(0.45 * min(h, w)))
    cx = int(reading.wrist_x)
    cy = int(reading.wrist_y - 0.4 * half)   # bias upward toward the hand
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    crop = frame[y0:y1, x0:x1]
    if max(crop.shape[:2]) < _HAND_CROP_UPSCALE_PX:
        crop = cv2.resize(crop, (_HAND_CROP_UPSCALE_PX, _HAND_CROP_UPSCALE_PX),
                          interpolation=cv2.INTER_LINEAR)
    return crop


def _diff_moves(diff: np.ndarray, mean_thresh: float, area_min: float) -> bool:
    """Both motion tests over one patch of the frame difference."""
    if float(diff.mean()) / 255.0 > mean_thresh:
        return True
    return float((diff > _MOTION_CELL_DELTA).sum()) / diff.size > area_min


def _boxes_to_thumb(boxes, pw: int, ph: int) -> list[tuple]:
    """Person boxes in frame pixels → motion-thumbnail rectangles."""
    out = []
    for box in boxes:
        x1 = max(0, int(box[0] * _MOTION_THUMB_W / pw))
        y1 = max(0, int(box[1] * _MOTION_THUMB_H / ph))
        x2 = min(_MOTION_THUMB_W, int(box[2] * _MOTION_THUMB_W / pw) + 1)
        y2 = min(_MOTION_THUMB_H, int(box[3] * _MOTION_THUMB_H / ph) + 1)
        if (x2 - x1) * (y2 - y1) >= _MOTION_REGION_MIN_CELLS:
            out.append((x1, y1, x2, y2))
    return out


def _frame_has_motion(prev_small, small, mean_thresh: float,
                      area_min: float, regions=None) -> bool:
    """Has enough of the scene changed to be worth running the pose model?

    Two tests, either sufficient. The mean is the original and is the wrong
    statistic for one person in a wide view: measured on a real living-room
    frame, an arm-sized limb moving changes the mean by 0.0016 and even a
    whole person shifting only reaches 0.0115 — both under the 0.015 default,
    so gestures never woke the detector at all. The area test asks how much
    of the frame changed appreciably, which is what a moving limb looks like
    regardless of how much empty room surrounds it, and stays at zero for
    sensor noise.

    *regions* re-runs the same two tests inside the last known person boxes.
    Both statistics divide by the area they are measured over, so a sweeping
    arm that is a rounding error against a whole living room is a large
    fraction of the person it belongs to — the same denominator problem that
    kept the gate shut before, one scale down. The full-frame test still runs
    first, so somebody walking in where nobody is tracked yet still wakes it.
    """
    if prev_small is None:
        return True
    diff = cv2.absdiff(small, prev_small)
    if _diff_moves(diff, mean_thresh, area_min):
        return True
    for x1, y1, x2, y2 in regions or ():
        sub = diff[y1:y2, x1:x2]
        if sub.size and _diff_moves(sub, mean_thresh, area_min):
            return True
    return False


class _StageTimer:
    """Where the analyser loop's time actually goes.

    Added because a live system was sampling poses only 0.63 times a second
    while its camera reported 11 fps, and nothing in the logs could say
    which stage was responsible — the capture rate in the stats line says
    nothing about how often the pose model actually looks at the scene, and
    that rate is what decides whether an arm sweep is measurable. Guessing
    from YOLO-run gaps gave the wrong answer once already.

    Costs one monotonic() per stage per frame and logs a summary every
    *period_s*, so it stays on in normal operation rather than being a
    debugging build someone has to reproduce the problem under.
    """

    def __init__(self, period_s: float = 30.0) -> None:
        self._period = period_s
        self._totals: "collections.OrderedDict[str, float]" = collections.OrderedDict()
        self._iters = 0
        self._yolo_at_start = 0
        self._started = time.monotonic()

    def mark(self, stage: str, since: float) -> float:
        t = time.monotonic()
        self._totals[stage] = self._totals.get(stage, 0.0) + (t - since)
        return t

    def tick(self, label: str, yolo_runs: int) -> None:
        self._iters += 1
        elapsed = time.monotonic() - self._started
        if elapsed < self._period:
            return
        yolo = yolo_runs - self._yolo_at_start
        parts = " ".join(
            f"{k} {v / self._iters * 1000:.0f}" for k, v in self._totals.items()
        )
        log.info(
            "[%s] loop %.1f iter/s, pose %.2f sample/s (%d in %.0fs) | "
            "mean ms/iter: %s",
            label, self._iters / elapsed, yolo / elapsed, yolo, elapsed, parts,
        )
        self._totals.clear()
        self._iters = 0
        self._yolo_at_start = yolo_runs
        self._started = time.monotonic()


def _sweep_meets_flourish(reading, climb_min: float, rate_min: float,
                          rate_max: float = 4.00) -> bool:
    """Does this reading show a sweep worth chiming for?

    Requires BOTH terms of the gesture gate, and that the arm actually
    arrived somewhere — i.e. the reading is a confirmed raise.

    Rate alone is not usable: it is climb divided by elapsed time, so a
    hand's worth of keypoint jitter between two adjacent frames divides a
    tiny climb by a tiny interval and produces a large rate.

    The raised-state requirement matters just as much. Sweep is computed for
    the highest arm whatever its state, so an arm swung up with a bent elbow
    — which never clears arm_extension_min and never becomes a gesture —
    still produced a climb and chimed. A live install logged three chimes in
    151 seconds with no SINGLE_UP transition at all.
    """
    if reading is None:
        return False
    if getattr(reading, 'state', None) is not ArmState.SINGLE_UP:
        return False
    return (float(getattr(reading, 'sweep_climb', 0.0)) >= climb_min
            and rate_min <= float(getattr(reading, 'sweep_rate', 0.0)) <= rate_max)


class CameraAnalyser(threading.Thread):
    def __init__(
        self,
        camera,
        settings,
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
        debug_frame_cb=None,
        debug_wanted_cb=None,
        feedback_logger=None,
        chime_notifier=None,
        gesture_sound_cb=None,
        num_threads: int = 0,
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self.debug_frame_cb = debug_frame_cb
        # Asked before each stream render. None means "always render", which
        # keeps every existing caller and test behaving as before.
        self.debug_wanted_cb = debug_wanted_cb
        self._num_threads = num_threads
        self._stop_evt = threading.Event()
        self._feedback = feedback_logger
        self._chime = chime_notifier
        # Picks the sound for a fired gesture — some have their own, the rest
        # get the confirmation chime. Owned by GestureSystem, which holds the
        # mapping; the analyser must not play both, which is what happened
        # when the two were decided in different places.
        self._gesture_sound_cb = gesture_sound_cb
        self._near_miss_cb = feedback_logger.on_near_miss if feedback_logger else None

        # CLAHE cache — object is expensive; recreate only when clip changes.
        self._clahe_obj = None
        self._clahe_clip: float = -1.0

        # Optional test recorder (set by main.py when ENDORA_RECORD_TESTS=1)
        self._recorder = None

        # Frame capture for gesture debugging
        try:
            self._frame_capture: Optional[FrameCapture] = FrameCapture()
        except Exception as e:
            log.warning("[%s] FrameCapture unavailable: %s", label, e)
            self._frame_capture = None

        # Per-person tracking: each detected person gets their own ArmTracker +
        # GestureStateMachine, keyed by an auto-incrementing integer ID assigned
        # by nearest-centroid matching across YOLO frames.
        self._persons: dict[int, _PersonEntry] = {}
        self._next_pid: int = 0
        # "Is somebody lying down in this room" is a question about the SCENE,
        # not about a track id — one reclining occupant produces two
        # simultaneous detections that get separate pids, and the one that
        # fires FOLDED_ARMS is the hallucinated upright skeleton, whose own
        # tracker never observes the recline. Shared by every person's
        # tracker on this camera, and it outlives them all. See
        # ReclineWitness.
        self._recline_witness = ReclineWitness()
        # Counts actual YOLO runs, so person pruning can be expressed in
        # missed detections rather than elapsed time (see _PERSON_PRUNE_MISSES).
        self._yolo_runs: int = 0
        # Last known person boxes, in motion-thumbnail coordinates.
        self._motion_regions: list[tuple] = []

    def stop(self):
        self._stop_evt.set()

    # ── Person pool management ─────────────────────────────────────────────

    def _make_person_entry(self, lm, centroid: tuple, now: float) -> _PersonEntry:
        s = self.s
        arm_tracker = ArmTracker(ArmTrackerConfig(
            raise_elevation_min=float(getattr(s, 'raise_elevation_min', 0.70)),
            arm_extension_min=float(getattr(s, 'arm_extension_min', 0.80)),
            min_arm_len_frac=float(getattr(s, 'min_arm_len_frac', 0.55)),
            pose_visibility_min=float(getattr(s, 'pose_visibility_min', 0.45)),
            keypoint_visibility_min=float(getattr(s, 'keypoint_visibility_min', 0.30)),
            leg_raise_margin=float(getattr(s, 'leg_raise_margin', 0.05)),
            # Skip poses that are switched off entirely — they are tested
            # before the single-arm raise and would otherwise shadow it.
            detect_cross_arms=bool(getattr(s, 'gesture_cross_arms_enable', True)),
            detect_t_pose=bool(getattr(s, 'gesture_t_pose_enable', True)),
            detect_both_up=bool(getattr(s, 'gesture_raise_both_enable', True)),
            detect_folded_arms=bool(getattr(s, 'gesture_folded_arms_enable', True)),
            folded_wrist_proximity=float(getattr(s, 'folded_wrist_proximity', 0.50)),
            folded_midline_max=float(getattr(s, 'folded_midline_max', 0.35)),
            folded_chest_depth=float(getattr(s, 'folded_chest_depth', 0.45)),
            folded_resolution_min=float(getattr(s, 'folded_resolution_min', 1.60)),
            folded_visibility_min=float(getattr(s, 'folded_visibility_min', 0.50)),
            folded_extension_max=float(getattr(s, 'folded_extension_max', 0.80)),
            folded_arm_span_max=float(getattr(s, 'folded_arm_span_max', 1.60)),
            folded_lean_max_deg=float(getattr(s, 'folded_lean_max_deg', 22.0)),
            folded_knee_above_hip_max=float(
                getattr(s, 'folded_knee_above_hip_max', 0.15)),
            folded_recline_memory_s=float(
                getattr(s, 'folded_recline_memory_s', 1.5)),
            folded_posture_visibility_min=float(
                getattr(s, 'folded_posture_visibility_min', 0.50)),
            facing_shoulder_min=float(getattr(s, 'facing_shoulder_min', 0.45)),
            state_confirm_s=float(getattr(s, 'state_confirm_s', 0.20)),
            state_release_s=float(getattr(s, 'state_release_s', 0.30)),
            rise_elevation_delta=float(getattr(s, 'rise_elevation_delta', 0.35)),
            rise_start_elevation_max=float(getattr(s, 'rise_start_elevation_max', 0.35)),
            wrist_still_max_travel=float(getattr(s, 'wrist_still_max_travel_arm', 0.15)),
        ), recline_witness=self._recline_witness)
        state_machine = GestureStateMachine(StateMachineConfig(
            cooldown_s=float(getattr(s, 'cooldown_s', 2.0)),
            cross_gesture_cooldown_s=float(
                getattr(s, 'cross_gesture_cooldown_s', 0.5)),
            snap_elevation_min=float(getattr(s, 'snap_elevation_min', 0.70)),
            hold_duration_s=float(getattr(s, 'hold_duration_s', 1.5)),
            double_snap_window_s=float(getattr(s, 'double_snap_window_s', 3.0)),
            sustain_s=float(getattr(s, 'sustain_s', 0.5)),
            # sustain_gap_s was in Settings, in the registry and in
            # /effective, but was never passed here — so it silently had no
            # effect at all while the debug page reported it as live.
            sustain_gap_s=float(getattr(s, 'sustain_gap_s', 0.85)),
            sustain_credit_ticks=float(getattr(s, 'sustain_credit_ticks', 2.5)),
            snap_sustain_s=float(getattr(s, 'snap_sustain_s', 0.0)),
            snap_roll_threshold=float(getattr(s, 'snap_roll_threshold', 0.0)),
            snap_require_flourish=bool(getattr(s, 'snap_require_flourish', True)),
            flourish_min_climb=float(getattr(s, 'flourish_min_climb', 0.60)),
            flourish_min_rate=float(getattr(s, 'flourish_min_rate', 0.80)),
            flourish_max_rate=float(getattr(s, 'flourish_max_rate', 4.00)),
            snap_require_rise=bool(getattr(s, 'snap_require_rise', True)),
            snap_require_still=bool(getattr(s, 'snap_require_still', False)),
            sustained_rearm_s=float(getattr(s, 'sustained_rearm_s', 2.0)),
            enable_snap=bool(getattr(s, 'gesture_snap_enable', True)),
            enable_hold=bool(getattr(s, 'gesture_hold_enable', True)),
            enable_double_snap=bool(getattr(s, 'gesture_double_snap_enable', True)),
            enable_cross_arms=bool(getattr(s, 'gesture_cross_arms_enable', True)),
            enable_t_pose=bool(getattr(s, 'gesture_t_pose_enable', True)),
            enable_raise_both=bool(getattr(s, 'gesture_raise_both_enable', True)),
            enable_folded_arms=bool(getattr(s, 'gesture_folded_arms_enable', True)),
        ), on_near_miss=self._near_miss_cb)
        # A brand-new pid can only be created from a candidate that didn't
        # match any existing tracked pid (see _match_persons) — and since
        # _LIVENESS_EXEMPT_DIST is always tighter than _PERSON_MATCH_DIST,
        # any candidate close enough to a known centroid to be exempted in
        # _all_valid_landmarks would also have matched that pid here rather
        # than spawning a new one. So this first sighting was necessarily a
        # genuine (non-exempted) wrist-liveness pass — count it as such.
        return _PersonEntry(
            arm_tracker=arm_tracker,
            state_machine=state_machine,
            centroid=centroid,
            last_seen=now,
            last_seen_run=self._yolo_runs,
            last_lm=lm,
            last_genuine_live_at=now,
            # Birth counts as the window's first pass, as it always has —
            # the anchor just has its own field now so that
            # last_genuine_live_at can keep advancing for _known_centroids.
            first_genuine_live_at=now,
        )

    def _note_liveness(
        self, e: "_PersonEntry", raw_live: bool, moved: bool, now: float
    ) -> None:
        """Update a pid's confirmed-human status from this frame's checks.

        Two genuine AND MOVED passes, at least _LIVENESS_CONFIRM_MIN_GAP_S
        and at most _LIVENESS_CONFIRM_WINDOW_S apart, permanently confirm the
        pid as human (sticky — never revoked short of the pid itself being
        pruned). Both bounds matter: with only the upper one, two adjacent
        frames qualified. Requiring actual
        keypoint displacement (not just the foreground-mask check passing)
        is what closes the gap the window alone left open: correlated
        lighting noise (a flicker, an exposure adjustment) can make the
        background-subtraction mask fire "genuine" on several frames in a
        row for a completely static object, but it can never make that
        object's own reported keypoint positions actually shift.
        """
        if not raw_live or not moved or e.confirmed_human:
            return
        first = e.first_genuine_live_at
        if first is None or now - first > _LIVENESS_CONFIRM_WINDOW_S:
            e.first_genuine_live_at = now        # anchor a fresh window
        elif now - first >= _LIVENESS_CONFIRM_MIN_GAP_S:
            e.confirmed_human = True
            log.info("[%s] pid confirmed human (2 genuine, moved passes "
                     "%.1fs apart)", self.label, now - first)
        e.last_genuine_live_at = now

    def _match_persons(
        self, detected: list, frame_w: int, frame_h: int, now: float
    ) -> None:
        """Match detected (landmarks, centroid, raw_live) triples to existing
        person entries. Updates existing entries in-place; creates new ones
        for novel persons. Uses greedy nearest-centroid matching — sufficient
        for small N (< 10).
        """
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        max_dist = _PERSON_MATCH_DIST * diag
        moved_min_dist = _CENTROID_MOVED_MIN_FRAC * diag
        available = list(self._persons.keys())

        for lm, centroid, raw_live in detected:
            best_pid, best_dist = None, float('inf')
            for pid in available:
                e = self._persons[pid]
                d = (
                    (centroid[0] - e.centroid[0]) ** 2 +
                    (centroid[1] - e.centroid[1]) ** 2
                ) ** 0.5
                if d < best_dist:
                    best_dist, best_pid = d, pid

            if best_pid is not None and best_dist <= max_dist:
                e = self._persons[best_pid]
                moved = best_dist >= moved_min_dist
                e.centroid  = centroid
                e.last_lm   = lm
                e.last_seen = now
                e.last_seen_run = self._yolo_runs
                available.remove(best_pid)
                self._note_liveness(e, raw_live, moved, now)
            else:
                pid = self._next_pid
                self._next_pid += 1
                self._persons[pid] = self._make_person_entry(lm, centroid, now)
                log.info("[%s] New person pid=%d at (%.0f, %.0f)",
                         self.label, pid, *centroid)

    def _prune_persons(self, now: float) -> None:
        # Both conditions, deliberately: missed runs alone is wrong when the
        # model runs fast, elapsed time alone is wrong when it runs slowly,
        # and this project has now been bitten by each in turn.
        stale = [pid for pid, e in self._persons.items()
                 if self._yolo_runs - e.last_seen_run >= _PERSON_PRUNE_MISSES
                 and now - e.last_seen >= _PERSON_PRUNE_MIN_S]
        for pid in stale:
            log.info("[%s] Lost person pid=%d", self.label, pid)
            del self._persons[pid]

    # ── Frame preprocessing ───────────────────────────────────────────────

    def _preprocess(self, frame):
        """Apply dewarp, flip, crop. Returns (proc_frame, w, h).

        CLAHE is intentionally NOT applied here — see _apply_low_light_enhance,
        called separately after the background subtractor samples the frame.
        """
        h, w = frame.shape[:2]

        if getattr(self.s, 'dewarp_enable', False):
            from cameras.dewarp import build_dewarp_maps, apply_dewarp
            cx_raw = float(getattr(self.s, 'dewarp_cx', -1.0))
            cy_raw = float(getattr(self.s, 'dewarp_cy', -1.0))
            dw = int(getattr(self.s, 'dewarp_out_width', 640))
            dh = int(getattr(self.s, 'dewarp_out_height', 480))
            fov = float(getattr(self.s, 'dewarp_fov', 180.0))
            pan = float(getattr(self.s, 'dewarp_pan', 0.0))
            tilt = float(getattr(self.s, 'dewarp_tilt', 20.0))
            roll = float(getattr(self.s, 'dewarp_roll', 0.0))
            vfov = float(getattr(self.s, 'dewarp_vfov', 75.0))
            key = (w, h, dw, dh, fov, pan, tilt, roll, vfov, cx_raw, cy_raw)
            if getattr(self, '_dewarp_key', None) != key:
                self._dewarp_maps = build_dewarp_maps(
                    in_w=w, in_h=h, out_w=dw, out_h=dh,
                    fisheye_fov_deg=fov, pan_deg=pan, tilt_deg=tilt,
                    roll_deg=roll, vfov_deg=vfov,
                    cx=None if cx_raw < 0 else cx_raw,
                    cy=None if cy_raw < 0 else cy_raw,
                )
                self._dewarp_key = key
            frame = apply_dewarp(frame, *self._dewarp_maps)
            h, w = frame.shape[:2]

        if getattr(self.s, 'flip_image', False):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]

        ct = float(getattr(self.s, 'frame_crop_top', 0))
        cb = float(getattr(self.s, 'frame_crop_bottom', 0))
        cl = float(getattr(self.s, 'frame_crop_left', 0))
        cr = float(getattr(self.s, 'frame_crop_right', 0))
        y0, y1 = int(h * ct / 100), h - int(h * cb / 100)
        x0, x1 = int(w * cl / 100), w - int(w * cr / 100)
        if y0 > 0 or y1 < h or x0 > 0 or x1 < w:
            frame = frame[y0:y1, x0:x1]

        ph, pw = frame.shape[:2]
        return frame, pw, ph

    def _apply_low_light_enhance(self, frame):
        """CLAHE local-contrast boost, kept separate from _preprocess so it can
        run AFTER the background subtractor samples the frame. CLAHE amplifies
        contrast most aggressively in dim regions — exactly where it can also
        amplify ordinary sensor noise into something that reads as motion,
        which previously let static objects (e.g. a framed picture in a
        shadowed corner) intermittently clear the background-subtraction
        liveness check with no real movement involved.
        """
        if not getattr(self.s, 'low_light_enhance', False):
            return frame
        clip = float(getattr(self.s, 'low_light_clip', 2.0))
        if clip != self._clahe_clip:
            self._clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            self._clahe_clip = clip
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe_obj.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        try:
            self._run()
        except Exception:
            log.exception("[%s] Analyser crashed", self.label)
            raise

    def _run(self):
        import os
        from cameras.pose_model import PoseModel

        model_name = getattr(self.s, 'yolo_pose_model', 'yolo11n-pose.onnx')
        if not os.path.isabs(model_name):
            model_name = os.path.join('/app', model_name)

        yolo_imgsz = int(getattr(self.s, 'yolo_imgsz', 320))
        yolo_conf  = float(getattr(self.s, 'yolo_conf',  0.45))
        model = PoseModel(
            model_path=model_name,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            num_threads=self._num_threads,
            execution_provider=str(getattr(self.s, 'yolo_execution_provider', 'auto')),
        )

        # Adaptive background model — flags framed pictures, mirrors, TV content
        # etc. that YOLO mis-detects as a permanently "raised arm". Continuously
        # re-learns the static scene so it tolerates gradual lighting drift, but
        # anything that hasn't settled into the background yet reads as
        # foreground. detectShadows=False keeps the mask a clean 0/255.
        # Object creation is cheap and unconditional; bg_subtract_enable is
        # re-read every frame below (like every other live-tunable setting)
        # so toggling it on the debug page takes effect immediately, with no
        # add-on restart required.
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        # grlib/MediaPipe Hands is initialized lazily on the first SINGLE_UP
        # frame to avoid loading two ML runtimes simultaneously at startup.
        _hand_pipeline = None
        _NoHandDetected = None
        _grlib_ok = True

        log.info("[%s] Analyser running (v%s — YOLO pose + grlib hands)",
                 self.label, __version__)

        _stage_timer = _StageTimer()
        _cached_kps: Optional[np.ndarray] = None   # [N, 17, 3] from PoseModel
        _cached_boxes: Optional[np.ndarray] = None  # [N, 5] xyxy+conf, same rows
        _prev_small: Optional[np.ndarray] = None   # for motion gate
        _frames_since_yolo: int = 999              # force run on first frame
        _prev_primary_state: ArmState = ArmState.DOWN
        # Set on every YOLO run; the debug-overlay loop below reads it on
        # frames where YOLO did not run, so it needs a value from the start.
        _acquire_conf: float = float(getattr(self.s, 'yolo_conf', 0.45)) * 1.3
        _last_person_count: int = -1
        _logged_frame_geometry: bool = False

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            _t_stage = now

            proc_frame, pw, ph = self._preprocess(frame)
            _t_stage = _stage_timer.mark("prep", _t_stage)

            # Report the real pixel budget once. Everything downstream is
            # limited by it, and it is not otherwise visible anywhere: an
            # RTSP *sub* stream dewarped up to a larger canvas looks the
            # same in the debug view as a main stream, while carrying a
            # fraction of the detail the pose model needs.
            if not _logged_frame_geometry:
                _logged_frame_geometry = True
                fh_, fw_ = frame.shape[:2]
                eff = min(yolo_imgsz / max(pw, ph), 1.0)
                log.info("[%s] frame geometry: camera %dx%d -> analysed %dx%d "
                         "-> model sees %dx%d of a %dx%d canvas",
                         self.label, fw_, fh_, pw, ph,
                         int(pw * eff), int(ph * eff), yolo_imgsz, yolo_imgsz)
                if fw_ * fh_ < 640 * 480:
                    log.warning("[%s] camera stream is only %dx%d — if this is an "
                                "RTSP 'sub' stream, the main stream would give the "
                                "pose model far more to work with", self.label, fw_, fh_)

            # Fed every frame (not just motion-gated ones) so the model keeps
            # tracking gradual lighting drift even when nothing is moving.
            # Sampled BEFORE CLAHE (below) — CLAHE's local-contrast boost can
            # amplify sensor noise in dim areas into something that looks like
            # motion, which would undermine the liveness check it feeds.
            if getattr(self.s, 'bg_subtract_enable', True):
                if pw > _BG_SUBTRACT_WIDTH:
                    bg_in = cv2.resize(
                        proc_frame,
                        (_BG_SUBTRACT_WIDTH, max(1, ph * _BG_SUBTRACT_WIDTH // pw)),
                        interpolation=cv2.INTER_AREA)
                else:
                    bg_in = proc_frame
                fg_mask = bg_subtractor.apply(bg_in)
            else:
                fg_mask = None
            _t_stage = _stage_timer.mark("bgsub", _t_stage)
            min_fg_frac = float(getattr(self.s, 'bg_subtract_min_foreground', 0.12))
            # Confirmed-human persons' positions, plus pids still inside
            # their own confirmation grace period (see _known_centroids),
            # exempt from the wrist-liveness check below (see
            # _passes_liveness_gate and _LIVENESS_CONFIRM_WINDOW_S) —
            # computed from state as of the end of the previous iteration.
            known_centroids = _known_centroids(self._persons, now)
            match_dist = _LIVENESS_EXEMPT_DIST * ((pw ** 2 + ph ** 2) ** 0.5)

            proc_frame = self._apply_low_light_enhance(proc_frame)
            _t_stage = _stage_timer.mark("clahe", _t_stage)

            # ── Motion gate ───────────────────────────────────────────────
            # Resize to 80×60 (~0.1 ms) and diff against previous frame.
            # Skip YOLO when the scene is static — reuse cached landmarks.
            # Always run YOLO when:
            #   • significant motion detected  (something is moving)
            #   • any arm already raised       (responsive snap detection)
            #   • heartbeat interval reached   (catch slow arm lifts)
            mot_thresh = float(getattr(self.s, 'motion_threshold', 0.015))
            area_min   = float(getattr(self.s, 'motion_area_min', 0.002))
            max_skip   = int(getattr(self.s,   'yolo_max_skip',    12))
            # While somebody is tracked, sample as fast as the loop allows.
            # The gate exists to keep an empty room cheap; it was also
            # throttling the one moment that matters, because during a
            # sweep's ascent the arm state is still DOWN and neither
            # `motion` nor `any_arm_up` is reliably true.
            if self._persons:
                max_skip = max(1, int(getattr(self.s, 'yolo_max_skip_active', 1)))

            gray  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (_MOTION_THUMB_W, _MOTION_THUMB_H),
                               interpolation=cv2.INTER_AREA)
            motion = _frame_has_motion(_prev_small, small, mot_thresh, area_min,
                                       regions=self._motion_regions)
            _prev_small = small
            _frames_since_yolo += 1
            _t_stage = _stage_timer.mark("motion", _t_stage)

            any_arm_up = any(e.last_arm_state != ArmState.DOWN
                             for e in self._persons.values())
            run_yolo = motion or any_arm_up or (_frames_since_yolo >= max_skip)

            if run_yolo:
                # Confidence hysteresis:
                #   acquire  (no one tracked): base_conf * 1.3 — strict, rejects ghosts
                #   maintain (someone tracked): base_conf * 0.65 — bridges arm-raise dropouts
                base_conf = float(getattr(self.s, 'yolo_conf', 0.45))
                _acquire_conf = base_conf * 1.3      # strict: start a track
                _maintain_conf = base_conf * 0.65    # permissive: keep one
                # Inference runs permissive whenever anyone is tracked, so a
                # raise's dropouts do not lose the person. Which detections
                # that relaxation is allowed to APPLY to is decided per
                # detection in _acquire_ok — see there for why letting it
                # apply frame-wide is self-sustaining.
                model.conf = _maintain_conf if self._persons else _acquire_conf

                # Boxes are carried alongside the keypoints purely for the
                # debug overlay; nothing in the gesture path reads them.
                if getattr(self.s, 'crop_refine_enable', False):
                    _cached_kps, _cached_boxes = model.infer_refined(
                        proc_frame,
                        margin=float(getattr(self.s, 'crop_refine_margin', 0.15)),
                        max_persons=int(getattr(self.s, 'crop_refine_max_persons', 2)),
                        min_box_frac=float(
                            getattr(self.s, 'crop_refine_min_box_frac', 0.55)),
                    )
                else:
                    _cached_kps, _cached_boxes = model.infer(proc_frame)
                _frames_since_yolo = 0
                self._yolo_runs += 1
                _t_stage = _stage_timer.mark("yolo", _t_stage)
                log.debug("[%s] YOLO ran (motion=%s any_arm_up=%s persons=%d "
                          "refined=%d)",
                          self.label, motion, any_arm_up, len(self._persons),
                          getattr(model, 'last_refined', 0))

                detected = _all_valid_landmarks(
                    _cached_kps, pw, ph, fg_mask=fg_mask, min_foreground_frac=min_fg_frac,
                    known_centroids=known_centroids, match_dist=match_dist,
                    boxes=_cached_boxes,
                    tracked_centroids=[e.centroid for e in self._persons.values()],
                    acquire_conf=_acquire_conf,
                )
                self._match_persons(detected, pw, ph, now)
                self._prune_persons(now)

                # Feed the recorder one buffer per tracked person.
                #
                # This used to record a single buffer fed with valid[0] — the
                # first detection by array index, which is not stable between
                # frames and need not be the person who fires. One occupant
                # regularly produces two concurrent detections here, so the
                # saved trace interleaved two bodies and replayed as neither.
                if self._recorder is not None:
                    for pid, e in self._persons.items():
                        kps_row = getattr(e.last_lm, "kps", None)
                        if kps_row is not None and e.last_seen == now:
                            self._recorder.on_frame(kps_row, pw, ph, now, pid=pid)

            # ── Hand landmarks (grlib / MediaPipe Hands) ──────────────────
            # Only run when at least one person has an arm raised — avoids
            # running both ML models every frame on resource-constrained hardware.
            any_single_up = any(e.last_arm_state == ArmState.SINGLE_UP
                                for e in self._persons.values())
            hand_lm: Optional[np.ndarray] = None
            if any_single_up and _grlib_ok:
                if _hand_pipeline is None:
                    try:
                        sys.modules.setdefault('cv2.cv2', cv2)
                        from grlib.feature_extraction.pipeline import Pipeline
                        from grlib.exceptions import NoHandDetectedException as _NHD
                        _NoHandDetected = _NHD
                        _hand_pipeline = Pipeline(num_hands=1, optimize_pipeline=True)
                        _hand_pipeline.add_stage()
                        log.info("[%s] grlib hand pipeline ready", self.label)
                    except Exception as e:
                        log.warning("[%s] grlib init failed, snap_roll disabled: %s",
                                    self.label, e)
                        _grlib_ok = False

                if _hand_pipeline is not None:
                    # Crop around the raised wrist so the hand fills a useful
                    # fraction of the image MediaPipe sees — full-frame hands
                    # at couch distance are too small to detect reliably.
                    # snap_roll is a ratio of intra-hand x-distances, so it is
                    # unaffected by the crop's translation/upscale.
                    hand_img = proc_frame
                    if bool(getattr(self.s, 'hand_crop_enable', True)):
                        up_entry = next(
                            (e for e in self._persons.values()
                             if e.last_arm_state == ArmState.SINGLE_UP
                             and e.last_reading is not None),
                            None)
                        if up_entry is not None:
                            crop = _crop_around_wrist(
                                proc_frame, up_entry.last_reading, up_entry.last_lm)
                            if crop is not None:
                                hand_img = crop
                    try:
                        flat_lm, _ = _hand_pipeline.get_landmarks_from_image(hand_img)
                        hand_lm = flat_lm
                    except Exception as e:
                        if _NoHandDetected is None or not isinstance(e, _NoHandDetected):
                            log.debug("[%s] grlib hand error: %s", self.label, e)

            # Only on change. Logged every frame, this flooded the debug
            # ring buffer down to under a minute of history — useless for
            # diagnosing a gesture that happened 90 seconds ago.
            if len(self._persons) != _last_person_count:
                _last_person_count = len(self._persons)
                log.debug("[%s] %d person(s) tracked", self.label, _last_person_count)

            # ── Per-person gesture processing ──────────────────────────────
            _primary_reading: Optional[object] = None
            _primary_gesture: Optional[Gesture] = None

            for pid, entry in list(self._persons.items()):
                prev_state = entry.last_arm_state
                # Only pass hand_lm to the person whose arm is already up —
                # grlib detects one hand in the frame and we can't tell whose.
                lm_for_hand = hand_lm if entry.last_arm_state == ArmState.SINGLE_UP else None
                reading = entry.arm_tracker.classify(
                    entry.last_lm, pw, ph, lm_for_hand, now
                )

                if reading is not None:
                    # No head start for a sustained pose. v1.9.175 played the
                    # sound the moment the pose was confirmed, to recover the
                    # 0.58 s that state_confirm_s and sustain_s cost — and two
                    # of every three sounds then had no gesture behind them,
                    # because you pass THROUGH a folded-arms shape constantly
                    # and only holding it means anything. The head start works
                    # for SNAP because a sweep predicts a snap; "held this
                    # shape for 0.2 s" predicts nothing. Latency is the price
                    # of the hold, and the hold is the gesture — shorten
                    # sustain_s if it matters, do not guess ahead of it.
                    entry.last_arm_state = reading.state
                    entry.last_reading = reading
                    if reading.state != entry.last_logged_state:
                        log.info("[%s] pid=%d state → %s",
                                 self.label, pid, reading.state.name)
                        entry.last_logged_state = reading.state
                    if reading.state.name == 'SINGLE_UP':
                        log.debug(
                            "[%s] pid=%d SINGLE_UP elev=%.2f ext=%.2f "
                            "sweep=%.2f@%.2f/s rose=%s still=%s",
                            self.label, pid, reading.elevation, reading.extension,
                            reading.sweep_climb, reading.sweep_rate,
                            reading.rose_recently, reading.wrist_still)
                    # Chime as soon as a sweep meets the FULL flourish bar —
                    # a head start on speaker latency (an Echo is ~1-2s), so
                    # the sound lands with the gesture rather than after it.
                    #
                    # Both climb AND rate are required, exactly as the
                    # gesture gate requires them. Rate alone is not a usable
                    # trigger: it is climb/elapsed, so a hand's worth of
                    # keypoint jitter between two adjacent frames divides a
                    # tiny climb by a tiny interval and yields a large rate.
                    # An arm resting near horizontal — where elevation is
                    # most sensitive to wrist noise — chimed several times a
                    # minute that way, with no gesture behind it.
                    _sweeping = _sweep_meets_flourish(
                        reading,
                        float(getattr(self.s, 'flourish_min_climb', 0.60)),
                        float(getattr(self.s, 'flourish_min_rate', 0.80)),
                        float(getattr(self.s, 'flourish_max_rate', 4.00)))
                    if self._chime is not None:
                        if _sweeping and not entry.chimed_this_sweep:
                            entry.chimed_this_sweep = True
                            self._chime.notify()
                        elif not _sweeping and reading.state == ArmState.DOWN:
                            entry.chimed_this_sweep = False

                    if self._feedback:
                        self._feedback.push_reading(reading)

                gesture = entry.state_machine.tick(reading, now)
                if gesture is not None:
                    log.debug("[%s] pid=%d gesture: %s", self.label, pid, gesture)
                    # Guarantee the sound accompanies a real gesture. The
                    # sweep-onset chime above is only a head start on speaker
                    # latency and may not have fired (a sustained pose has no
                    # sweep at all); chime_debounce_s dedupes when it did.
                    #
                    # Exactly one sound per gesture: the callback decides
                    # which. Firing the chime here AND a per-gesture sound
                    # elsewhere sent two clips to the speaker a millisecond
                    # apart, where announce:true makes the second replace the
                    # first — so the gesture's own sound was audible only if
                    # it happened to win the race.
                    if self._gesture_sound_cb is not None:
                        entry.chimed_this_sweep = True
                        self._gesture_sound_cb(gesture)
                    elif self._chime is not None:
                        entry.chimed_this_sweep = True
                        self._chime.notify()
                    self.on_candidate(gesture, 1.0, self.label)
                    if self._recorder is not None:
                        # This person's buffer, so the trace and the gesture
                        # describe the same body.
                        self._recorder.on_gesture(gesture, self.label, pid=pid)
                    _primary_gesture = gesture

                # Prefer the arm-up person's reading for the debug overlay
                if reading is not None and (
                    _primary_reading is None or reading.state != ArmState.DOWN
                ):
                    _primary_reading = reading

            # ── Collect all valid persons' kps for the debug overlay ───────
            # Same gate _all_valid_landmarks applies to gesture candidates, so
            # a static ghost (e.g. a framed picture) never draws a skeleton —
            # but an already-tracked real person is exempt from the wrist-
            # liveness half of that gate (see _passes_liveness_gate), so
            # holding still doesn't make them vanish from the overlay as
            # "NO POSE DETECTED" while they're plainly still there.
            # Boxes are collected for *every* detection, including the ones
            # rejected above. A rejection is invisible in the skeleton view —
            # a ghost on the wall art and a person the gate wrongly dropped
            # both render as empty frame — and telling those two apart is the
            # whole diagnostic question. The box, with its confidence and
            # reject reason, answers it.
            _all_dbg_kps: list[np.ndarray] = []
            _dbg_boxes: list[tuple] = []          # (box[5], reason|None)
            if _cached_kps is not None:
                for i in range(_cached_kps.shape[0]):
                    box = (_cached_boxes[i] if _cached_boxes is not None
                           and i < _cached_boxes.shape[0] else None)
                    reason: Optional[str] = None
                    if _person_visible_kp_count(_cached_kps[i]) < _MIN_VISIBLE_KPS:
                        reason = "few kps"
                    c = _person_centroid(_cached_kps[i])
                    if reason is None and c is None:
                        reason = "no centroid"
                    if reason is None and not _acquire_ok(
                        float(box[4]) if box is not None else None, c,
                        [e.centroid for e in self._persons.values()],
                        match_dist, _acquire_conf,
                    ):
                        reason = "too weak to acquire"
                    if reason is None and not _passes_liveness_gate(
                        _cached_kps[i], c, fg_mask, pw, ph, min_fg_frac,
                        known_centroids, match_dist,
                    ):
                        reason = "not live"
                    if reason is None:
                        _all_dbg_kps.append(_cached_kps[i])
                    if box is not None:
                        _dbg_boxes.append((box, reason))

            # Next iteration's motion gate measures inside these. Accepted
            # boxes only: a rejected detection is usually a static ghost, and
            # watching its box for motion would hand it the wake-up the
            # liveness gate exists to deny it.
            self._motion_regions = _boxes_to_thumb(
                [b for b, reason in _dbg_boxes if reason is None], pw, ph)

            # ── Frame capture on noteworthy events ────────────────────────────
            current_primary_state = (
                _primary_reading.state if _primary_reading else ArmState.DOWN
            )
            if self._frame_capture is not None:
                _cap_event: Optional[str] = None
                if _primary_gesture is not None:
                    _cap_event = f"gesture_{_primary_gesture.name}"
                elif current_primary_state != _prev_primary_state:
                    _cap_event = f"state_{current_primary_state.name}"
                if _cap_event is not None:
                    r = _primary_reading
                    try:
                        _cap_frame = _draw_debug(
                            proc_frame, _all_dbg_kps, hand_lm, r, _primary_gesture,
                            person_boxes=_dbg_boxes,
                        )
                        self._frame_capture.save(
                            _cap_frame, _cap_event,
                            camera=self.label,
                            arm_state=r.state.name if r else _prev_primary_state.name,
                            gesture=_primary_gesture.name if _primary_gesture else None,
                            forearm_dy=r.forearm_dy if r else 0.0,
                            upright=r.upright if r else None,
                        )
                    except Exception as e:
                        log.debug("[%s] frame capture error: %s", self.label, e)
            _prev_primary_state = current_primary_state

            _t_stage = _stage_timer.mark("gesture", _t_stage)

            # Rendered only while someone is actually looking. The overlay
            # copies the frame and draws over it, which measured ~8% of this
            # loop on the Jetson — paid on every iteration regardless, because
            # nothing tracked viewers. Turning the debug port off to reclaim
            # it is not an option now that the chime is served from the same
            # server, so the idle cost is removed instead.
            #
            # The gesture frame captures above are deliberately NOT gated:
            # their value is having the picture from before anyone opened the
            # page, which is how the intermittent bugs here were found.
            if self.debug_frame_cb is not None and (
                self.debug_wanted_cb is None or self.debug_wanted_cb()
            ):
                try:
                    dbg = _draw_debug(
                        proc_frame, _all_dbg_kps, hand_lm, _primary_reading,
                        _primary_gesture, person_boxes=_dbg_boxes,
                    )
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)
            _stage_timer.mark("debug", _t_stage)

            # Pose sample rate is the number that decides whether a sweep is
            # measurable at all: the flourish window only holds a climb if
            # several samples land inside it. It is not derivable from the
            # capture fps in the stats line — the analyser loop runs slower
            # than the camera, and YOLO slower still.
            _stage_timer.tick(self.label, self._yolo_runs)

        log.info("[%s] Analyser stopped", self.label)


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_person_boxes(img, person_boxes) -> None:
    """Draw one rectangle per person YOLO returned.

    *person_boxes* is a list of ``(box, reason)`` where box is
    ``[x1, y1, x2, y2, conf]`` in frame pixels and reason is None for a
    detection that survived the gates, or a short string naming the gate that
    dropped it. Accepted boxes are green and labelled with the confidence;
    rejected ones are dim red and labelled with the reason, so a ghost and a
    wrongly-dropped person are distinguishable at a glance.
    """
    h, w = img.shape[:2]
    fs = max(0.35, w / 2200)
    for box, reason in person_boxes:
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        ok = reason is None
        color = (0, 200, 0) if ok else (60, 60, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if ok else 1)
        label = f"{box[4]:.2f}" if ok else f"{box[4]:.2f} {reason}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        # Label above the box, or inside it when the box touches the top edge.
        ly = y1 - 4 if y1 - th - 6 >= 0 else min(y1 + th + 6, h - 2)
        cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 6, ly + 3), color, -1)
        cv2.putText(img, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_debug(frame, all_person_kps, hand_lm, reading, fired_gesture,
                person_boxes=None):
    """Draw YOLO skeleton for all detected persons + gesture state overlay.

    *all_person_kps* is a list of [17, 3] numpy arrays, one per valid person.
    *person_boxes* is the optional ``(box, reason)`` list described in
    :func:`_draw_person_boxes` — every detection, accepted or not.
    """
    img = frame.copy()
    h, w = img.shape[:2]

    # Boxes first, so skeletons draw over them rather than under.
    if person_boxes:
        _draw_person_boxes(img, person_boxes)

    for person_kps in all_person_kps:
        for a, b in _COCO_UPPER_BODY:
            x1, y1, c1 = person_kps[a]
            x2, y2, c2 = person_kps[b]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 200, 0), 2)
        for i in range(5, 13):
            x, y, c = person_kps[i]
            if c > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    if not all_person_kps:
        # "NO POSE DETECTED" is only true when YOLO returned nothing. When it
        # returned someone the gates then dropped, say so — the two look
        # identical without the boxes and need opposite fixes.
        n_rejected = len(person_boxes or ())
        msg = (f"{n_rejected} DETECTED, ALL REJECTED" if n_rejected
               else "NO POSE DETECTED")
        fs = max(0.6, w / 800)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx, ty = (w - tw) // 2, 60
        cv2.rectangle(img, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      (0, 0, 180), -1)
        cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 2, cv2.LINE_AA)

    # Wrist marker (only for SINGLE_UP)
    if reading and reading.state.name == 'SINGLE_UP':
        wx, wy = int(reading.wrist_x), int(reading.wrist_y)
        cv2.circle(img, (wx, wy), 12, (255, 255, 0), -1)
        cv2.circle(img, (wx, wy), 12, (0, 0, 0), 2)

    # Status panel
    if reading is not None:
        state_name = reading.state.name
        forearm = reading.forearm_dy
        snap_roll = reading.snap_roll if reading.state.name == 'SINGLE_UP' else 0.0
        hand_str = f"{snap_roll:+.2f}" if hand_lm is not None else "none"
        lines = [
            (f"state: {state_name}", (0, 255, 100)),
            (f"elevation: {getattr(reading, 'elevation', 0.0):+.2f}", (255, 255, 255)),
            (f"extension: {getattr(reading, 'extension', 0.0):.2f}", (255, 255, 255)),
            (f"snap_roll:  {hand_str}", (255, 255, 255)),
            (f"upright: {reading.upright}", (255, 255, 255)),
        ]
        if state_name == 'SINGLE_UP':
            rose = getattr(reading, 'rose_recently', True)
            still = getattr(reading, 'wrist_still', True)
            color = (0, 255, 100) if (rose and still) else (0, 165, 255)
            lines.append((f"rose: {'Y' if rose else 'N'} (+{getattr(reading, 'rise_delta', 0.0):.2f})"
                          f"  still: {'Y' if still else 'N'}", color))
        _sc = getattr(reading, 'sweep_climb', 0.0)
        _sr = getattr(reading, 'sweep_rate', 0.0)
        if _sc > 0.01 or state_name == 'SINGLE_UP':
            lines.append((f"sweep: climb {_sc:.2f}  rate {_sr:.2f}/s",
                          (0, 255, 100) if _sr >= 0.8 else (200, 200, 200)))
    else:
        lines = [("state: none", (160, 160, 160))]

    fs = max(0.35, w / 1800)
    lh = int(fs * 42)
    pad = int(fs * 12)
    panel_h = len(lines) * lh + pad * 2
    panel_w = int(w * 0.32)
    y_start = h - panel_h - 6
    overlay = img.copy()
    cv2.rectangle(overlay, (4, y_start - 2),
                  (4 + panel_w, y_start + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (line, color) in enumerate(lines):
        y = y_start + pad + i * lh + lh - 4
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color, 1, cv2.LINE_AA)

    return img
