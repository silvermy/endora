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

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from version import __version__
from cameras.arm_tracker import ArmState, ArmTracker, ArmTrackerConfig
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
        # kps: shape [17, 3] — (x_px, y_px, conf)
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
_PERSON_PRUNE_S    = 2.0   # seconds without a YOLO detection before dropping entry

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
    """
    if fg_mask is None:
        return True
    mh, mw = fg_mask.shape[:2]
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    half = max(2, int(_WRIST_PATCH_FRAC * diag))
    seen_wrist = False
    for idx in (_COCO_LEFT_WRIST, _COCO_RIGHT_WRIST):
        if kps_row[idx, 2] <= 0.3:
            continue
        seen_wrist = True
        x, y = kps_row[idx, 0], kps_row[idx, 1]
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


def _all_valid_landmarks(
    kps: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    fg_mask: Optional[np.ndarray] = None,
    min_foreground_frac: float = 0.12,
    known_centroids: Optional[list] = None,
    match_dist: float = 0.0,
) -> list[tuple]:
    """Return list of (_YOLOLandmarks, centroid_px, raw_live) for every real
    person detected. known_centroids should come from _known_centroids (see
    _passes_liveness_gate), not an unconditional list of every tracked pid.
    raw_live is this candidate's own wrist-liveness result, independent of
    any known-centroid exemption — the caller (_match_persons) uses it to
    decide whether a tracked pid has earned confirmed-human status, which
    feeds back into _known_centroids on the next frame.
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
    last_lm:           object        # cached _YOLOLandmarks from last YOLO frame
    last_arm_state:    ArmState      = ArmState.DOWN
    last_logged_state: object        = None
    # Liveness confirmation — see _LIVENESS_CONFIRM_WINDOW_S. Sticky once
    # True; confirmed_human pids (and pids still inside their own grace
    # period since last_genuine_live_at — see _known_centroids) contribute
    # to known_centroids, which is what exempts a matching candidate from
    # the wrist-liveness check.
    last_genuine_live_at: Optional[float] = None
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


class CameraAnalyser(threading.Thread):
    def __init__(
        self,
        camera,
        settings,
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
        debug_frame_cb=None,
        feedback_logger=None,
        sonos_notifier=None,
        num_threads: int = 0,
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self.debug_frame_cb = debug_frame_cb
        self._num_threads = num_threads
        self._stop_evt = threading.Event()
        self._feedback = feedback_logger
        self._sonos = sonos_notifier
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

    def stop(self):
        self._stop_evt.set()

    # ── Person pool management ─────────────────────────────────────────────

    def _make_person_entry(self, lm, centroid: tuple, now: float) -> _PersonEntry:
        s = self.s
        arm_tracker = ArmTracker(ArmTrackerConfig(
            arm_above_head_tolerance=float(getattr(s, 'arm_above_head_tolerance', 0.15)),
            arm_above_head_tolerance_reclined=float(getattr(s, 'arm_above_head_tolerance_reclined', 0.38)),
            body_upright_min=float(getattr(s, 'body_upright_min', -0.15)),
            pose_visibility_min=float(getattr(s, 'pose_visibility_min', 0.45)),
            keypoint_visibility_min=float(getattr(s, 'keypoint_visibility_min', 0.30)),
            forearm_vertical_min=float(getattr(s, 'forearm_vertical_min', 0.10)),
            wrist_head_exclude_dist=float(getattr(s, 'wrist_head_exclude_dist', 0.09)),
            leg_raise_margin=float(getattr(s, 'leg_raise_margin', 0.05)),
            state_confirm_s=float(getattr(s, 'state_confirm_s', 0.20)),
            state_release_s=float(getattr(s, 'state_release_s', 0.30)),
        ))
        state_machine = GestureStateMachine(StateMachineConfig(
            cooldown_s=float(getattr(s, 'cooldown_s', 2.0)),
            snap_forearm_min=float(getattr(s, 'snap_forearm_min', 0.10)),
            hold_duration_s=float(getattr(s, 'hold_duration_s', 1.5)),
            double_snap_window_s=float(getattr(s, 'double_snap_window_s', 3.0)),
            sustain_s=float(getattr(s, 'sustain_s', 0.5)),
            snap_sustain_s=float(getattr(s, 'snap_sustain_s', 0.50)),
            snap_roll_threshold=float(getattr(s, 'snap_roll_threshold', 0.0)),
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
            last_lm=lm,
            last_genuine_live_at=now,
        )

    def _note_liveness(
        self, e: "_PersonEntry", raw_live: bool, moved: bool, now: float
    ) -> None:
        """Update a pid's confirmed-human status from this frame's checks.

        Two genuine AND MOVED passes within _LIVENESS_CONFIRM_WINDOW_S of
        each other permanently confirm the pid as human (sticky — never
        revoked short of the pid itself being pruned). Requiring actual
        keypoint displacement (not just the foreground-mask check passing)
        is what closes the gap the window alone left open: correlated
        lighting noise (a flicker, an exposure adjustment) can make the
        background-subtraction mask fire "genuine" on several frames in a
        row for a completely static object, but it can never make that
        object's own reported keypoint positions actually shift.
        """
        if not raw_live or not moved or e.confirmed_human:
            return
        if (e.last_genuine_live_at is not None
                and now - e.last_genuine_live_at <= _LIVENESS_CONFIRM_WINDOW_S):
            e.confirmed_human = True
            log.info("[%s] pid confirmed human (2 genuine, moved passes within %.0fs)",
                      self.label, _LIVENESS_CONFIRM_WINDOW_S)
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
                available.remove(best_pid)
                self._note_liveness(e, raw_live, moved, now)
            else:
                pid = self._next_pid
                self._next_pid += 1
                self._persons[pid] = self._make_person_entry(lm, centroid, now)
                log.info("[%s] New person pid=%d at (%.0f, %.0f)",
                         self.label, pid, *centroid)

    def _prune_persons(self, now: float) -> None:
        stale = [pid for pid, e in self._persons.items()
                 if now - e.last_seen > _PERSON_PRUNE_S]
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

        _cached_kps: Optional[np.ndarray] = None   # [N, 17, 3] from PoseModel
        _prev_small: Optional[np.ndarray] = None   # for motion gate
        _frames_since_yolo: int = 999              # force run on first frame
        _prev_primary_state: ArmState = ArmState.DOWN

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            proc_frame, pw, ph = self._preprocess(frame)

            # Fed every frame (not just motion-gated ones) so the model keeps
            # tracking gradual lighting drift even when nothing is moving.
            # Sampled BEFORE CLAHE (below) — CLAHE's local-contrast boost can
            # amplify sensor noise in dim areas into something that looks like
            # motion, which would undermine the liveness check it feeds.
            fg_mask = (
                bg_subtractor.apply(proc_frame)
                if getattr(self.s, 'bg_subtract_enable', True) else None
            )
            min_fg_frac = float(getattr(self.s, 'bg_subtract_min_foreground', 0.12))
            # Confirmed-human persons' positions, plus pids still inside
            # their own confirmation grace period (see _known_centroids),
            # exempt from the wrist-liveness check below (see
            # _passes_liveness_gate and _LIVENESS_CONFIRM_WINDOW_S) —
            # computed from state as of the end of the previous iteration.
            known_centroids = _known_centroids(self._persons, now)
            match_dist = _LIVENESS_EXEMPT_DIST * ((pw ** 2 + ph ** 2) ** 0.5)

            proc_frame = self._apply_low_light_enhance(proc_frame)

            # ── Motion gate ───────────────────────────────────────────────
            # Resize to 80×60 (~0.1 ms) and diff against previous frame.
            # Skip YOLO when the scene is static — reuse cached landmarks.
            # Always run YOLO when:
            #   • significant motion detected  (something is moving)
            #   • any arm already raised       (responsive snap detection)
            #   • heartbeat interval reached   (catch slow arm lifts)
            mot_thresh = float(getattr(self.s, 'motion_threshold', 0.015))
            max_skip   = int(getattr(self.s,   'yolo_max_skip',    12))

            gray  = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)
            if _prev_small is None:
                motion = True
            else:
                motion = (
                    float(cv2.absdiff(small, _prev_small).mean()) / 255.0
                ) > mot_thresh
            _prev_small = small
            _frames_since_yolo += 1

            any_arm_up = any(e.last_arm_state != ArmState.DOWN
                             for e in self._persons.values())
            run_yolo = motion or any_arm_up or (_frames_since_yolo >= max_skip)

            if run_yolo:
                # Confidence hysteresis:
                #   acquire  (no one tracked): base_conf * 1.3 — strict, rejects ghosts
                #   maintain (someone tracked): base_conf * 0.65 — bridges arm-raise dropouts
                base_conf = float(getattr(self.s, 'yolo_conf', 0.45))
                model.conf = base_conf * 0.65 if self._persons else base_conf * 1.3

                _cached_kps = model(proc_frame)    # Optional[ndarray [N,17,3]]
                _frames_since_yolo = 0
                log.debug("[%s] YOLO ran (motion=%s any_arm_up=%s persons=%d)",
                          self.label, motion, any_arm_up, len(self._persons))

                detected = _all_valid_landmarks(
                    _cached_kps, pw, ph, fg_mask=fg_mask, min_foreground_frac=min_fg_frac,
                    known_centroids=known_centroids, match_dist=match_dist,
                )
                self._match_persons(detected, pw, ph, now)
                self._prune_persons(now)

                # Feed recorder if active (keypoints for regression tests)
                if self._recorder is not None:
                    if _cached_kps is not None and _cached_kps.shape[0] > 0:
                        valid = [i for i in range(_cached_kps.shape[0])
                                 if _person_visible_kp_count(_cached_kps[i]) >= _MIN_VISIBLE_KPS]
                        kps_rec = _cached_kps[valid[0]] if valid else np.zeros((17, 3), dtype=np.float32)
                    else:
                        kps_rec = np.zeros((17, 3), dtype=np.float32)
                    self._recorder.on_frame(kps_rec, pw, ph, now)

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
                    try:
                        flat_lm, _ = _hand_pipeline.get_landmarks_from_image(proc_frame)
                        hand_lm = flat_lm
                    except Exception as e:
                        if _NoHandDetected is None or not isinstance(e, _NoHandDetected):
                            log.debug("[%s] grlib hand error: %s", self.label, e)

            log.debug("[%s] %d person(s) tracked", self.label, len(self._persons))

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
                    entry.last_arm_state = reading.state
                    if reading.state != entry.last_logged_state:
                        log.info("[%s] pid=%d state → %s",
                                 self.label, pid, reading.state.name)
                        entry.last_logged_state = reading.state
                        # Chime on rising edge — fire as early as possible so
                        # speaker latency (e.g. Alexa ~2s) lands near gesture time.
                        if (self._sonos is not None and
                                reading.state.name in ('SINGLE_UP', 'BOTH_UP') and
                                prev_state == ArmState.DOWN):
                            self._sonos.notify()
                    if reading.state.name == 'SINGLE_UP':
                        log.debug("[%s] pid=%d SINGLE_UP forearm_dy=%.3f snap_roll=%.3f",
                                  self.label, pid, reading.forearm_dy, reading.snap_roll)
                    if self._feedback:
                        self._feedback.push_reading(reading)

                gesture = entry.state_machine.tick(reading, now)
                if gesture is not None:
                    log.debug("[%s] pid=%d gesture: %s", self.label, pid, gesture)
                    self.on_candidate(gesture, 1.0, self.label)
                    if self._recorder is not None:
                        self._recorder.on_gesture(gesture, self.label)
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
            _all_dbg_kps: list[np.ndarray] = []
            if _cached_kps is not None:
                for i in range(_cached_kps.shape[0]):
                    if _person_visible_kp_count(_cached_kps[i]) < _MIN_VISIBLE_KPS:
                        continue
                    c = _person_centroid(_cached_kps[i])
                    if c is None:
                        continue
                    if _passes_liveness_gate(
                        _cached_kps[i], c, fg_mask, pw, ph, min_fg_frac,
                        known_centroids, match_dist,
                    ):
                        _all_dbg_kps.append(_cached_kps[i])

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
                            proc_frame, _all_dbg_kps, hand_lm, r, _primary_gesture
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

            if self.debug_frame_cb is not None:
                try:
                    dbg = _draw_debug(
                        proc_frame, _all_dbg_kps, hand_lm, _primary_reading, _primary_gesture
                    )
                    self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        log.info("[%s] Analyser stopped", self.label)


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, all_person_kps, hand_lm, reading, fired_gesture):
    """Draw YOLO skeleton for all detected persons + gesture state overlay.

    *all_person_kps* is a list of [17, 3] numpy arrays, one per valid person.
    """
    img = frame.copy()
    h, w = img.shape[:2]

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
        msg = "NO POSE DETECTED"
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
            (f"forearm_dy: {forearm:.3f}", (255, 255, 255)),
            (f"snap_roll:  {hand_str}", (255, 255, 255)),
            (f"upright: {reading.upright}", (255, 255, 255)),
        ]
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
