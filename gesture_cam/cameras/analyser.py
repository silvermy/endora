"""
cameras/analyser.py

Runs in its own thread, consuming frames from CameraCapture and emitting
raw gesture candidates to the fusion layer.

Pipeline per frame:
  1. MediaPipe Pose  → detect body skeleton, check if arm is raised
  2. MediaPipe Hands → when arm is raised, classify hand shape (open / fist)
  3. Velocity tracker → track wrist XY history to detect wave direction /
                         vertical movement
  4. State machine    → require N consistent frames before emitting a candidate
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Deque, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Gesture enum ──────────────────────────────────────────────────────────────

class Gesture(Enum):
    WAVE_LEFT       = auto()   # open hand, arm up, moving left
    WAVE_RIGHT      = auto()   # open hand, arm up, moving right
    PALM_UP         = auto()   # flat open palm, arm up, moving up
    PALM_DOWN       = auto()   # flat open palm, arm up, moving down
    FIST_GESTURE    = auto()   # fist raised (direction agnostic)

    def __str__(self):
        return self.name.replace("_", " ").lower()


# ── Wrist history ─────────────────────────────────────────────────────────────

WristSample = collections.namedtuple("WristSample", ["x", "y", "t"])


class VelocityTracker:
    """
    Rolling window of wrist positions → smoothed velocity.
    Uses simple first-last delta rather than linear regression so that
    a single fast sweep across 3-4 frames registers even if intermediate
    frames are dropped (common with RTSP streams).
    """

    HISTORY = 6

    def __init__(self):
        self._samples: Deque[WristSample] = collections.deque(maxlen=self.HISTORY)

    def update(self, x: float, y: float):
        self._samples.append(WristSample(x, y, time.monotonic()))

    def velocity(self) -> tuple[float, float]:
        """
        Return (vx, vy) in pixels-per-frame.
        Uses delta between oldest and newest sample, normalised by count.
        This is more responsive to fast sweeps than linear regression.
        """
        if len(self._samples) < 2:
            return 0.0, 0.0
        oldest = self._samples[0]
        newest = self._samples[-1]
        n = len(self._samples) - 1  # number of intervals
        dt = newest.t - oldest.t
        if dt < 0.01:
            return 0.0, 0.0
        # Express as pixels per frame assuming ~15fps actual detection rate
        fps_est = n / dt
        vx = (newest.x - oldest.x) / n if fps_est > 0 else 0.0
        vy = (newest.y - oldest.y) / n if fps_est > 0 else 0.0
        return vx, vy

    def peak_velocity(self) -> tuple[float, float]:
        """
        Return the maximum frame-to-frame velocity seen in the window.
        Catches brief fast sweeps that the smoothed average misses.
        """
        if len(self._samples) < 2:
            return 0.0, 0.0
        max_vx = max_vy = 0.0
        samples = list(self._samples)
        for i in range(1, len(samples)):
            dvx = abs(samples[i].x - samples[i-1].x)
            dvy = abs(samples[i].y - samples[i-1].y)
            if dvx > abs(max_vx):
                max_vx = samples[i].x - samples[i-1].x
            if dvy > abs(max_vy):
                max_vy = samples[i].y - samples[i-1].y
        return max_vx, max_vy

    def reset(self):
        self._samples.clear()


# ── Per-camera analyser ───────────────────────────────────────────────────────

class CameraAnalyser(threading.Thread):
    """
    Reads frames from a CameraCapture, runs MediaPipe, and calls
    `on_candidate(gesture, confidence, source_label)` when a gesture
    is reliably detected.
    """

    def __init__(
        self,
        camera,           # cameras.capture.RtspCapture
        settings,         # config.settings.Settings
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        import mediapipe as mp

        mp_pose  = mp.solutions.pose
        mp_hands = mp.solutions.hands

        pose = mp_pose.Pose(
            model_complexity=int(self.s.pose_model_complexity),
            min_detection_confidence=float(self.s.pose_min_detection_confidence),
            min_tracking_confidence=float(self.s.pose_min_tracking_confidence),
            enable_segmentation=False,
            static_image_mode=False,
        )
        hands = mp_hands.Hands(
            max_num_hands=int(self.s.hand_model_max_hands),
            min_detection_confidence=float(self.s.hand_min_detection_confidence),
            min_tracking_confidence=float(self.s.hand_min_tracking_confidence),
            static_image_mode=False,
        )

        velocity = VelocityTracker()
        sustain_counts: dict[Gesture, int] = {g: 0 for g in Gesture}
        last_frame_arm_raised = False

        import cv2

        log.info("[%s] Analyser running", self.label)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            pose_res  = pose.process(rgb)
            hand_res  = hands.process(rgb)

            rgb.flags.writeable = True

            # ── Debug: log raw landmark positions every frame ─────────────
            if log.isEnabledFor(logging.DEBUG):
                if pose_res and pose_res.pose_landmarks:
                    import mediapipe as _mp
                    _PL = _mp.solutions.pose.PoseLandmark
                    _lm = pose_res.pose_landmarks.landmark
                    _rsh = _lm[_PL.RIGHT_SHOULDER]
                    _rwr = _lm[_PL.RIGHT_WRIST]
                    _lsh = _lm[_PL.LEFT_SHOULDER]
                    _lwr = _lm[_PL.LEFT_WRIST]
                    log.debug(
                        "[%s] pose detected | "
                        "R: sh_y=%.3f wr_y=%.3f diff=%.3f | "
                        "L: sh_y=%.3f wr_y=%.3f diff=%.3f | "
                        "threshold=%.3f",
                        self.label,
                        _rsh.y, _rwr.y, _rsh.y - _rwr.y,
                        _lsh.y, _lwr.y, _lsh.y - _lwr.y,
                        self.s.arm_raised_wrist_above_shoulder_frac,
                    )
                else:
                    log.debug("[%s] no pose detected in frame", self.label)

            # ── 1. Arm-raised check ───────────────────────────────────────
            arm_raised, wrist_xy, hand_side = _check_arm_raised(
                pose_res, self.s, w, h
            )

            if not arm_raised:
                if last_frame_arm_raised:
                    log.debug("[%s] arm lowered — resetting velocity", self.label)
                velocity.reset()
                for g in Gesture:
                    sustain_counts[g] = 0
                last_frame_arm_raised = False
                continue

            if not last_frame_arm_raised:
                log.debug("[%s] arm RAISED on %s side", self.label, hand_side)
                velocity.reset()
            last_frame_arm_raised = True

            wx, wy = wrist_xy
            velocity.update(wx, wy)
            vx, vy = velocity.velocity()
            pvx, pvy = velocity.peak_velocity()

            # Use whichever is larger — average or peak — so brief fast
            # sweeps register even if smoothed average is low
            eff_vx = pvx if abs(pvx) > abs(vx) else vx
            eff_vy = pvy if abs(pvy) > abs(vy) else vy

            # ── 2. Hand shape ─────────────────────────────────────────────
            is_fist, hand_conf = _classify_hand(hand_res, self.s)

            # ── 3. Candidate gesture ──────────────────────────────────────
            candidate = _pick_candidate(eff_vx, eff_vy, is_fist, self.s)

            log.debug(
                "[%s] arm up | wrist=(%.0f,%.0f) vx=%.1f vy=%.1f "
                "pvx=%.1f pvy=%.1f fist=%s hand_conf=%.2f candidate=%s sustain=%s",
                self.label, wx, wy, vx, vy, pvx, pvy,
                is_fist, hand_conf,
                candidate.name if candidate else "none",
                {g.name: sustain_counts[g] for g in Gesture if sustain_counts[g] > 0},
            )

            # ── 4. Sustain — require N consecutive matching frames ─────────
            for g in Gesture:
                if g == candidate:
                    sustain_counts[g] += 1
                else:
                    sustain_counts[g] = max(0, sustain_counts[g] - 1)

            needed = (
                self.s.wave_sustain_frames
                if candidate in (Gesture.WAVE_LEFT, Gesture.WAVE_RIGHT)
                else self.s.vertical_sustain_frames
            )

            if candidate and sustain_counts.get(candidate, 0) >= needed:
                confidence = min(1.0, sustain_counts[candidate] / (needed * 2))
                log.debug(
                    "[%s] candidate=%s conf=%.2f vx=%.1f vy=%.1f fist=%s",
                    self.label, candidate, confidence, vx, vy, is_fist,
                )
                self.on_candidate(candidate, confidence, self.label)
                # Reset this gesture's count to avoid repeated firing
                sustain_counts[candidate] = 0

        pose.close()
        hands.close()
        log.info("[%s] Analyser stopped", self.label)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_arm_raised(
    pose_res, settings, frame_w: int, frame_h: int
) -> tuple[bool, tuple[float, float], str]:
    """
    Returns (arm_raised, (wrist_x, wrist_y), side).

    MediaPipe Y coordinate: 0 = top of frame, 1 = bottom.
    So a HIGHER wrist position = SMALLER y value.

    With a ceiling-mounted camera looking down, even a fully raised arm
    will have wrist_y > shoulder_y (wrist appears lower in the frame than
    the shoulder).  We therefore check that the wrist is within a threshold
    of the shoulder rather than strictly above it.

    arm_raised_wrist_above_shoulder_frac:
      Positive (e.g. 0.10)  → wrist must be 10% of frame height ABOVE shoulder
                               (only works if camera is at eye level or below)
      Negative (e.g. -0.15) → wrist may be up to 15% of frame height BELOW
                               shoulder (correct for high/ceiling cameras)

    Read from the debug log: when your arm is raised, note the diff value.
    Set this threshold to a value slightly more negative than that diff.
    E.g. if raised-arm diff is -0.08, set threshold to -0.12.
    """
    if not pose_res or not pose_res.pose_landmarks:
        return False, (0.0, 0.0), ""

    import mediapipe as mp
    lm = pose_res.pose_landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    pairs = [
        ("RIGHT", PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
        ("LEFT",  PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,  PL.LEFT_WRIST),
    ]

    for side, sh_id, el_id, wr_id in pairs:
        sh = lm[sh_id]
        el = lm[el_id]
        wr = lm[wr_id]

        # diff = shoulder_y - wrist_y
        # Positive → wrist is above shoulder in frame (camera at eye level)
        # Negative → wrist is below shoulder in frame (high/ceiling camera)
        # Arm is raised when diff > threshold (threshold is often negative)
        wrist_diff = sh.y - wr.y
        elbow_diff = sh.y - el.y

        wrist_raised = wrist_diff > settings.arm_raised_wrist_above_shoulder_frac
        elbow_raised = elbow_diff > settings.arm_raised_elbow_above_shoulder_frac

        if wrist_raised and elbow_raised:
            wx = wr.x * frame_w
            wy = wr.y * frame_h
            return True, (wx, wy), side

    return False, (0.0, 0.0), ""


def _classify_hand(hand_res, settings) -> tuple[bool, float]:
    """
    Returns (is_fist, confidence).
    Uses finger-curl heuristic on MediaPipe hand landmarks.
    """
    if not hand_res or not hand_res.multi_hand_landmarks:
        return False, 0.0

    landmarks = hand_res.multi_hand_landmarks[0].landmark

    # Finger tip and pip (knuckle) indices for 4 fingers (excluding thumb)
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]
    MCPS = [5,  9, 13, 17]

    curled = 0
    for tip_i, pip_i, mcp_i in zip(TIPS, PIPS, MCPS):
        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        mcp = landmarks[mcp_i]
        # Finger is curled if tip is below pip (larger Y) relative to mcp
        if tip.y > pip.y and tip.y > mcp.y:
            curled += 1

    frac = curled / 4.0
    is_fist = frac >= settings.fist_curl_threshold
    confidence = frac if is_fist else (1.0 - frac)
    return is_fist, confidence


def _pick_candidate(
    vx: float, vy: float, is_fist: bool, settings
) -> Optional[Gesture]:
    """
    Given horizontal/vertical velocity and hand shape, return the best gesture.
    Priority: fist > horizontal wave > vertical move.
    """
    wh = settings.wave_velocity_threshold_px
    vh = settings.vertical_velocity_threshold_px

    if is_fist:
        return Gesture.FIST_GESTURE

    abs_vx, abs_vy = abs(vx), abs(vy)

    if abs_vx > wh and abs_vx > abs_vy:
        return Gesture.WAVE_LEFT if vx < 0 else Gesture.WAVE_RIGHT

    if abs_vy > vh and abs_vy > abs_vx:
        return Gesture.PALM_UP if vy < 0 else Gesture.PALM_DOWN

    return None
