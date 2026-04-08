"""
cameras/analyser.py

Gesture detection using MediaPipe Hands only — no pose/skeleton required.

This approach detects Bewitched-style wrist flicks from any body position:
sitting, standing, laying down, facing any direction. The hand just needs
to be visible in frame.

Pipeline per frame:
  1. MediaPipe Hands  → detect hand landmarks anywhere in frame
  2. Wrist tracking   → track wrist position history
  3. Velocity         → compute directional velocity of the wrist
  4. Hand shape       → open palm vs fist
  5. State machine    → require N consistent frames before emitting

The "arm raised" qualifier from the pose-based approach is replaced by a
minimum velocity threshold — a deliberate flick hits 20+ px/frame, idle
hand movement is typically under 5 px/frame.
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
    WAVE_LEFT       = auto()
    WAVE_RIGHT      = auto()
    PALM_UP         = auto()
    PALM_DOWN       = auto()
    FIST_GESTURE    = auto()

    def __str__(self):
        return self.name.replace("_", " ").lower()


# ── Wrist history / velocity ──────────────────────────────────────────────────

WristSample = collections.namedtuple("WristSample", ["x", "y", "t"])


class VelocityTracker:
    """
    Tracks wrist position history and computes both smoothed and peak velocity.
    Peak velocity catches brief sharp flicks even if the average is low.
    Only considers recent samples for peak to avoid stale values persisting.
    """

    HISTORY = 6

    def __init__(self):
        self._samples: Deque[WristSample] = collections.deque(maxlen=self.HISTORY)

    def update(self, x: float, y: float):
        self._samples.append(WristSample(x, y, time.monotonic()))

    def velocity(self) -> tuple[float, float]:
        """Smoothed velocity — delta between oldest and newest sample."""
        if len(self._samples) < 2:
            return 0.0, 0.0
        oldest = self._samples[0]
        newest = self._samples[-1]
        n = len(self._samples) - 1
        vx = (newest.x - oldest.x) / n
        vy = (newest.y - oldest.y) / n
        return vx, vy

    def peak_velocity(self) -> tuple[float, float]:
        """Peak frame-to-frame velocity in last 3 samples."""
        samples = list(self._samples)
        recent = samples[-3:] if len(samples) >= 3 else samples
        if len(recent) < 2:
            return 0.0, 0.0
        max_vx = max_vy = 0.0
        for i in range(1, len(recent)):
            dvx = recent[i].x - recent[i-1].x
            dvy = recent[i].y - recent[i-1].y
            if abs(dvx) > abs(max_vx):
                max_vx = dvx
            if abs(dvy) > abs(max_vy):
                max_vy = dvy
        return max_vx, max_vy

    def reset(self):
        self._samples.clear()


# ── Per-camera analyser ───────────────────────────────────────────────────────

class CameraAnalyser(threading.Thread):
    """
    Reads frames from an RtspCapture, runs MediaPipe Hands, and calls
    on_candidate(gesture, confidence, source_label) when a gesture fires.
    """

    def __init__(
        self,
        camera,
        settings,
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

    def run(self):
        import mediapipe as mp
        import cv2

        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
            max_num_hands=int(self.s.hand_model_max_hands),
            min_detection_confidence=float(self.s.hand_min_detection_confidence),
            min_tracking_confidence=float(self.s.hand_min_tracking_confidence),
            static_image_mode=False,
        )

        velocity = VelocityTracker()
        sustain_counts: dict[Gesture, int] = {g: 0 for g in Gesture}
        last_frame_had_hand = False

        log.info("[%s] Analyser running (hands-only mode)", self.label)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_res = hands.process(rgb)
            rgb.flags.writeable = True

            # ── 1. Hand detected? ─────────────────────────────────────────
            if not hand_res or not hand_res.multi_hand_landmarks:
                if last_frame_had_hand:
                    log.debug("[%s] hand lost — resetting", self.label)
                    velocity.reset()
                    for g in Gesture:
                        sustain_counts[g] = 0
                last_frame_had_hand = False

                if log.isEnabledFor(logging.DEBUG):
                    log.debug("[%s] no hand detected", self.label)
                continue

            # Use first detected hand
            landmarks = hand_res.multi_hand_landmarks[0].landmark

            # Wrist is landmark 0 in MediaPipe Hands
            wrist = landmarks[0]
            wx = wrist.x * w
            wy = wrist.y * h

            if not last_frame_had_hand:
                log.debug("[%s] hand detected at (%.0f, %.0f)", self.label, wx, wy)
                velocity.reset()
            last_frame_had_hand = True

            velocity.update(wx, wy)
            vx, vy = velocity.velocity()
            pvx, pvy = velocity.peak_velocity()

            # Use whichever is larger — avg or peak
            eff_vx = pvx if abs(pvx) > abs(vx) else vx
            eff_vy = pvy if abs(pvy) > abs(vy) else vy

            # ── 2. Hand shape ─────────────────────────────────────────────
            is_fist, hand_conf = _classify_hand(landmarks, self.s)

            # ── 3. Candidate gesture ──────────────────────────────────────
            candidate = _pick_candidate(eff_vx, eff_vy, is_fist, self.s)

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[%s] hand (%.0f,%.0f) vx=%.1f vy=%.1f "
                    "pvx=%.1f pvy=%.1f fist=%s conf=%.2f candidate=%s sustain=%s",
                    self.label, wx, wy, vx, vy, pvx, pvy,
                    is_fist, hand_conf,
                    candidate.name if candidate else "none",
                    {g.name: sustain_counts[g] for g in Gesture if sustain_counts[g] > 0},
                )

            # ── 4. Sustain ────────────────────────────────────────────────
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
                    "[%s] FIRING candidate=%s conf=%.2f",
                    self.label, candidate, confidence,
                )
                self.on_candidate(candidate, confidence, self.label)
                sustain_counts[candidate] = 0

        hands.close()
        log.info("[%s] Analyser stopped", self.label)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_hand(landmarks, settings) -> tuple[bool, float]:
    """
    Classify hand as fist or open using finger curl heuristic.
    landmarks: list of MediaPipe hand landmark objects.
    """
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]
    MCPS = [5,  9, 13, 17]

    curled = 0
    for tip_i, pip_i, mcp_i in zip(TIPS, PIPS, MCPS):
        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        mcp = landmarks[mcp_i]
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
    Select gesture from velocity and hand shape.
    Fist takes priority over directional gestures.
    Requires minimum velocity to filter idle hand movement.
    """
    wh = settings.wave_velocity_threshold_px
    vh = settings.vertical_velocity_threshold_px

    if is_fist:
        # Fist still needs minimum movement to avoid triggering when
        # you just happen to be holding your hand in a fist shape
        if abs(vx) > wh * 0.5 or abs(vy) > vh * 0.5:
            return Gesture.FIST_GESTURE

    abs_vx, abs_vy = abs(vx), abs(vy)

    if abs_vx > wh and abs_vx > abs_vy:
        return Gesture.WAVE_LEFT if vx < 0 else Gesture.WAVE_RIGHT

    if abs_vy > vh and abs_vy > abs_vx:
        return Gesture.PALM_UP if vy < 0 else Gesture.PALM_DOWN

    return None
