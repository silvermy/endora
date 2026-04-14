"""
cameras/analyser.py

Hybrid gesture detection: MediaPipe Pose + Hands.

Gesture set:
  WAVE_LEFT    — arm raised above head, open palm, wrist flicks left
  WAVE_RIGHT   — arm raised above head, open palm, wrist flicks right
  PALM_UP      — arm raised above head, palm rotated to face ceiling
  PALM_DOWN    — arm raised above head, palm rotated to face floor
  FIST_PUMP    — arm raised above head, closed fist, upward pump motion

Detection pipeline per frame:
  1. Pose  → is arm raised above head? (wrist above nose level)
  2. Hands → wrist flick velocity (wave left/right)
             palm orientation (up/down via wrist roll angle)
             hand shape (open/fist)
  3. Velocity tracker → directional velocity of wrist
  4. State machine → N consistent frames before firing
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Deque, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Gesture enum ──────────────────────────────────────────────────────────────

class Gesture(Enum):
    WAVE_LEFT    = auto()  # arm up, open hand, flick left
    WAVE_RIGHT   = auto()  # arm up, open hand, flick right
    PALM_UP      = auto()  # arm up, palm rotated to face ceiling
    PALM_DOWN    = auto()  # arm up, palm rotated to face floor
    FIST_PUMP    = auto()  # arm up, fist, upward pump

    def __str__(self):
        return self.name.replace("_", " ").lower()


# ── Velocity tracker ──────────────────────────────────────────────────────────

WristSample = collections.namedtuple("WristSample", ["x", "y", "t"])


class VelocityTracker:
    HISTORY = 6

    def __init__(self):
        self._samples: Deque[WristSample] = collections.deque(maxlen=self.HISTORY)

    def update(self, x: float, y: float):
        self._samples.append(WristSample(x, y, time.monotonic()))

    def velocity(self) -> tuple[float, float]:
        if len(self._samples) < 2:
            return 0.0, 0.0
        oldest = self._samples[0]
        newest = self._samples[-1]
        n = len(self._samples) - 1
        return (newest.x - oldest.x) / n, (newest.y - oldest.y) / n

    def peak_velocity(self) -> tuple[float, float]:
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


# ── Analyser ──────────────────────────────────────────────────────────────────

class CameraAnalyser(threading.Thread):

    def __init__(
        self,
        camera,
        settings,
        on_candidate: Callable[[Gesture, float, str], None],
        label: str = "cam",
        debug_frame_cb=None,
    ):
        super().__init__(daemon=True, name=f"Analyser-{label}")
        self.camera = camera
        self.s = settings
        self.on_candidate = on_candidate
        self.label = label
        self.debug_frame_cb = debug_frame_cb
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        import mediapipe as mp

        mp_pose  = mp.solutions.pose
        mp_hands = mp.solutions.hands

        _complexity = max(0, min(2, int(self.s.pose_model_complexity)))
        if _complexity != int(self.s.pose_model_complexity):
            log.warning(
                "[%s] pose_model_complexity=%s is invalid — "
                "MediaPipe Pose only accepts 0, 1, or 2. Clamping to %d.",
                self.label, self.s.pose_model_complexity, _complexity,
            )
        pose = mp_pose.Pose(
            model_complexity=_complexity,
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
        last_arm_raised = False
        consecutive_no_pose = 0
        NO_POSE_TOLERANCE = 4
        arm_raised_since: float = 0.0
        # Reset tracking if arm held still for this many seconds
        ARM_HELD_TIMEOUT_S = 5.0
        # Arm must be raised for this many consecutive frames before
        # gestures can fire — filters phantom 1-2 frame detections
        consecutive_arm_raised = 0
        ARM_RAISE_MIN_FRAMES = 5
        # Furniture-lock breaker: if pose keeps landing on furniture
        # (shoulders too low) for this many frames, recreate the model
        _furniture_rejection_streak = 0
        FURNITURE_RESET_FRAMES = 12

        log.info("[%s] Analyser running (hybrid pose+hands mode)", self.label)

        while not self._stop_evt.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # ── Optional fisheye dewarping ────────────────────────────────
            # Converts raw equidistant fisheye to flat perspective before
            # any cropping or MediaPipe processing.  Maps are built once on
            # the first frame (lazy-init so we know the actual input size).
            # Requires raw fisheye RTSP — disable in-camera dewarping first.
            if getattr(self.s, 'dewarp_enable', False):
                from cameras.dewarp import build_dewarp_maps, apply_dewarp
                cx_raw = float(getattr(self.s, 'dewarp_cx',   -1.0))
                cy_raw = float(getattr(self.s, 'dewarp_cy',   -1.0))
                dw     = int(getattr(self.s,   'dewarp_out_width',  640))
                dh     = int(getattr(self.s,   'dewarp_out_height', 480))
                fov    = float(getattr(self.s, 'dewarp_fov',   180.0))
                pan    = float(getattr(self.s, 'dewarp_pan',     0.0))
                tilt   = float(getattr(self.s, 'dewarp_tilt',   20.0))
                roll   = float(getattr(self.s, 'dewarp_roll',    0.0))
                vfov   = float(getattr(self.s, 'dewarp_vfov',   75.0))
                # Cache key — rebuild maps if any param changes
                _key = (w, h, dw, dh, fov, pan, tilt, roll, vfov, cx_raw, cy_raw)
                if getattr(self, '_dewarp_key', None) != _key:
                    self._dewarp_maps = build_dewarp_maps(
                        in_w=w, in_h=h,
                        out_w=dw, out_h=dh,
                        fisheye_fov_deg=fov,
                        pan_deg=pan,
                        tilt_deg=tilt,
                        roll_deg=roll,
                        vfov_deg=vfov,
                        cx=None if cx_raw < 0 else cx_raw,
                        cy=None if cy_raw < 0 else cy_raw,
                    )
                    self._dewarp_key = _key
                frame = apply_dewarp(frame, *self._dewarp_maps)
                h, w = frame.shape[:2]

            # ── Optional asymmetric crop (removes fisheye distortion) ────
            # frame_crop_top/bottom/left/right = % to remove from each edge
            ct = float(getattr(self.s, 'frame_crop_top',    0))
            cb = float(getattr(self.s, 'frame_crop_bottom', 0))
            cl = float(getattr(self.s, 'frame_crop_left',   0))
            cr = float(getattr(self.s, 'frame_crop_right',  0))
            # Legacy symmetric crop_pct support
            crop_pct = float(getattr(self.s, 'frame_crop_pct', 100))
            if crop_pct < 100.0 and ct == 0 and cb == 0 and cl == 0 and cr == 0:
                margin = (100.0 - crop_pct) / 2.0
                ct = cb = cl = cr = margin
            y0 = int(h * ct / 100)
            y1 = h - int(h * cb / 100)
            x0 = int(w * cl / 100)
            x1 = w - int(w * cr / 100)
            if y0 > 0 or y1 < h or x0 > 0 or x1 < w:
                proc_frame = frame[y0:y1, x0:x1]
                ph, pw = y1 - y0, x1 - x0
                crop_offset = (x0, y0)
            else:
                proc_frame = frame
                ph, pw = h, w
                crop_offset = (0, 0)

            # ── Optional CLAHE low-light enhancement ──────────────────────
            # Boosts local contrast in dark/IR images before MediaPipe sees
            # the frame. Much better than simple brightness — preserves
            # structure while making body landmarks pop against background.
            if getattr(self.s, 'low_light_enhance', False):
                clip = float(getattr(self.s, 'low_light_clip', 2.0))
                lab = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                proc_frame = cv2.cvtColor(
                    cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR
                )

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res  = pose.process(rgb)
            hand_res  = hands.process(rgb)
            rgb.flags.writeable = True

            # ── Furniture / false-pose filter ─────────────────────────────
            # If the detected skeleton's shoulders are too low in the frame
            # it's almost certainly furniture or floor geometry, not a person.
            # We reject it, and after FURNITURE_RESET_FRAMES consecutive
            # rejections we recreate the Pose model to break the tracking lock
            # (MediaPipe tracking keeps latching back to the same object).
            if pose_res and pose_res.pose_landmarks:
                _lm_f = pose_res.pose_landmarks.landmark
                _PL_f = mp.solutions.pose.PoseLandmark
                _ls_y = _lm_f[_PL_f.LEFT_SHOULDER].y
                _rs_y = _lm_f[_PL_f.RIGHT_SHOULDER].y
                _avg_sh_y = (_ls_y + _rs_y) / 2.0
                _max_sh_y = float(getattr(self.s, 'pose_shoulder_max_y', 0.85))
                if _avg_sh_y > _max_sh_y:
                    _furniture_rejection_streak += 1
                    log.debug(
                        "[%s] pose rejected: avg_shoulder_y=%.2f > max=%.2f "
                        "(furniture? streak=%d)",
                        self.label, _avg_sh_y, _max_sh_y, _furniture_rejection_streak,
                    )
                    if _furniture_rejection_streak >= FURNITURE_RESET_FRAMES:
                        log.info(
                            "[%s] Breaking furniture tracking lock — recreating pose model",
                            self.label,
                        )
                        pose.close()
                        pose = mp_pose.Pose(
                            model_complexity=_complexity,
                            min_detection_confidence=float(self.s.pose_min_detection_confidence),
                            min_tracking_confidence=float(self.s.pose_min_tracking_confidence),
                            enable_segmentation=False,
                            static_image_mode=False,
                        )
                        _furniture_rejection_streak = 0
                    pose_res = None   # treat this frame as no-pose
                else:
                    _furniture_rejection_streak = 0

            # ── 1. Arm raised above head? ─────────────────────────────────
            arm_raised, wrist_xy, raised_side = _arm_above_head(
                pose_res, self.s, pw, ph
            )

            # wrist_xy stays in proc_frame pixel space — used for velocity
            # tracking (relative movement only) and debug overlay which draws
            # on proc_frame.  No remapping to full-frame needed.

            if not arm_raised:
                pose_detected = bool(pose_res and pose_res.pose_landmarks)
                consecutive_no_pose += 1
                if consecutive_no_pose >= NO_POSE_TOLERANCE:
                    consecutive_arm_raised = 0
                if consecutive_no_pose >= NO_POSE_TOLERANCE:
                    if last_arm_raised:
                        log.debug("[%s] arm lowered — resetting", self.label)
                        velocity.reset()
                        for g in Gesture:
                            sustain_counts[g] = 0
                    last_arm_raised = False
                if log.isEnabledFor(logging.DEBUG) and consecutive_no_pose % 10 == 1:
                    if not pose_detected:
                        log.debug("[%s] NO POSE DETECTED — body not found in frame", self.label)
                    else:
                        log.debug("[%s] arm not raised (pose OK, arm down)", self.label)
                # Debug: still render frame even when arm not raised
                if self.debug_frame_cb is not None:
                    try:
                        _debug_frame_counter = getattr(self, '_dfc', 0) + 1
                        self._dfc = _debug_frame_counter
                        if _debug_frame_counter % 3 == 0:
                            dbg = _draw_debug(proc_frame, pose_res, None,
                                              0, 0, 0, 0, None, False, "unknown",
                                              consecutive_arm_raised, ARM_RAISE_MIN_FRAMES)
                            self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            consecutive_no_pose = 0
            consecutive_arm_raised += 1
            wx, wy = wrist_xy

            # Don't process gestures until arm has been raised for
            # enough consecutive frames to rule out phantom detections
            if consecutive_arm_raised < ARM_RAISE_MIN_FRAMES:
                if not last_arm_raised:
                    velocity.reset()
                last_arm_raised = True
                # Debug: render warming-up state
                if self.debug_frame_cb is not None:
                    try:
                        _debug_frame_counter = getattr(self, '_dfc', 0) + 1
                        self._dfc = _debug_frame_counter
                        if _debug_frame_counter % 3 == 0:
                            dbg = _draw_debug(proc_frame, pose_res, wrist_xy,
                                              0, 0, 0, 0, None, False, "unknown",
                                              consecutive_arm_raised, ARM_RAISE_MIN_FRAMES)
                            self.debug_frame_cb(self.label, dbg)
                    except Exception as e:
                        log.debug("[%s] debug render error: %s", self.label, e)
                continue

            if not last_arm_raised:
                log.debug("[%s] arm raised (%s side) wrist=(%.0f,%.0f)",
                          self.label, raised_side, wx, wy)
                velocity.reset()
                arm_raised_since = time.monotonic()
            last_arm_raised = True

            # Reset if arm held still too long — clears stale velocity state
            now = time.monotonic()
            if now - arm_raised_since > ARM_HELD_TIMEOUT_S:
                vx_check, _ = velocity.velocity()
                pvx_check, _ = velocity.peak_velocity()
                if abs(vx_check) < 2.0 and abs(pvx_check) < 5.0:
                    velocity.reset()
                    sustain_counts = {g: 0 for g in Gesture}
                    arm_raised_since = now
                    log.debug("[%s] arm held still — resetting tracker", self.label)

            velocity.update(wx, wy)
            vx, vy   = velocity.velocity()
            pvx, pvy = velocity.peak_velocity()

            # ── 2. Hand shape and orientation ────────────────────────────
            is_fist, palm_facing, hand_conf = _classify_hand_full(
                hand_res, self.s
            )

            # ── 3. Pick candidate ─────────────────────────────────────────
            candidate = _pick_candidate(
                vx, vy, pvx, pvy, is_fist, palm_facing, self.s
            )

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[%s] arm up | wrist=(%.0f,%.0f) vx=%.1f vy=%.1f "
                    "pvx=%.1f pvy=%.1f fist=%s palm=%s candidate=%s sustain=%s",
                    self.label, wx, wy, vx, vy, pvx, pvy,
                    is_fist, palm_facing,
                    candidate.name if candidate else "none",
                    {g.name: sustain_counts[g] for g in Gesture
                     if sustain_counts[g] > 0},
                )

            # ── 4. Sustain ────────────────────────────────────────────────
            for g in Gesture:
                if g == candidate:
                    sustain_counts[g] += 1
                else:
                    sustain_counts[g] = max(0, sustain_counts[g] - 1)

            needed = self.s.wave_sustain_frames

            if candidate and sustain_counts.get(candidate, 0) >= needed:
                confidence = min(1.0, sustain_counts[candidate] / (needed * 2))
                log.debug("[%s] FIRING %s conf=%.2f", self.label, candidate, confidence)
                self.on_candidate(candidate, confidence, self.label)
                sustain_counts = {g: 0 for g in Gesture}
                velocity.reset()
                consecutive_arm_raised = 0

            # ── Debug overlay ─────────────────────────────────────────────
            if self.debug_frame_cb is not None:
                try:
                    _debug_frame_counter = getattr(self, '_dfc', 0) + 1
                    self._dfc = _debug_frame_counter
                    if _debug_frame_counter % 3 == 0:
                        dbg = _draw_debug(
                            proc_frame, pose_res,
                            wrist_xy if arm_raised else None,
                            vx, vy, pvx, pvy, candidate, is_fist, palm_facing,
                            consecutive_arm_raised, ARM_RAISE_MIN_FRAMES,
                        )
                        self.debug_frame_cb(self.label, dbg)
                except Exception as e:
                    log.debug("[%s] debug render error: %s", self.label, e)

        pose.close()
        hands.close()
        log.info("[%s] Analyser stopped", self.label)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _arm_above_head(
    pose_res, settings, frame_w: int, frame_h: int
) -> tuple[bool, tuple[float, float], str]:
    """
    Returns (raised, (wrist_x, wrist_y), side).

    Uses absolute wrist Y position in frame rather than wrist-vs-shoulder.
    This works regardless of camera angle.

    arm_above_head_tolerance is now the absolute Y threshold:
      0.70 = wrist must be in top 70% of frame (y < 0.70)
      0.65 = wrist must be in top 65% of frame (stricter)

    From your logs: raised hand wrist_y ≈ 0.644-0.686, resting wrist_y > 0.68
    Set to 0.70 to catch raised hands. Adjust lower if false triggers occur.
    """
    if not pose_res or not pose_res.pose_landmarks:
        return False, (0.0, 0.0), ""

    import mediapipe as mp
    lm  = pose_res.pose_landmarks.landmark
    PL  = mp.solutions.pose.PoseLandmark

    pairs = [
        ("RIGHT", PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
        ("LEFT",  PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,  PL.LEFT_WRIST),
    ]

    for side, sh_id, el_id, wr_id in pairs:
        sh = lm[sh_id]
        el = lm[el_id]
        wr = lm[wr_id]

        # Wrist must appear in upper portion of frame
        wrist_in_upper_frame = wr.y < settings.arm_above_head_tolerance
        # Elbow must be at or above shoulder level (arm up, not just wrist)
        elbow_elevated = el.y < sh.y + 0.15

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  [arm-check] %s wr_y=%.3f tol=%.3f → in_upper=%s el_y=%.3f sh_y=%.3f elbow=%s",
                side, wr.y, settings.arm_above_head_tolerance,
                wrist_in_upper_frame, el.y, sh.y, elbow_elevated,
            )

        if wrist_in_upper_frame and elbow_elevated:
            return True, (wr.x * frame_w, wr.y * frame_h), side

    return False, (0.0, 0.0), ""



def _classify_hand_full(
    hand_res, settings
) -> tuple[bool, str, float]:
    """
    Returns (is_fist, palm_facing, confidence).

    palm_facing values:
      'camera'  — palm faces the camera (neutral / waving position)
      'up'      — palm faces ceiling (wrist bent backward)
      'down'    — palm faces floor (wrist bent forward)
      'unknown' — hand not detected or ambiguous

    Palm orientation is determined by the z-depth of the middle finger MCP
    relative to the wrist. MediaPipe Hands provides z coordinates that encode
    depth within the hand — when the palm faces up the finger MCPs have
    negative z relative to wrist; when facing down they have positive z.
    """
    if not hand_res or not hand_res.multi_hand_landmarks:
        return False, "unknown", 0.0

    lm = hand_res.multi_hand_landmarks[0].landmark

    # ── Fist detection ────────────────────────────────────────────────────
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]
    MCPS = [5,  9, 13, 17]

    curled = 0
    for tip_i, pip_i, mcp_i in zip(TIPS, PIPS, MCPS):
        if lm[tip_i].y > lm[pip_i].y and lm[tip_i].y > lm[mcp_i].y:
            curled += 1

    frac    = curled / 4.0
    is_fist = frac >= settings.fist_curl_threshold
    conf    = frac if is_fist else (1.0 - frac)

    # ── Palm orientation ──────────────────────────────────────────────────
    # Use the z coordinate of the middle finger MCP (landmark 9) vs wrist (0)
    # z is normalised: negative = closer to camera, positive = further
    # Palm up  (facing ceiling): knuckles point up, z_mcp << z_wrist
    # Palm down (facing floor):  knuckles point down, z_mcp >> z_wrist
    wrist_z  = lm[0].z
    middle_z = lm[9].z   # middle finger MCP
    index_z  = lm[5].z   # index finger MCP
    ring_z   = lm[13].z  # ring finger MCP
    avg_mcp_z = (middle_z + index_z + ring_z) / 3.0
    z_diff = avg_mcp_z - wrist_z

    palm_thresh = settings.palm_orientation_threshold
    if z_diff < -palm_thresh:
        palm_facing = "up"
    elif z_diff > palm_thresh:
        palm_facing = "down"
    else:
        palm_facing = "camera"

    return is_fist, palm_facing, conf


def _pick_candidate(
    vx: float, vy: float,
    pvx: float, pvy: float,
    is_fist: bool,
    palm_facing: str,
    settings,
) -> Optional[Gesture]:
    """
    Require BOTH average velocity AND peak velocity to exceed threshold.
    This prevents stale peak values from triggering on a still hand.
    Average velocity catches sustained movement; requiring it alongside
    peak means a single old high-velocity sample can't fire alone.
    """
    wh = settings.wave_velocity_threshold_px
    vh = settings.vertical_velocity_threshold_px

    # Palm orientation — static gesture, no velocity required
    if not is_fist:
        if palm_facing == "up":
            return Gesture.PALM_UP
        if palm_facing == "down":
            return Gesture.PALM_DOWN

    # Fist pump — fist moving upward, peak velocity confirms intentional move
    if is_fist:
        if pvy < -vh and abs(pvx) < abs(pvy):
            return Gesture.FIST_PUMP

    # Wave — open hand, horizontal movement.
    # Use PEAK velocity as the primary signal — a fast wrist flick spikes
    # the peak even if the 6-frame average is diluted by still frames.
    # Average velocity is only used as a minimum sanity check (> 1/3 threshold)
    # to confirm the hand actually moved and the peak isn't just stale noise.
    if not is_fist:
        abs_pvx = abs(pvx)
        abs_pvy = abs(pvy)
        abs_vx  = abs(vx)

        if abs_pvx > wh and abs_pvx > abs_pvy and abs_vx > wh * 0.2:
            # If camera is mirrored, flip left/right interpretation
            effective_pvx = -pvx if settings.mirror_camera else pvx
            return Gesture.WAVE_LEFT if effective_pvx < 0 else Gesture.WAVE_RIGHT

    return None


# ── Debug overlay ─────────────────────────────────────────────────────────────

def _draw_debug(frame, pose_res, wrist_xy, vx, vy, pvx, pvy,
                candidate, is_fist, palm_facing,
                consec_raised, min_frames):
    """Draw skeleton + gesture state overlay onto a copy of frame."""
    import mediapipe as mp
    img = frame.copy()
    h, w = img.shape[:2]

    # Draw pose skeleton — or a bright warning if pose was not detected
    if pose_res and pose_res.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            img,
            pose_res.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 200, 0), thickness=2),
        )
    else:
        # Pose model did not detect a body — make this very obvious
        msg = "NO POSE DETECTED"
        fs = max(0.6, w / 800)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        tx = (w - tw) // 2
        ty = 60
        cv2.rectangle(img, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      (0, 0, 180), -1)
        cv2.putText(img, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    # Wrist marker + velocity arrow
    if wrist_xy:
        wx, wy = int(wrist_xy[0]), int(wrist_xy[1])
        color = (255, 255, 0) if consec_raised >= min_frames else (0, 128, 255)
        cv2.circle(img, (wx, wy), 12, color, -1)
        cv2.circle(img, (wx, wy), 12, (0, 0, 0), 2)
        # Peak velocity arrow
        ax = int(wx + pvx * 2)
        ay = int(wy + pvy * 2)
        cv2.arrowedLine(img, (wx, wy), (ax, ay), (255, 0, 255), 3, tipLength=0.3)

    # Status panel — bottom-left, scaled to frame size
    ready = consec_raised >= min_frames
    arm_state = "ARM READY" if ready else f"warm {consec_raised}/{min_frames}"
    cand_str = candidate.name if candidate else "none"
    lines = [
        (arm_state, (0, 255, 100) if ready else (0, 165, 255)),
        (f"pvx={pvx:.0f} vx={vx:.0f}", (255, 255, 255)),
        (f"pvy={pvy:.0f} vy={vy:.0f}", (255, 255, 255)),
        (f"fist={is_fist} palm={palm_facing}", (255, 255, 255)),
        (f"cand: {cand_str}", (255, 255, 0) if cand_str != "none" else (160, 160, 160)),
    ]
    fs = max(0.35, w / 1800)          # font scale relative to frame width
    lh = int(fs * 42)
    pad = int(fs * 12)
    panel_h = len(lines) * lh + pad * 2
    panel_w = int(w * 0.30)
    y_start = h - panel_h - 6
    overlay = img.copy()
    cv2.rectangle(overlay, (4, y_start - 2), (4 + panel_w, y_start + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (line, color) in enumerate(lines):
        y = y_start + pad + i * lh + lh - 4
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

    return img
