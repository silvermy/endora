"""
cameras/arm_tracker.py

Pure classifier: pose landmarks → ArmState.

No pose-backend dependency in the classifier itself — it takes a dict-like
landmarks object with x/y/visibility per landmark index, so tests can pass
fake landmarks without installing any inference library.

Geometry
--------
Everything is measured in PIXEL space and normalised by the person's own
arm length. That combination is what makes the thresholds mean the same
thing regardless of how far away the person is, what shape the frame is,
and whether they are standing, sitting or lying on a couch.

Landmarks arrive normalised to [0, 1] on each axis *independently*, so on a
non-square frame — e.g. the 1280x640 a fisheye dewarp produces — a distance
computed straight from them mixes two different units and is meaningless.
Converting to pixels first is not a detail; it is a correctness fix.

Two numbers describe an arm:

  elevation = (shoulder_y - wrist_y) / |shoulder - wrist|
      +1.0 = wrist straight above the shoulder, 0.0 = horizontal,
      -1.0 = hanging straight down. This is the sine of the arm's angle
      above the horizon: dimensionless, so distance and body size cancel.

  extension = |shoulder - wrist| / (|shoulder - elbow| + |elbow - wrist|)
      1.0 = perfectly straight arm, ~0.71 = elbow bent 90 degrees.
      A hand held against your own face scores low here, which is why no
      separate "wrist near the nose" veto is needed any more.

"Arm extended straight up" is then exactly: elevation high AND extension
high. There is no posture detection, no separate reclined threshold, and
no frame-fraction margin to re-tune whenever the camera moves.

Why image-vertical and not the body's own axis? Because raising your arm is
a statement about gravity, not anatomy. Standing, the two agree. Lying on a
couch they do not — and what the camera actually sees when someone reclined
raises their arm is the wrist moving up the frame, not along their spine.
Image-vertical is (modulo camera roll) world-vertical, so one threshold
covers every posture.
"""
from __future__ import annotations

import dataclasses
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional, Protocol, Tuple

import numpy as np


class ArmState(Enum):
    DOWN        = auto()  # neither arm raised
    SINGLE_UP   = auto()  # one arm raised above head
    BOTH_UP     = auto()  # both arms raised above head
    T_POSE      = auto()  # both arms extended horizontal
    CROSS_ARMS  = auto()  # arms crossed in front of chest


class Side(Enum):
    LEFT  = auto()
    RIGHT = auto()


@dataclass
class ArmReading:
    """Output of ArmTracker.classify()."""
    state: ArmState
    # For SINGLE_UP only: which arm is raised + wrist pixel location.
    raised_side: Optional[Side] = None
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    # ── Primary geometry (SINGLE_UP; also populated for the raised side) ──
    # How far above horizontal the arm points: +1 straight up, 0 level,
    # -1 straight down. Scale-free and posture-free.
    elevation: float = 0.0
    # How straight the arm is: 1.0 fully extended, ~0.71 elbow at 90 degrees.
    extension: float = 0.0
    # Shoulder-to-wrist distance in pixels — used to normalise the movement
    # thresholds, and reported for diagnosing foreshortening.
    arm_len_px: float = 0.0
    # Forearm verticality (elbow_y - wrist_y, frame fraction). No longer used
    # for classification; kept because it is a genuine quantity and appears
    # in historical feedback logs.
    forearm_dy: float = 0.0
    # True if the body pose is upright (hips below shoulders, torso more
    # vertical than horizontal). Reported for diagnostics only — no
    # threshold depends on it any more.
    upright: bool = True
    # Palm roll from grlib hand landmarks; only populated when hand_lm is
    # provided to classify(). See _hand_snap_roll.
    snap_roll: float = 0.0
    # True when an arm is on its way up but not yet a confirmed raise — used
    # to fire the chime early to compensate for speaker latency.
    arm_rising: bool = False
    # ── Trajectory evidence (SINGLE_UP only) ─────────────────────────────
    #   rose_recently — this arm was seen low, or has climbed by
    #     rise_elevation_delta, within raise_travel_window_s. Distinguishes
    #     a deliberate lift from an arm that has simply been resting in a
    #     raised-looking position (armrest, backrest, propped on a cushion).
    #   wrist_still — the wrist has held position, so the arm is being HELD
    #     up rather than passing through on the way to grab something.
    rose_recently: bool = True
    wrist_still: bool = True
    # How much this arm's elevation has climbed within the rise window.
    rise_delta: float = 0.0
    # ── Flourish (the actual gesture: a sweep, not a held pose) ───────────
    # sweep_climb — elevation gained since this arm's lowest point inside
    #   flourish_window_s. A full sweep from hanging-down to straight-up is
    #   ~2.0; starting from an armrest, ~0.7.
    # sweep_rate  — that climb divided by how long it took, in elevation
    #   units per second. This is the number that separates a deliberate
    #   flourish from every static pose we have ever had trouble with: a
    #   resting arm's rate is ~0 no matter how much it looks like a raise.
    # Both are computed for whichever arm is highest, including while the
    # state is still DOWN, so the chime can fire on the way up.
    sweep_climb: float = 0.0
    sweep_rate: float = 0.0
    # True when the climb was progressive — the arm was actually observed
    # passing through intermediate elevations rather than jumping between
    # two extremes. A real sweep is progressive; a keypoint flickering
    # between a good and a garbage position is bimodal and is not, which is
    # what stops noise from being mistaken for a fast gesture.
    sweep_progressive: bool = False


# ── Landmark protocol for type hints ──────────────────────────────────────────

class _Point(Protocol):
    x: float
    y: float
    visibility: float


class _Landmarks(Protocol):
    def __getitem__(self, idx: int) -> _Point: ...


# Landmark indices used by ArmTracker.  These match MediaPipe PoseLandmark
# values; the YOLO backend remaps COCO indices to these before calling classify().
NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26

_Pt = Tuple[float, float]


def _hand_snap_roll(hand_lm: np.ndarray) -> float:
    """Palm-orientation signal from a flat grlib hand-landmark array
    (1 hand, 63 floats; MediaPipe indices WRIST=0, INDEX_FINGER_MCP=5,
    MIDDLE_FINGER_MCP=9, PINKY_MCP=17).

    roll = (index_mcp.x - pinky_mcp.x) / hand_size, where hand_size is the
    wrist->middle-MCP distance — an apparent-hand-size proxy that doesn't
    collapse when the knuckle line is edge-on.  |roll| ~ 0.8-1.1 when the
    palm or back of the hand faces the camera, ~0-0.3 when edge-on; the sign
    encodes palm-vs-back / left-vs-right hand. Clamped to +/-1.5.
    """
    if len(hand_lm) < 21 * 3:
        return 0.0
    wr_x, wr_y = float(hand_lm[0]), float(hand_lm[1])          # WRIST
    md_x, md_y = float(hand_lm[9 * 3]), float(hand_lm[9 * 3 + 1])  # MIDDLE_MCP
    hand_size = ((md_x - wr_x) ** 2 + (md_y - wr_y) ** 2) ** 0.5
    if hand_size < 1e-6:
        return 0.0
    idx_mcp_x = float(hand_lm[5 * 3])    # INDEX_FINGER_MCP.x
    pnk_mcp_x = float(hand_lm[17 * 3])   # PINKY_MCP.x
    roll = (idx_mcp_x - pnk_mcp_x) / hand_size
    return max(-1.5, min(1.5, roll))


def _dist(a: _Pt, b: _Pt) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _arm_metrics(shoulder: _Pt, elbow: _Pt, wrist: _Pt) -> Tuple[float, float, float]:
    """Return (elevation, extension, arm_len_px) for one arm.

    All inputs are pixel coordinates with y increasing downward.
    """
    arm_len = _dist(shoulder, wrist)
    if arm_len < 1e-6:
        return 0.0, 0.0, 0.0
    elevation = (shoulder[1] - wrist[1]) / arm_len
    segments = _dist(shoulder, elbow) + _dist(elbow, wrist)
    extension = arm_len / segments if segments > 1e-6 else 0.0
    return elevation, extension, arm_len


# ── Tracker ───────────────────────────────────────────────────────────────────

@dataclass
class _SideSample:
    """One frame of one arm's state, for the trajectory checks."""
    ok: bool = False
    x: float = 0.0
    y: float = 0.0
    elevation: float = 0.0


@dataclass
class _HistSample:
    t: float
    left: _SideSample
    right: _SideSample


@dataclass
class ArmTrackerConfig:
    """Thresholds for arm-state classification.

    Angular/ratio thresholds (elevation, extension) are dimensionless.
    Length thresholds are expressed as multiples of a measured body part —
    arm length or shoulder width — never as a fraction of the frame, so
    nothing here needs re-tuning when the camera or its resolution changes.
    """

    # ── Raise: "arm extended straight up" ─────────────────────────────────
    # Minimum elevation for an arm to count as raised. 0.70 ~ 45 degrees
    # above horizontal; 1.0 would demand a perfectly vertical arm. A truly
    # vertical arm still projects near 1.0 even on a steeply tilted camera,
    # so there is plenty of headroom here.
    raise_elevation_min: float = 0.70
    # Minimum extension (straightness). 0.80 ~ elbow open past ~105 degrees.
    # Rejects a hand held at the face or a chicken-winged forearm, which is
    # why no separate wrist-near-nose exclusion is needed.
    arm_extension_min: float = 0.80
    # Foreshortening guard: an arm pointing at the camera projects to almost
    # nothing, and elevation/extension computed from a 5-pixel vector are
    # noise. Require the projected arm to be at least this multiple of
    # shoulder width (or torso length when shoulders are unusable) before
    # trusting it. Refusing to judge is the correct failure here.
    min_arm_len_frac: float = 0.55
    # Elevation at/above which an arm counts as "on its way up" for the
    # early chime.
    arm_rising_elevation: float = 0.45

    # ── T-pose: both arms out sideways ────────────────────────────────────
    # |elevation| must stay within this band of horizontal. 0.30 ~ 17 deg.
    tpose_elevation_band: float = 0.30
    # Each wrist must sit this far to its own side of the body midline,
    # as a multiple of that arm's length.
    tpose_lateral_min: float = 0.50

    # ── Cross-arms: wrists crossed at chest height ────────────────────────
    # Each wrist must cross the body midline by this multiple of shoulder
    # width, and the two wrists must be closer together than
    # cross_arms_wrist_proximity shoulder-widths.
    #
    # These two pull against each other: crossing further past the midline
    # necessarily pushes the wrists apart, so a tight proximity leaves only
    # a narrow band of poses that satisfy both. At 1.20 the band was
    # 0.80-1.20 shoulder-widths of wrist separation, and folding your arms
    # more tightly fell out of it. 2.00 keeps the crossing requirement doing
    # the real work while accepting any reasonably folded pose.
    cross_arms_min_crossing: float = 0.40
    cross_arms_wrist_proximity: float = 2.00
    # Vertical band counting as "chest", as a multiple of torso length
    # beyond the shoulder line and the hip line respectively.
    cross_arms_chest_pad: float = 0.15

    # ── Leg-raise guard ───────────────────────────────────────────────────
    # If both knees rise this far above shoulder level (frame fraction of
    # height — this one stays frame-relative because it is a coarse
    # whole-body sanity check, not a gesture measurement), suppress
    # everything. Catches legs-in-a-V while lying on the couch.
    leg_raise_margin: float = 0.05

    # ── Detection quality ─────────────────────────────────────────────────
    # At least ONE shoulder must exceed this confidence. Uses max (not
    # average) so a real person with one shoulder hidden by a blanket,
    # cushion or side-on pose is not rejected outright.
    pose_visibility_min: float = 0.55
    # Per-keypoint confidence below which a single landmark is treated as
    # not-visible, gating each arm independently.
    keypoint_visibility_min: float = 0.30

    # ── Hysteresis ────────────────────────────────────────────────────────
    # Seconds a new non-DOWN state must be seen before being accepted.
    state_confirm_s: float = 0.20
    # Seconds of contradictory frames before releasing a stable state.
    state_release_s: float = 0.30

    # ── Trajectory (rise / stillness evidence for SNAP) ───────────────────
    raise_travel_window_s: float = 2.5
    # An arm counts as having risen if it was seen at/below this elevation
    # within the window (a normal lift starts from a low arm) …
    rise_start_elevation_max: float = 0.35
    # … OR if its elevation has climbed by at least this much within the
    # window, wherever it started. The second route is what lets a raise
    # that begins with the arm already up on an armrest or draped over the
    # backrest fire at all — such an arm never reads low, so the first
    # route alone would block it forever.
    rise_elevation_delta: float = 0.35
    # The wrist must stay within this multiple of its own arm length over
    # wrist_still_window_s for the raise to count as held rather than in
    # transit toward a phone, a glass or a blanket.
    wrist_still_window_s: float = 0.30
    wrist_still_max_travel: float = 0.15

    # ── Flourish ──────────────────────────────────────────────────────────
    # How far back to look for the bottom of the sweep. Must comfortably
    # outlast the whole lift PLUS the confirm/sustain delay before the raise
    # is evaluated, or the ascent ages out of the window and the gesture is
    # judged on an empty record. At 0.80 that is exactly what happened.
    flourish_window_s: float = 2.50
    # A raise arriving on the end of a sweep at least this large skips the
    # state_confirm_s wait. That delay exists to reject single-frame
    # phantoms, and a coherent multi-frame climb of this size is already
    # proof the arm is genuinely moving — whereas paying the delay would
    # eat most of the brief window a fast flourish spends at the top, and
    # a really snappy one could never confirm at all.
    sweep_confirm_climb: float = 0.60


class ArmTracker:
    def __init__(self, config: ArmTrackerConfig):
        self.c = config
        self._stable_reading: Optional[ArmReading] = None
        self._pending_state: Optional[ArmState] = None
        self._pending_since: float = 0.0
        # Rolling per-arm history for the trajectory checks.
        self._hist: Deque[_HistSample] = deque()

    # ── Trajectory history ────────────────────────────────────────────────

    def _record(self, now: float, left: _SideSample, right: _SideSample) -> None:
        self._hist.append(_HistSample(t=now, left=left, right=right))
        horizon = max(self.c.raise_travel_window_s,
                      self.c.wrist_still_window_s) + 1.0
        while self._hist and now - self._hist[0].t > horizon:
            self._hist.popleft()

    def _side_hist(self, side: Side, now: float, window: float):
        """Yield (timestamp, sample) for this side inside *window* seconds."""
        for s in self._hist:
            if now - s.t > window:
                continue
            sample = s.left if side is Side.LEFT else s.right
            if sample.ok:
                yield s.t, sample

    def _sweep(self, side: Side, elevation: float,
               now: float) -> Tuple[float, float, bool]:
        """Return (climb, rate) for this arm's current upward sweep.

        Returns (climb, rate, progressive).

        climb is the elevation gained since the arm's lowest point inside
        flourish_window_s; rate is that climb per second since that low
        point. Together they describe a flourish — a fast arc from low to
        high — and they are what distinguishes a deliberate gesture from a
        resting arm, which produces a rate of essentially zero however much
        it happens to resemble a raise geometrically.

        progressive says the arm was seen at intermediate elevations on the
        way, so the climb is a trajectory rather than a jump between two
        extremes. Pose-keypoint noise flickering between a good position and
        a garbage one produces a large climb and a huge rate but is bimodal,
        and must not be read as a very fast gesture.
        """
        samples = [(ts, s.elevation)
                   for ts, s in self._side_hist(side, now, self.c.flourish_window_s)]
        if not samples:
            return 0.0, 0.0, False

        # The ASCENT: the lowest point in the window, then the highest point
        # reached AFTER it. Measuring to "now" instead makes the sweep
        # evaporate the moment the arm is held up — every sample in the
        # window becomes the top, so the minimum equals the current value and
        # the climb collapses to zero. Live feedback showed exactly that:
        # raises with a full 1.9 of travel reported sweep_climb 0.00 and were
        # rejected as "did not sweep", while the chime had already fired
        # mid-ascent. One bug, both symptoms.
        # The LAST time the arm was at its lowest, not the first: an arm
        # that rested at its low point for a while before lifting would
        # otherwise have all that resting time counted as part of the
        # ascent, diluting the rate until a genuine flourish looked slow.
        low_i = 0
        for i in range(len(samples)):
            if samples[i][1] <= samples[low_i][1]:
                low_i = i
        low_t, low_elev = samples[low_i]
        peak_i = low_i
        for i in range(low_i, len(samples)):
            if samples[i][1] > samples[peak_i][1]:
                peak_i = i
        peak_t, peak_elev = samples[peak_i]

        climb = peak_elev - low_elev
        dt = peak_t - low_t
        if climb <= 0.0 or dt <= 1e-6:
            return max(climb, 0.0), 0.0, False

        # Rate over the ascent alone, so holding the arm up afterwards cannot
        # dilute it. The speed of the lift is what identifies a flourish, and
        # that does not change because you kept your arm there.
        band_lo = low_elev + 0.25 * climb
        band_hi = peak_elev - 0.25 * climb
        progressive = any(band_lo < e < band_hi
                          for _ts, e in samples[low_i:peak_i + 1])
        return climb, climb / dt, progressive

    def _rose_recently(self, side: Side, elevation: float,
                       now: float) -> Tuple[bool, float]:
        """Did this arm actually rise? Returns (rose, rise_delta).

        Two independent routes, either of which suffices: the arm was seen
        low at some point in the window, or its elevation has climbed by
        rise_elevation_delta from its lowest point in the window. The second
        route is essential — an arm resting on an armrest or backrest never
        reads low, so requiring the first alone makes a genuine raise from
        that position impossible.

        A buffer that does not yet span the window (person just acquired)
        returns True: only assert "did not rise" once there is enough
        history to know. A pose that persists longer than the window loses
        the benefit of the doubt.
        """
        window = self.c.raise_travel_window_s
        lowest = None
        for _t, sample in self._side_hist(side, now, window):
            if lowest is None or sample.elevation < lowest:
                lowest = sample.elevation
        if lowest is None:
            return True, 0.0
        delta = elevation - lowest
        if lowest <= self.c.rise_start_elevation_max or delta >= self.c.rise_elevation_delta:
            return True, delta
        covered = bool(self._hist) and (now - self._hist[0].t) >= window * 0.9
        return (not covered), delta

    def _wrist_still(self, side: Side, x: float, y: float,
                     now: float, arm_len: float) -> bool:
        """Has this wrist held position, relative to its own arm length?
        A sparse/young buffer is permissive — during a real raise the buffer
        fills at frame rate, so motion is observed whenever it exists.
        """
        if arm_len <= 1e-6:
            return True
        max_travel = self.c.wrist_still_max_travel * arm_len
        for _t, sample in self._side_hist(side, now, self.c.wrist_still_window_s):
            if _dist((x, y), (sample.x, sample.y)) > max_travel:
                return False
        return True

    # ── Public API ────────────────────────────────────────────────────────

    def classify(
        self,
        landmarks: Optional[_Landmarks],
        frame_w: int,
        frame_h: int,
        hand_lm: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> Optional[ArmReading]:
        """Public entry point — time-based hysteresis + optional grlib snap_roll.

        hand_lm: flat numpy array from grlib Pipeline (21*3 floats, 1 hand).
        snap_roll is attached to SINGLE_UP readings only.
        now: monotonic timestamp; defaults to time.monotonic().
        """
        if now is None:
            now = time.monotonic()
        result = self._hyst_classify(landmarks, frame_w, frame_h, now)
        if result is not None and result.state == ArmState.SINGLE_UP and hand_lm is not None:
            result = dataclasses.replace(result, snap_roll=_hand_snap_roll(hand_lm))
        return result

    def _hyst_classify(self, landmarks: Optional[_Landmarks],
                       frame_w: int, frame_h: int, now: float) -> Optional[ArmReading]:
        """Time-based hysteresis: a new state must persist for state_confirm_s
        seconds before being accepted; a stable state requires state_release_s
        seconds of contradictory frames before being released.
        """
        raw = self._classify_raw(landmarks, frame_w, frame_h, now)
        raw_state = raw.state if raw is not None else ArmState.DOWN

        if self._stable_reading is None:
            if raw_state == ArmState.DOWN:
                self._stable_reading = raw
                self._pending_state = None
                return raw

            if self._pending_state != raw_state:
                self._pending_state = raw_state
                self._pending_since = now

            if (now - self._pending_since) >= self._confirm_needed(raw):
                self._stable_reading = raw
                self._pending_state = None
                return raw

            return ArmReading(state=ArmState.DOWN, arm_rising=raw.arm_rising if raw else False)

        stable_state = self._stable_reading.state
        if raw_state == stable_state:
            self._stable_reading = raw if raw is not None else self._stable_reading
            self._pending_state = None
            return self._stable_reading

        if self._pending_state != raw_state:
            self._pending_state = raw_state
            self._pending_since = now

        needed = (self.c.state_release_s if raw_state == ArmState.DOWN
                  else self._confirm_needed(raw))

        if (now - self._pending_since) >= needed:
            self._stable_reading = raw if raw is not None else ArmReading(state=ArmState.DOWN)
            self._pending_state = None
            return self._stable_reading

        return self._stable_reading

    def _confirm_needed(self, raw: Optional[ArmReading]) -> float:
        """How long a new non-DOWN state must persist before it is accepted.

        Normally state_confirm_s, which rejects single-frame phantoms. A
        raise that arrives on the end of a substantial sweep skips the wait:
        the sweep is itself multi-frame evidence of real movement, and a
        fast flourish may spend barely longer than state_confirm_s above the
        raise threshold — paying it would drop the gesture entirely.
        """
        if (raw is not None and raw.state == ArmState.SINGLE_UP
                and raw.sweep_climb >= self.c.sweep_confirm_climb
                and raw.sweep_progressive):
            return 0.0
        return self.c.state_confirm_s

    # ── Classification ────────────────────────────────────────────────────

    def _classify_raw(self, landmarks: Optional[_Landmarks],
                      frame_w: int, frame_h: int,
                      now: Optional[float] = None) -> Optional[ArmReading]:
        """Classify pose landmarks into an ArmReading.

        Returns None if landmarks are missing or visibility is too low.
        now=None (direct/unit-test calls) skips the trajectory checks —
        rose_recently/wrist_still stay at their permissive defaults.
        """
        if landmarks is None:
            return None

        KV = self.c.keypoint_visibility_min

        def px(idx: int) -> _Pt:
            p = landmarks[idx]
            return (p.x * frame_w, p.y * frame_h)

        def vis(idx: int) -> float:
            return landmarks[idx].visibility

        # Furniture/ghost rejection — require at least ONE confident shoulder.
        if max(vis(LEFT_SHOULDER), vis(RIGHT_SHOULDER)) < self.c.pose_visibility_min:
            return None

        ls_ok, rs_ok = vis(LEFT_SHOULDER) >= KV, vis(RIGHT_SHOULDER) >= KV
        ls, rs = px(LEFT_SHOULDER), px(RIGHT_SHOULDER)
        le, re = px(LEFT_ELBOW), px(RIGHT_ELBOW)
        lw, rw = px(LEFT_WRIST), px(RIGHT_WRIST)
        lh, rh = px(LEFT_HIP), px(RIGHT_HIP)

        # Body reference from whichever shoulder(s) are actually visible.
        if ls_ok and rs_ok:
            sh_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            shoulder_w = _dist(ls, rs)
        elif ls_ok:
            sh_mid, shoulder_w = ls, 0.0
        else:
            sh_mid, shoulder_w = rs, 0.0

        # Torso, and the upright flag — reported for diagnostics only now;
        # no gesture threshold depends on posture any more.
        hip_vis = (vis(LEFT_HIP) + vis(RIGHT_HIP)) / 2.0
        torso_len = 0.0
        upright: Optional[bool] = None
        hip_mid: Optional[_Pt] = None
        if hip_vis >= 0.20:
            hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
            torso_len = _dist(sh_mid, hip_mid)
            torso_dy = hip_mid[1] - sh_mid[1]
            torso_dx = hip_mid[0] - sh_mid[0]
            upright = abs(torso_dy) >= abs(torso_dx) and torso_dy > 0

        # Reference length for the foreshortening guard and the cross-arms
        # geometry. Both candidates are now in pixels, so comparing them is
        # meaningful (in normalised coords it was not — see module docstring).
        body_size = max(shoulder_w, torso_len)

        # Leg-raise guard: both knees above shoulder level means legs in the
        # air, not arms. Deliberately still frame-relative — a coarse
        # whole-body sanity check rather than a gesture measurement.
        lk, rk = px(LEFT_KNEE), px(RIGHT_KNEE)
        if (vis(LEFT_KNEE) + vis(RIGHT_KNEE)) / 2.0 >= 0.20:
            if (lk[1] + rk[1]) / 2.0 < sh_mid[1] - self.c.leg_raise_margin * frame_h:
                return ArmReading(state=ArmState.DOWN, upright=bool(upright))

        # ── Per-arm metrics ───────────────────────────────────────────────
        l_ok = ls_ok and vis(LEFT_WRIST) >= KV and vis(LEFT_ELBOW) >= KV
        r_ok = rs_ok and vis(RIGHT_WRIST) >= KV and vis(RIGHT_ELBOW) >= KV

        l_elev, l_ext, l_len = _arm_metrics(ls, le, lw) if l_ok else (0.0, 0.0, 0.0)
        r_elev, r_ext, r_len = _arm_metrics(rs, re, rw) if r_ok else (0.0, 0.0, 0.0)

        if now is not None:
            self._record(
                now,
                _SideSample(ok=l_ok, x=lw[0], y=lw[1], elevation=l_elev),
                _SideSample(ok=r_ok, x=rw[0], y=rw[1], elevation=r_elev),
            )

        def _judgeable(arm_len: float) -> bool:
            """Is this arm's projection long enough to trust its direction?"""
            if body_size <= 1e-6:
                return True   # nothing to compare against; stay permissive
            return arm_len >= self.c.min_arm_len_frac * body_size

        def _raised(elev: float, ext: float, arm_len: float, ok: bool) -> bool:
            return (ok
                    and _judgeable(arm_len)
                    and elev >= self.c.raise_elevation_min
                    and ext >= self.c.arm_extension_min)

        l_raised = _raised(l_elev, l_ext, l_len, l_ok)
        r_raised = _raised(r_elev, r_ext, r_len, r_ok)

        both_sh_ok = ls_ok and rs_ok

        # ── Two-handed gestures (need BOTH shoulders confidently visible) ──
        if both_sh_ok and l_ok and r_ok:
            # Sideways position is measured along the person's OWN left-right
            # axis (right shoulder -> left shoulder), not raw image x.
            #
            # Using image x silently assumes an orientation. A person facing
            # the camera is mirrored: their right shoulder appears on the
            # image's LEFT, so their right wrist sits left of the midline
            # while simply hanging at their side — which the old test read as
            # "crossed past the midline". Both arms resting down therefore
            # classified as CROSS_ARMS, stealing gestures from an obvious
            # raise. Projecting onto the shoulder axis makes "toward my other
            # shoulder" mean the same thing whichever way the person faces.
            axis = ((ls[0] - rs[0]) / shoulder_w,
                    (ls[1] - rs[1]) / shoulder_w) if shoulder_w > 1e-6 else (1.0, 0.0)

            def _toward_left(p: _Pt) -> float:
                """Signed distance from the body midline along the shoulder
                axis: positive = toward this person's left shoulder."""
                return ((p[0] - sh_mid[0]) * axis[0]
                        + (p[1] - sh_mid[1]) * axis[1])

            # CROSS_ARMS: each wrist past the body midline onto the OTHER
            # side, at chest height, wrists close together. Distances are
            # multiples of shoulder width so this holds at any camera distance.
            if shoulder_w > 1e-6:
                cross = self.c.cross_arms_min_crossing * shoulder_w
                rw_crossed = _toward_left(rw) >= cross      # right hand gone left
                lw_crossed = _toward_left(lw) <= -cross     # left hand gone right
                pad = self.c.cross_arms_chest_pad * (torso_len or shoulder_w)
                chest_top = sh_mid[1] - pad
                chest_bottom = (hip_mid[1] if hip_mid is not None
                                else sh_mid[1] + 1.2 * shoulder_w) + pad
                at_chest = (chest_top < rw[1] < chest_bottom
                            and chest_top < lw[1] < chest_bottom)
                close = _dist(lw, rw) < self.c.cross_arms_wrist_proximity * shoulder_w
                if rw_crossed and lw_crossed and at_chest and close:
                    return ArmReading(state=ArmState.CROSS_ARMS, upright=bool(upright))

            # T_POSE: both arms straight, level, and reaching to their own sides.
            band = self.c.tpose_elevation_band
            lat = self.c.tpose_lateral_min
            if (abs(l_elev) <= band and abs(r_elev) <= band
                    and l_ext >= self.c.arm_extension_min
                    and r_ext >= self.c.arm_extension_min
                    and _judgeable(l_len) and _judgeable(r_len)
                    # …each reaching out on its OWN side, measured along the
                    # body's left-right axis rather than image x, so this
                    # works whichever way the person faces (see _toward_left).
                    and _toward_left(lw) >= lat * l_len
                    and _toward_left(rw) <= -lat * r_len):
                return ArmReading(state=ArmState.T_POSE, upright=bool(upright))

            # BOTH_UP: both arms raised.
            if l_raised and r_raised:
                return ArmReading(state=ArmState.BOTH_UP, upright=bool(upright))

        # ── SINGLE_UP — per side, works with only one visible shoulder ─────
        if r_raised or l_raised:
            if r_raised:
                side, wrist, elbow = Side.RIGHT, rw, re
                elev, ext, arm_len = r_elev, r_ext, r_len
            else:
                side, wrist, elbow = Side.LEFT, lw, le
                elev, ext, arm_len = l_elev, l_ext, l_len

            if now is not None:
                rose, rise_delta = self._rose_recently(side, elev, now)
                still = self._wrist_still(side, wrist[0], wrist[1], now, arm_len)
                climb, rate, progressive = self._sweep(side, elev, now)
            else:
                rose, still, rise_delta = True, True, 0.0
                climb, rate, progressive = 0.0, 0.0, False

            return ArmReading(
                state=ArmState.SINGLE_UP,
                raised_side=side,
                wrist_x=wrist[0],
                wrist_y=wrist[1],
                elevation=elev,
                extension=ext,
                arm_len_px=arm_len,
                forearm_dy=(elbow[1] - wrist[1]) / frame_h,
                upright=bool(upright),
                rose_recently=rose,
                wrist_still=still,
                rise_delta=rise_delta,
                sweep_climb=climb,
                sweep_rate=rate,
                sweep_progressive=progressive,
            )

        # Nothing raised — report the higher arm's geometry for the overlay,
        # including its sweep, so the chime can fire while the arm is still
        # on its way up rather than after it arrives.
        if l_ok and (not r_ok or l_elev >= r_elev):
            best_side, best_elev, best_ext, best_len = Side.LEFT, l_elev, l_ext, l_len
        else:
            best_side, best_elev, best_ext, best_len = Side.RIGHT, r_elev, r_ext, r_len
        if now is not None and (l_ok or r_ok):
            climb, rate, progressive = self._sweep(best_side, best_elev, now)
        else:
            climb, rate, progressive = 0.0, 0.0, False
        rising = max(l_elev if l_ok else -1.0,
                     r_elev if r_ok else -1.0) >= self.c.arm_rising_elevation
        return ArmReading(state=ArmState.DOWN, upright=bool(upright),
                          elevation=best_elev, extension=best_ext,
                          arm_len_px=best_len, arm_rising=rising,
                          sweep_climb=climb, sweep_rate=rate,
                          sweep_progressive=progressive)
