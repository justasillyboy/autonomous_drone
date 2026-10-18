"""
mission.py — Mission 2 + Mission 3 state machine for the UAV fire-fighting competition.

Competition context (from rule document):
  Mission 2: Drop 5 tennis balls onto fire circles F1-F5 autonomously.
             F1, F2 → Red circles   (diameter 1.0 m)
             F3, F4 → Yellow circles (diameter 0.6 m)
             F5     → Blue circle    (diameter 0.4 m)
  Mission 3: After all 5 balls are dropped (or when directed), auto-land on
             the H_landing helipad in the Safe Zone.

How this module works:
  1. YOLO detects 'circle' or 'H_landing' bounding masks every frame.
  2. For each detected circle, color_of_mask() applies HSV filtering to
     identify Red / Yellow / Blue.
  3. MissionStateMachine.update() checks whether the detected color matches
     the CURRENT target in the fire sequence.
  4. Once diameter_ratio >= DROP_THRESHOLD, it signals a ball drop.
  5. After 5 drops, the state transitions to FIND_HELIPAD.
  6. In FIND_HELIPAD / LANDING, YOLO's H_landing class is used for guidance.

Integration with test_video.py:
    from mission import MissionStateMachine, color_of_mask, draw_mission_hud

    mission = MissionStateMachine()

    while cap.isOpened():
        ret, frame = cap.read()
        results = model(frame, verbose=False)[0]
        frame = mission.update(frame, results)
        out.write(frame)
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Mission constants
# ---------------------------------------------------------------------------

# Drop threshold: circle must fill this fraction of frame width before ball drop
DROP_THRESHOLD = 0.70

# Fill-ratio filter — ratio of actual mask pixels to a solid circle of the same radius.
# Solid fire target circle → ~0.70-0.85
# Hollow H_landing ring   → ~0.20-0.35  (rejected)
# Thin court arc line     → ~0.04-0.12  (rejected)
FILL_RATIO_MIN_CIRCLE = 0.40

# Maximum diameter_ratio for a fire target.
# At 3-5 m altitude, largest real circle (1.0 m) = ~21-34% of frame width.
# Court boundary arcs span 50-80% → rejected by this cap.
MAX_FIRE_DIAMETER_RATIO = 0.55

# Cross-class NMS: how much of a circle bbox must overlap H_landing before suppressing it.
H_LANDING_OVERLAP_THRESH = 0.45

# ---------------------------------------------------------------------------
# Competition fire target specs  (verified against rulebook 27/02/2026)
# Each entry: (color_name, label, points_qualifier, points_finals, real_diameter_m)
# NOTE: In finals, F2 is a MOVING target (10 cm/s) worth 50 pts vs 20 pts static.
# ---------------------------------------------------------------------------

FIRE_TARGETS_QUALIFIER = [
    ("red",    "F1", 20, 1.0),   # static  — 1000 mm red
    ("red",    "F2", 20, 1.0),   # static  — 1000 mm red
    ("yellow", "F3", 30, 0.6),   # static  —  600 mm yellow
    ("yellow", "F4", 30, 0.6),   # static  —  600 mm yellow
    ("blue",   "F5", 50, 0.4),   # static  —  400 mm blue
]

FIRE_TARGETS_FINALS = [
    ("red",    "F1", 10, 1.0),   # static  — 1000 mm red
    ("red",    "F2", 50, 1.0),   # MOVING  — 1000 mm red, 10 cm/s on orange line
    ("yellow", "F3", 30, 0.6),   # static  —  600 mm yellow
    ("yellow", "F4", 30, 0.6),   # static  —  600 mm yellow
    ("blue",   "F5", 50, 0.4),   # static  —  400 mm blue
]

# Team chooses attack order — default: highest value first to maximise score/time
FIRE_SEQUENCE = FIRE_TARGETS_QUALIFIER   # switch to FIRE_TARGETS_FINALS for finals

# Real physical diameters (m) keyed by colour — used for altitude estimation
FIRE_REAL_DIAMETERS: dict[str, float] = {
    "red":    1.0,
    "yellow": 0.6,
    "blue":   0.4,
}

# ---------------------------------------------------------------------------
# Altitude-based drop control  (replaces diameter_ratio for fire targets)
#
# Competition rules (rulebook §III-4):
#   UAV lowest point must be ≥ 3.0 m from floor during ball drop.
#   Auto-flight altitude must stay within 3.0 – 5.0 m.
#
# The old diameter_ratio ≥ 0.70 threshold was DANGEROUSLY WRONG:
#   At f_px ≈ 500 px, a 1.0 m red circle occupies 70% of a 640px frame
#   only at Z ≈ 1.1 m — far below the 3.0 m minimum, causing -5 pt penalties.
#
# Correct approach: estimate Z = f_px × D_real / d_px and drop when Z is safe.
# ---------------------------------------------------------------------------

DROP_ALTITUDE_MIN    = 3.05   # metres  — must be > rulebook minimum of 3.0m
DROP_ALTITUDE_TARGET = 3.25   # metres  — optimal drop height (accuracy + safety)
DROP_ALTITUDE_MAX    = 4.50   # metres  — too high, ball drift increases
ALIGN_PIXEL_THRESH   = 25     # pixels  — centre offset before DROP is allowed

# How close (in pixels) a new detection must be to an already-scored circle
# centroid before it is suppressed. At 3.5 m altitude, a 1.0 m red circle
# spans ~145 px radius. Using 80 px means: if the drone hasn't moved more
# than ~half a circle-width away, we treat it as the same circle.
# This prevents F1 re-lock when advancing to F2 (both red, same position).
SCORED_RADIUS_PX     = 80     # pixels

# ---------------------------------------------------------------------------
# Competition rule constants  (rulebook 27/02/2026 — DO NOT EDIT)
# ---------------------------------------------------------------------------

# Timing
MATCH_TIME_S       = 300   # 5 minutes total for Round 1 + Round 2 combined
PREP_TIME_S        =  60   # setup time at the field before clock starts
ALT_PENALTY_TIME_S =   5   # seconds below 3.0 m before −5 pt altitude penalty

# Penalties (§VII)
PENALTY_TZ_COLLISION = -5   # pts — touching any part of TZ gate frame
PENALTY_BOUNDARY     = -5   # pts — outside field boundary > 5 seconds
PENALTY_ALTITUDE     = -5   # pts — below 3.0 m during auto flight > 5 seconds

# Bonus scores (added to self.score on successful task)
SCORE_TAKEOFF_QUAL   = 20   # auto takeoff ≥ 3.0 m — Qualifier / Semi-final
SCORE_TAKEOFF_FINALS = 10   # auto takeoff ≥ 3.0 m — Finals
SCORE_LANDING_QUAL   = 30   # auto land in helipad — Qualifier / Semi-final
SCORE_LANDING_FINALS = 20   # auto land in helipad — Finals

# Field geometry (metres) — for reference/geofencing (§IV)
FIELD_LENGTH    = 24.0
FIELD_WIDTH     = 12.0
SAFE_ZONE_DEPTH =  3.5   # along field length axis
TZ_INNER_D      =  1.6   # inner diameter of TZ ring obstacle
TZ_CENTER_H     =  1.7   # height of TZ ring centre above floor
TZ_TOTAL_H      =  2.6   # total height of TZ stand
FIRE_CIRCLE_H   =  0.1   # fire circles are mounted 0.1 m above floor

# Default focal length when calibration file is not provided.
# IMX219 at 640px input (after YOLO letterbox): approximately 500 px.
# ALWAYS override with the actual value from camera_calibration.py.
DEFAULT_F_PX = 500.0

# HSV colour ranges for fire circle identification
# Tuned for competition circle colors: #FC2A1B, #FAFF37, #3C00FB
COLOR_HSV = {
    "red": [
        ((0,   120,  80), (10,  255, 255)),   # lower red hue
        ((165, 120,  80), (180, 255, 255)),   # upper red hue (wraps around 180)
    ],
    "yellow": [
        ((18, 100, 100), (38, 255, 255)),
    ],
    "blue": [
        ((100, 130,  50), (145, 255, 255)),
    ],
}

# Minimum fraction of contour pixels that must match the target color
COLOR_MATCH_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# FlightCommand — structured control output
#
# Every frame, MissionStateMachine.update() populates mission.last_command.
# Wire this to your MAVLink / ROS2 publisher — nothing in this file touches
# actual hardware. The FC (ArduPilot/PX4) executes the commands.
#
# Body-frame convention (NED):
#   vx > 0 → forward    vy > 0 → right    vz > 0 → down
#   yaw_rate > 0 → clockwise (deg/s)
# ---------------------------------------------------------------------------

@dataclass
class FlightCommand:
    vx:        float = 0.0    # forward  velocity  m/s
    vy:        float = 0.0    # lateral  velocity  m/s
    vz:        float = 0.0    # vertical velocity  m/s  (positive = descend)
    yaw_rate:  float = 0.0    # rotation deg/s (+ = clockwise)
    yaw_lock:  bool  = True   # True = hold current heading (send to FC)
    drop_ball: bool  = False  # True = trigger servo / relay this frame
    phase:     str   = ""     # current Phase name for logging
    action:    str   = ""     # human-readable action for HUD/logging

    def __str__(self) -> str:
        return (f"[CMD] {self.phase}/{self.action} | "
                f"v=({self.vx:+.2f},{self.vy:+.2f},{self.vz:+.2f}) m/s | "
                f"yaw_rate={self.yaw_rate:+.1f} deg/s | "
                f"yaw_lock={self.yaw_lock} | drop={self.drop_ball}")


# ---------------------------------------------------------------------------
# Mission phases
# ---------------------------------------------------------------------------

class Phase(Enum):
    TAKEOFF      = auto()   # ascending to safe flight altitude (≥ 3.0 m)
    SEARCHING    = auto()   # scanning for the next target colour
    TARGETING    = auto()   # colour confirmed, approaching and descending
    DROPPING     = auto()   # inside drop window — release ball
    RECOVERY     = auto()   # target lost — structured yaw scan to relocate
    FIND_HELIPAD = auto()   # all balls dropped, searching for H_landing
    LANDING      = auto()   # H_landing found, executing approach + land
    COMPLETE     = auto()   # mission finished


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

def color_of_mask(mask_binary: np.ndarray, frame_bgr: np.ndarray) -> Optional[str]:
    """
    Identify the dominant color inside a segmentation mask.

    Uses HSV masking instead of raw BGR so it is robust to lighting changes.

    Args:
        mask_binary : uint8 binary mask (H, W) — 255 inside object, 0 outside.
        frame_bgr   : Full BGR frame the mask belongs to.

    Returns:
        'red' | 'yellow' | 'blue' | None (if no colour dominates).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    roi = cv2.bitwise_and(hsv, hsv, mask=mask_binary)

    total_pixels = int(mask_binary.sum() / 255) + 1
    best_color, best_frac = None, 0.0

    for color_name, ranges in COLOR_HSV.items():
        color_mask = np.zeros_like(mask_binary)
        for (lo, hi) in ranges:
            part = cv2.inRange(roi, np.array(lo), np.array(hi))
            color_mask = cv2.bitwise_or(color_mask, part)

        # Only count pixels that are inside the object mask
        color_mask = cv2.bitwise_and(color_mask, mask_binary)
        frac = color_mask.sum() / 255 / total_pixels

        if frac > best_frac:
            best_frac = frac
            best_color = color_name

    if best_frac >= COLOR_MATCH_THRESHOLD:
        return best_color
    return None


def compute_diameter_ratio(radius: float, frame_w: int) -> float:
    """Diameter of circle as fraction of frame width (kept for H_landing approach)."""
    return (2.0 * radius) / float(frame_w)


# ---------------------------------------------------------------------------
# Flight Controller (FC) inner-loop PID — ArduPilot ATC_RAT values
#
# These are READ-ONLY reference constants. They describe what the hardware
# team has programmed into ArduPilot on the flight controller. They run at
# 400 Hz on gyro feedback and are COMPLETELY SEPARATE from the Python PID
# below, which runs at camera FPS on pixel error.
#
# Source: Mission Planner → Full Parameter List (ATC_RAT_PIT / ATC_RAT_RLL)
# Verified from hardware team screenshots 2026-04.
#
# Control loop architecture:
#
#   [camera] pixel_offset
#       ↓  our Python PIDController  (~20-30 Hz, units: px → m/s)
#   [MAVLink] velocity setpoint  vx / vy  (m/s)
#       ↓  ArduPilot velocity controller  (PSC_VELXY, ~50 Hz)
#   [ArduPilot] desired pitch/roll rate  (rad/s)
#       ↓  ATC_RAT PID  (400 Hz, units: rad/s → motor %)
#   [ESC/motors] PWM
#
# DO NOT copy FC_PID_P or FC_PID_D into PIDController below.
# The unit mismatch (rad/s vs pixels) makes the gains meaningless if reused.
# ---------------------------------------------------------------------------

FC_PID_PITCH_P  = 0.135   # ATC_RAT_PIT_P  (s/rad)
FC_PID_PITCH_I  = 0.135   # ATC_RAT_PIT_I  (1/rad)
FC_PID_PITCH_D  = 0.0036  # ATC_RAT_PIT_D  (s²/rad)
FC_PID_ROLL_P   = 0.135   # ATC_RAT_RLL_P  (s/rad)
FC_PID_ROLL_I   = 0.135   # ATC_RAT_RLL_I  (1/rad)
FC_PID_ROLL_D   = 0.0036  # ATC_RAT_RLL_D  (s²/rad)

# PX_TO_MS: scale factor converting our outer-loop PID pixel output → m/s setpoint.
#
# Derivation: at 3.5 m altitude, 1 m of lateral movement ≈ 285 px in a 640px
# frame (f_px≈500). We want a 100 px error to command ~0.5 m/s correction.
#   0.5 m/s / 100 px = 0.005 m/s per pixel
#
# This value was chosen so the FC's velocity controller (PSC_VELXY_P ≈ 2.0)
# translates our setpoint into a pitch/roll command that the ATC_RAT inner loop
# (P=0.135) can track cleanly without oscillation.
# Increase PX_TO_MS if the drone responds sluggishly; decrease if it oscillates.
PX_TO_MS_FIRE   = 0.005   # fire targeting (faster response needed)
PX_TO_MS_HELIPAD = 0.004  # helipad approach (slower, more precise)

class AltitudeEstimator:
    """
    Estimate drone altitude above a target using the pinhole camera equation:

        Z = f_px × D_real / d_px

    Where:
        f_px   : focal length in pixels (from camera calibration matrix[0][0])
        D_real : real physical diameter of the target in metres
        d_px   : pixel diameter of the target on the image

    Usage:
        est = AltitudeEstimator(f_px=500.0)
        Z = est.estimate(d_px=156, D_real=1.0)   # → 3.2 m

    Load from calibration file:
        est = AltitudeEstimator.from_calibration("calibration.npz")
    """

    def __init__(self, f_px: float = DEFAULT_F_PX):
        self.f_px = f_px

    def estimate(self, d_px: float, D_real: float) -> float:
        """Return estimated altitude in metres. Returns inf if d_px < 1."""
        if d_px < 1.0:
            return float("inf")
        return self.f_px * D_real / d_px

    @classmethod
    def from_calibration(cls, calib_path: str) -> "AltitudeEstimator":
        """Load f_px from calibration.npz produced by camera_calibration.py."""
        data  = np.load(calib_path)
        cam   = data["camera_matrix"]
        f_px  = float((cam[0, 0] + cam[1, 1]) / 2.0)
        print(f"[AltEst] Loaded f_px={f_px:.1f} from {calib_path}")
        return cls(f_px)


# ---------------------------------------------------------------------------
# PID controller  (Fix: needed for moving F2 in Finals round)
# ---------------------------------------------------------------------------

class PIDController:
    """
    Outer-loop position/velocity PID controller.

    CONTROL ARCHITECTURE:
      This is the OUTER loop only. It converts a pixel-offset error into a
      velocity setpoint (m/s), which is sent to ArduPilot via MAVLink
      SET_POSITION_TARGET (velocity mode). ArduPilot's own inner loops
      (ATC_RAT P=0.135, I=0.135, D=0.0036) then execute the velocity command.

      Our PID (outer):  px error  →  velocity setpoint  (m/s)   @20-30 Hz
      ArduPilot (inner): vel error →  motor PWM                  @400 Hz

    TUNING GUIDE (with FC ATC_RAT P=0.135, I=0.135, D=0.0036):
      The FC inner loop is fairly tight (P=0.135 is ArduPilot default for a
      well-tuned 5" quad). This means it tracks velocity commands faithfully
      with little lag. We can therefore afford moderate outer-loop gains.

      kp=0.5  — proportional: corrects ~50% of pixel error per second
      ki=0.01 — integral: eliminates steady hover offset over ~5-10 s
      kd=0.15 — derivative: damps overshoot when approaching the target

      For Finals F2 (moving target, 10 cm/s):
      kp=0.6, ki=0.02, kd=0.20  — slightly more aggressive to reduce lag

      If you see oscillation: lower kp first, then kd.
      If you see steady-state offset: raise ki slightly.
      DO NOT raise kp above 1.0 — the FC inner loop will start to saturate.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.15,
                 output_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = output_limit
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error: float, timestamp: float | None = None) -> float:
        """
        Feed one error sample, get one velocity command.

        Args:
            error     : Signed pixel offset (positive = target is right/below).
            timestamp : Current time in seconds. Uses 1/30 s if None.

        Returns:
            Velocity command in [-output_limit, +output_limit].
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        dt  = (now - self._prev_time) if self._prev_time is not None else (1.0 / 30.0)
        dt  = max(dt, 1e-4)

        self._integral  += error * dt
        # Anti-windup: clamp integral so it cannot alone saturate the output.
        # Without this, a long search phase (drone hovering off-target) winds up
        # the integrator, causing a large overshoot spike when the target appears.
        integral_limit   = self.limit / max(self.ki, 1e-9)
        self._integral   = float(np.clip(self._integral, -integral_limit, integral_limit))

        derivative       = (error - self._prev_error) / dt
        self._prev_error = error
        self._prev_time  = now

        output = (self.kp * error
                  + self.ki * self._integral
                  + self.kd * derivative)
        return float(np.clip(output, -self.limit, self.limit))


def fill_ratio(mask_u8: np.ndarray, radius: float) -> float:
    """
    Ratio of actual mask pixels to a solid circle of the same enclosing radius.

    Used to distinguish solid fire targets from thin arcs/rings:
      - Solid filled circle (real target) : ~0.70-0.85
      - Hollow ring (H_landing ring)      : ~0.20-0.35
      - Thin court arc                    : ~0.04-0.12

    Args:
        mask_u8 : uint8 binary mask (255 inside, 0 outside).
        radius  : enclosing circle radius in pixels (from cv2.minEnclosingCircle).

    Returns:
        float in [0, 1+].
    """
    filled     = int(mask_u8.sum() / 255)
    full_area  = max(1.0, np.pi * radius * radius)
    return filled / full_area


def collect_h_landing_bboxes(result, names: dict) -> list:
    """
    Return a list of (x1, y1, x2, y2) for every H_landing detection in a YOLO result.
    Used for cross-class NMS: any circle/semicircle inside these boxes is suppressed.
    """
    bboxes = []
    if result.boxes is None:
        return bboxes
    for box in result.boxes:
        if names.get(int(box.cls[0].item()), "") == "H_landing":
            x1, y1, x2, y2 = [int(box.xyxy[0][k].item()) for k in range(4)]
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def overlaps_h_landing(cx: int, cy: int, radius: int,
                        h_bboxes: list,
                        threshold: float = H_LANDING_OVERLAP_THRESH) -> bool:
    """
    True if the circle's bounding box overlaps any H_landing bbox by more than
    `threshold` fraction of the circle bbox area.

    This suppresses detections of the H_landing ring/glyph/brackets as
    separate circle/semicircle instances during both fire and helipad phases.
    """
    cbbox_area = max(1, (2 * radius) ** 2)
    for (hx1, hy1, hx2, hy2) in h_bboxes:
        ox1 = max(cx - radius, hx1)
        oy1 = max(cy - radius, hy1)
        ox2 = min(cx + radius, hx2)
        oy2 = min(cy + radius, hy2)
        if ox2 > ox1 and oy2 > oy1:
            overlap_area = (ox2 - ox1) * (oy2 - oy1)
            if overlap_area / cbbox_area >= threshold:
                return True
    return False


# ---------------------------------------------------------------------------
# Optimized mask geometry extraction  (Fix: no full-frame resize)
# ---------------------------------------------------------------------------

def _extract_geometry(mask_obj, frame_w: int, frame_h: int):
    """
    Extract circle geometry from a YOLO segmentation mask without resizing
    the mask to full frame dimensions.

    Previous approach: resize 320×320 mask → full frame (e.g. 1920×1080),
    then findContours on the large image. Expensive on CPU (Pi 5 / Jetson).

    This approach: findContours on the small mask (model output size),
    then scale the resulting contour coordinates to frame space.
    Saves ~10-40ms per detection at 1080p.

    Returns:
        (cx, cy, radius, fill_ratio, cnt_frame) in frame-pixel coordinates,
        or None if no contour found.
    """
    mask_np  = mask_obj.cpu().numpy()
    h_m, w_m = mask_np.shape
    mask_small = (mask_np > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt_small = max(contours, key=cv2.contourArea)

    # Scale contour points from mask space → frame space
    sx = frame_w / w_m
    sy = frame_h / h_m
    cnt_frame = (cnt_small.reshape(-1, 2).astype(np.float32)
                 * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
    cnt_frame = cnt_frame.reshape(-1, 1, 2)

    (cx, cy), radius = cv2.minEnclosingCircle(cnt_frame)

    # Fill ratio computed on small mask (avoid any resize)
    filled        = int(mask_small.sum())
    radius_small  = radius / max(sx, sy)
    full_area     = max(1.0, np.pi * radius_small ** 2)
    fr            = min(filled / full_area, 1.0)

    return int(cx), int(cy), int(radius), fr, cnt_frame


# ---------------------------------------------------------------------------
# Mission state machine
# ---------------------------------------------------------------------------

class MissionStateMachine:
    """
    Tracks fire-fighting progress and transitions to H_landing search.

    Call update() once per frame with the YOLO result.
    It draws the HUD and returns the annotated frame.
    """

    def __init__(self, smoothing_alpha: float = 0.4,
                 f_px: float = DEFAULT_F_PX,
                 is_finals: bool = False,
                 takeoff_alt: float = 3.3):
        """
        Args:
            smoothing_alpha : EMA alpha for position smoothing.
            f_px            : Focal length in pixels (from camera_calibration.py).
            is_finals       : True → Finals scoring + stronger PID.
            takeoff_alt     : Target altitude in metres before entering SEARCHING.
                              Rule: ≥ 3.0 m. Default 3.3 m gives 0.3 m safety margin.
        """
        self.phase              = Phase.TAKEOFF   # always start with controlled ascent
        self.balls_dropped      = 0
        self.current_target_idx = 0
        self.score              = 0
        self._drop_cooldown     = 0

        # Bonus task scores (applied once per run, §VI)
        self._takeoff_scored    = False   # +SCORE_TAKEOFF_QUAL/FINALS when altitude reached
        self._landing_scored    = False   # +SCORE_LANDING_QUAL/FINALS on signal_landed()
        self.penalties          = 0       # running penalty total (negative)
        self._low_alt_frames    = 0       # consecutive frames below 3.0 m in auto flight
        # fps_hint: used to convert time-based thresholds (e.g. ALT_PENALTY_TIME_S)
        # into frame counts.  Set this to the actual pipeline FPS in test_video.py:
        #   mission.fps_hint = cap.get(cv2.CAP_PROP_FPS) or 30
        # Hardcoding 30 means altitude penalty fires every 30 frames at ANY fps,
        # which is wrong at 20 fps (fires every 1.5 s instead of every 5 s).
        self._fps_hint          = 30

        # Scored-position exclusion zone — prevents re-locking on same circle.
        # After each drop, the frame centroid (cx, cy) is stored here.
        # Any new detection within SCORED_RADIUS_PX of a stored centroid is
        # suppressed — it is the circle we already scored (fixes F1/F2 red
        # and F3/F4 yellow same-color re-lock bug).
        self._scored_positions: list = []   # list of (cx, cy) in frame pixels

        # Takeoff
        self._takeoff_alt = takeoff_alt

        # Altitude estimation
        self.alt_est = AltitudeEstimator(f_px=f_px)

        # Fire sequence — instance variable to avoid global state corruption
        self.is_finals = is_finals
        self._fire_sequence = FIRE_TARGETS_FINALS if is_finals else FIRE_TARGETS_QUALIFIER

        # PID (stronger gains in Finals for moving F2)
        kp, ki, kd = (0.6, 0.02, 0.20) if is_finals else (0.5, 0.01, 0.15)
        self._pid_x = PIDController(kp=kp, ki=ki, kd=kd)
        self._pid_y = PIDController(kp=kp, ki=ki, kd=kd)

        # Temporal smoothing — lower alpha in Finals to reduce phase lag on F2.
        # alpha=0.4 is good for static targets; alpha=0.15 reacts faster to
        # F2's 10 cm/s movement without introducing too much noise.
        if smoothing_alpha == 0.4 and is_finals:
            smoothing_alpha = 0.15
            print("[Mission] Finals mode: EMA alpha lowered to 0.15 (less F2 lag)")
        self._smooth_alpha  = smoothing_alpha
        self._smooth_cx:    float | None = None
        self._smooth_cy:    float | None = None
        self._smooth_ratio: float | None = None

        # Position history for feedforward velocity estimation (F2 tracking).
        # deque(maxlen) auto-discards oldest entries in O(1) vs list.pop(0) O(n).
        from collections import deque as _deque
        self._pos_history: _deque = _deque(maxlen=8)  # 8 = _ff_history_len
        self._ff_history_len = 8   # frames of history for velocity estimate

        # No-detection fallback
        self._no_detect_frames = 0
        self._max_no_detect    = 10

        # Recovery / scan state
        self._recovery_frames   = 0
        self._recovery_max      = 90          # ~3 s at 30 fps before giving up
        self._scan_step         = 0           # which step in the yaw-scan sequence
        self._scan_yaw_steps    = [30, -60, 30, 0]   # deg/s per step (clockwise sweep)
        self._scan_step_frames  = 20          # frames per scan step

        # Structured command output — read this in your MAVLink integration
        self.last_command: FlightCommand = FlightCommand(phase="TAKEOFF",
                                                          action="ASCEND")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_target(self):
        """Return (color_name, label, points, real_diameter_m) for current fire target."""
        if self.current_target_idx < len(self._fire_sequence):
            return self._fire_sequence[self.current_target_idx]
        return None

    @property
    def mission_complete(self):
        return self.phase == Phase.COMPLETE

    # ── Temporal smoothing helper ─────────────────────────────────────────────

    def _smooth(self, cx: int, cy: int, ratio: float) -> tuple:
        """EMA on centre position + ratio. Also records history for feedforward."""
        a = self._smooth_alpha
        if self._smooth_cx is None:
            self._smooth_cx    = float(cx)
            self._smooth_cy    = float(cy)
            self._smooth_ratio = ratio
        else:
            self._smooth_cx    = a * self._smooth_cx    + (1 - a) * cx
            self._smooth_cy    = a * self._smooth_cy    + (1 - a) * cy
            self._smooth_ratio = a * self._smooth_ratio + (1 - a) * ratio

        # Record smoothed position for feedforward estimation
        self._pos_history.append(
            (self._smooth_cx, self._smooth_cy, time.perf_counter())
        )

        return int(self._smooth_cx), int(self._smooth_cy), self._smooth_ratio

    def _feedforward_velocity(self) -> tuple[float, float]:
        """
        Estimate target pixel velocity from smoothed position history using
        least-squares linear regression and return a (ff_vx, ff_vy) feedforward
        offset in pixel/s.

        Used to pre-compensate for moving F2 — the PID reacts to the current
        error, the feedforward anticipates where F2 is heading.

        Linear regression (polyfit) is used instead of simple endpoint difference
        because it is robust against noisy first/last samples: a single detection
        jitter at the beginning or end of the window cannot corrupt the estimate.

        Returns (0.0, 0.0) when history is insufficient or target is static.
        """
        if len(self._pos_history) < 4:
            return 0.0, 0.0

        xs  = np.array([p[0] for p in self._pos_history])
        ys  = np.array([p[1] for p in self._pos_history])
        ts  = np.array([p[2] for p in self._pos_history])

        # Normalise timestamps to avoid numerical issues in polyfit
        t0  = ts[0]
        ts  = ts - t0
        dt_total = ts[-1]
        if dt_total < 1e-4:
            return 0.0, 0.0

        # Least-squares slope = velocity in px/s
        vx_px = float(np.polyfit(ts, xs, 1)[0])
        vy_px = float(np.polyfit(ts, ys, 1)[0])

        # Cap feedforward to avoid over-shooting on noisy single-frame spikes
        MAX_FF_PX_S = 60.0   # ~2 px/frame at 30fps — tunable
        vx_px = float(np.clip(vx_px, -MAX_FF_PX_S, MAX_FF_PX_S))
        vy_px = float(np.clip(vy_px, -MAX_FF_PX_S, MAX_FF_PX_S))
        return vx_px, vy_px

    # ── Main per-frame update ─────────────────────────────────────────────────

    def update(self, frame: np.ndarray, yolo_result,
               current_altitude: float | None = None) -> np.ndarray:
        """
        Process one YOLO result and update mission state.

        Args:
            frame            : BGR frame (annotated in-place).
            yolo_result      : Single Ultralytics Results object.
            current_altitude : Altitude in metres from barometer/rangefinder if
                               available. When None, altitude is estimated from
                               camera geometry. Provide real sensor data when
                               wired to your flight controller.

        After calling, read mission.last_command for the structured FlightCommand.
        """
        h, w = frame.shape[:2]

        if self._drop_cooldown > 0:
            self._drop_cooldown -= 1

        # ── Altitude penalty tracking (§VII — <3.0m >5s in auto = −5pts) ────
        # Only applies during autonomous flight phases (not TAKEOFF or COMPLETE).
        # Counts consecutive frames below 3.0m and applies penalty each 5-second window.
        if (self.phase not in (Phase.TAKEOFF, Phase.COMPLETE)
                and current_altitude is not None
                and current_altitude < 3.0):
            self._low_alt_frames += 1
            penalty_frame_thresh = int(ALT_PENALTY_TIME_S * self._fps_hint)
            if self._low_alt_frames % penalty_frame_thresh == 0:
                self.penalties += PENALTY_ALTITUDE
                print(f"[PENALTY] Altitude {current_altitude:.2f}m < 3.0m for "
                      f"{ALT_PENALTY_TIME_S}s → {PENALTY_ALTITUDE} pts  "
                      f"(total penalties: {self.penalties})")
        else:
            self._low_alt_frames = 0   # reset on any frame above 3.0m

        # ── Phase routing ─────────────────────────────────────────────────────
        if self.phase == Phase.TAKEOFF:
            frame = self._takeoff_phase(frame, current_altitude, w, h)

        elif self.phase in (Phase.SEARCHING, Phase.TARGETING, Phase.DROPPING):
            frame = self._fire_phase(frame, yolo_result, w, h, current_altitude)

        elif self.phase == Phase.RECOVERY:
            frame = self._recovery_phase(frame, yolo_result, w, h)

        elif self.phase in (Phase.FIND_HELIPAD, Phase.LANDING):
            frame = self._helipad_phase(frame, yolo_result, w, h)

        # Draw HUD
        frame = draw_mission_hud(frame, self)
        return frame

    # ── Takeoff phase ─────────────────────────────────────────────────────────

    def _takeoff_phase(self, frame, current_altitude, frame_w, frame_h):
        """
        Controlled ascent to safe flight altitude before mission begins.

        Outputs a ASCEND FlightCommand until altitude ≥ takeoff_alt.
        Transitions to SEARCHING once safe altitude is confirmed.

        current_altitude: read from barometer/rangefinder via MAVLink.
        When None, we cannot confirm altitude from sensors, so we output
        the ASCEND command and trust the flight controller's auto-takeoff.
        """
        # Determine if we've reached safe altitude
        # Use real sensor value if provided, else rely on FC takeoff mode
        at_altitude = (current_altitude is not None and
                       current_altitude >= self._takeoff_alt)

        if at_altitude:
            self.phase = Phase.SEARCHING
            # Award takeoff bonus (§VI — 20 pts Qualifier, 10 pts Finals)
            if not self._takeoff_scored:
                bonus = SCORE_TAKEOFF_FINALS if self.is_finals else SCORE_TAKEOFF_QUAL
                self.score += bonus
                self._takeoff_scored = True
                print(f"[MISSION] Takeoff bonus +{bonus} pts → score={self.score}")
            self.last_command = FlightCommand(
                vz=0.0, yaw_lock=True,
                phase="TAKEOFF", action="REACHED_ALT"
            )
            print(f"[TAKEOFF] Altitude {current_altitude:.2f}m reached → SEARCHING")
        else:
            # Gentle ascent command — FC executes, we just report intent
            ascent_vz = -0.8   # m/s upward in NED (negative = up)
            alt_str   = f"{current_altitude:.2f}m" if current_altitude else "sensor=None"
            self.last_command = FlightCommand(
                vz=ascent_vz, yaw_lock=True,
                phase="TAKEOFF", action="ASCEND"
            )
            # HUD overlay
            h, w = frame.shape[:2]
            cv2.putText(frame, f"TAKEOFF — ascending (alt: {alt_str})",
                        (frame_w // 2 - 160, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(frame, f"Target: {self._takeoff_alt:.1f}m",
                        (frame_w // 2 - 80, frame_h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        return frame

    # ── Recovery phase ────────────────────────────────────────────────────────

    def _recovery_phase(self, frame, result, frame_w, frame_h):
        """
        Structured yaw-scan to relocate a lost fire target.

        Instead of blindly resetting to SEARCHING (which just hovers in place),
        the drone executes a yaw sweep: rotate right 30° → left 60° → centre.
        Each step lasts _scan_step_frames frames at 30 fps (~0.67 s each).

        If the target is spotted during recovery, transition back to TARGETING.
        If the full scan completes without finding the target, advance to the
        next target in sequence (assume this one was scored / missed).
        """
        self._recovery_frames += 1

        names    = result.names if hasattr(result, "names") else {}
        target   = self.current_target
        h_bboxes = collect_h_landing_bboxes(result, names)

        # Check if target is visible during scan
        target_found = False
        if target is not None and result.masks is not None:
            target_color = target[0]
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id = int(box.cls[0].item())
                if names.get(cls_id, "") != "circle":
                    continue
                geo = _extract_geometry(mask_obj, frame_w, frame_h)
                if geo is None:
                    continue
                cx, cy, radius, fr, cnt = geo
                if fr < FILL_RATIO_MIN_CIRCLE:
                    continue
                if h_bboxes and overlaps_h_landing(cx, cy, radius, h_bboxes):
                    continue
                x1b = max(0, cx - radius); y1b = max(0, cy - radius)
                x2b = min(frame_w, cx + radius); y2b = min(frame_h, cy + radius)
                if x2b > x1b and y2b > y1b:
                    roi = frame[y1b:y2b, x1b:x2b]
                    roi_m = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cr    = np.clip(cnt.reshape(-1, 2) - [x1b, y1b], 0,
                                    [roi.shape[1]-1, roi.shape[0]-1])
                    cv2.fillPoly(roi_m, [cr.reshape(-1, 1, 2)], 255)
                    if color_of_mask(roi_m, roi) == target_color:
                        target_found = True
                        break

        if target_found:
            # Back on target — resume approach
            self.phase = Phase.TARGETING
            self._recovery_frames  = 0
            self._scan_step        = 0
            self._no_detect_frames = 0
            self._pid_x.reset()
            self._pid_y.reset()
            self.last_command = FlightCommand(
                yaw_lock=True, phase="RECOVERY", action="TARGET_REACQUIRED"
            )
            return frame

        # Full scan exhausted — skip this target
        if self._recovery_frames >= self._recovery_max:
            print(f"[RECOVERY] Scan exhausted — skipping target {self.current_target_idx}")
            self._recovery_frames  = 0
            self._scan_step        = 0
            self._no_detect_frames = 0
            if self.current_target_idx < len(self._fire_sequence) - 1:
                self.current_target_idx += 1
            self.phase = Phase.SEARCHING
            self.last_command = FlightCommand(
                yaw_lock=True, phase="RECOVERY", action="SKIP_TARGET"
            )
            return frame

        # Compute yaw-scan command for this frame
        step_idx  = min(self._recovery_frames // self._scan_step_frames,
                        len(self._scan_yaw_steps) - 1)
        yaw_rate  = float(self._scan_yaw_steps[step_idx])
        self.last_command = FlightCommand(
            vx=0.0, vy=0.0, vz=0.0,
            yaw_rate=yaw_rate, yaw_lock=False,
            phase="RECOVERY", action=f"SCAN_STEP_{step_idx}"
        )

        # HUD
        cv2.putText(frame,
                    f"RECOVERY — scanning (step {step_idx+1}/{len(self._scan_yaw_steps)}"
                    f"  yaw_rate={yaw_rate:+.0f}°/s)",
                    (10, frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Draw recovery progress bar
        bar_w_total = 200
        prog = int(self._recovery_frames / self._recovery_max * bar_w_total)
        cv2.rectangle(frame, (10, frame_h - 20), (10 + bar_w_total, frame_h - 8),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (10, frame_h - 20), (10 + prog, frame_h - 8),
                      (0, 165, 255), -1)

        return frame

    # ── Fire-fighting phase logic ─────────────────────────────────────────────

    def _fire_phase(self, frame, result, frame_w, frame_h,
                    current_altitude: float | None = None):
        """
        Altitude authority hierarchy for the drop decision:

          1. current_altitude (barometer/rangefinder from MAVLink)  ← AUTHORITATIVE
          2. Z_est from pinhole model                               ← FALLBACK ONLY

        Why this matters: when the drone pitches to correct position, the camera
        tilts down.  A circle then projects as an ellipse on the sensor.
        cv2.minEnclosingCircle fits the major axis → d_px is inflated → Z_est
        is underestimated → the drop gate could trigger below 3.0 m (penalty).

        If current_altitude is provided, it is used for the altitude_safe check
        and for the descent vz command.  Z_est is still displayed on the HUD and
        used for approach decisions when no sensor is connected.
        """
        target = self.current_target
        if target is None:
            self._transition_to_helipad()
            return frame

        target_color, target_label, target_pts, D_real = target
        names = result.names if hasattr(result, 'names') else {}

        # ── Cross-class NMS prep ─────────────────────────────────────────────
        # Collect H_landing detections first. Any circle/semicircle whose bbox
        # overlaps one of these is part of the pad structure, not a fire target.
        h_bboxes = collect_h_landing_bboxes(result, names)

        best_match = None   # (Z_est, cx, cy, radius, color, fr, cnt_frame)

        if result.masks is not None:
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id     = int(box.cls[0].item())
                class_name = names.get(cls_id, "")
                if class_name != "circle":
                    continue

                # Optimized geometry — no full-frame resize
                geo = _extract_geometry(mask_obj, frame_w, frame_h)
                if geo is None:
                    continue
                cx, cy, radius, fr, cnt_frame = geo

                if fr < FILL_RATIO_MIN_CIRCLE:
                    cv2.drawContours(frame, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                d_ratio = compute_diameter_ratio(radius, frame_w)
                if d_ratio > MAX_FIRE_DIAMETER_RATIO:
                    cv2.drawContours(frame, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                if h_bboxes and overlaps_h_landing(cx, cy, radius, h_bboxes):
                    continue


                # ── Filter 4: scored-position exclusion ─────────────────
                # Reject detections that overlap an already-scored circle.
                # Fixes the F1→F2 re-lock bug: both are red, so color check
                # passes on F1 even after we've advanced idx to F2.
                already_scored = any(
                    ((cx - sx) ** 2 + (cy - sy) ** 2) < SCORED_RADIUS_PX ** 2
                    for sx, sy in self._scored_positions
                )
                if already_scored:
                    cv2.drawContours(frame, [cnt_frame], -1, (80, 80, 80), 1)
                    cv2.putText(frame, "SCORED", (cx - 25, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                    continue

                # Altitude estimation (Fix: replaces diameter_ratio >= 0.70)
                Z_est = self.alt_est.estimate(radius * 2.0, D_real)

                # Color check using bbox ROI (avoids full-frame resize)
                x1b = max(0, cx - radius)
                y1b = max(0, cy - radius)
                x2b = min(frame_w, cx + radius)
                y2b = min(frame_h, cy + radius)
                detected_color = None
                if x2b > x1b and y2b > y1b:
                    roi_bgr  = frame[y1b:y2b, x1b:x2b]
                    roi_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
                    cnt_roi  = (cnt_frame.reshape(-1, 2)
                                - np.array([x1b, y1b], dtype=np.int32))
                    cnt_roi  = np.clip(cnt_roi, 0,
                                       [roi_bgr.shape[1]-1, roi_bgr.shape[0]-1])
                    cv2.fillPoly(roi_mask, [cnt_roi.reshape(-1, 1, 2)], 255)
                    detected_color = color_of_mask(roi_mask, roi_bgr)

                if detected_color != target_color:
                    cv2.drawContours(frame, [cnt_frame], -1, (60, 60, 60), 1)
                    continue

                # Prefer detection closest to optimal drop altitude
                if (best_match is None or
                        abs(Z_est - DROP_ALTITUDE_TARGET) <
                        abs(best_match[0] - DROP_ALTITUDE_TARGET)):
                    best_match = (Z_est, cx, cy, radius, detected_color, fr,
                                  cnt_frame)

        if best_match is None:
            self._no_detect_frames += 1
            if self._no_detect_frames >= self._max_no_detect:
                # Lost target too long — start structured scan instead of blind reset
                self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
                self._pid_x.reset()
                self._pid_y.reset()
                self._recovery_frames = 0
                self._scan_step       = 0
                self.phase = Phase.RECOVERY
                self.last_command = FlightCommand(
                    yaw_lock=True, phase="FIRE", action="ENTERING_RECOVERY"
                )
            else:
                # Target lost for a few frames only — hover in place.
                # BUG FIX: without this, last_command keeps its previous non-zero
                # vx/vy, causing the drone to continue drifting toward a target
                # that may no longer be there.
                self.last_command = FlightCommand(
                    vx=0.0, vy=0.0, vz=0.0,
                    yaw_lock=True, phase="FIRE",
                    action=f"NO_DETECT_{self._no_detect_frames}/{self._max_no_detect}",
                )
            return frame

        self._no_detect_frames = 0
        Z_est, raw_cx, raw_cy, raw_r, color, fr, cnt_frame = best_match
        cx, cy, _  = self._smooth(raw_cx, raw_cy, Z_est)
        radius     = raw_r
        color_bgr  = _color_bgr(color)

        # PID alignment
        offset_x = cx - frame_w // 2
        offset_y = cy - frame_h // 2
        vel_x    = self._pid_x.update(float(offset_x))
        vel_y    = self._pid_y.update(float(offset_y))
        aligned  = (abs(offset_x) < ALIGN_PIXEL_THRESH and
                    abs(offset_y) < ALIGN_PIXEL_THRESH)

        # ── Altitude authority ────────────────────────────────────────────────
        # Hardware sensor (rangefinder/barometer) is AUTHORITATIVE when connected.
        # Vision Z_est is the fallback — it is unreliable during active pitch because
        # a tilted camera projects circles as ellipses (major axis inflates d_px,
        # causing Z_est underestimation that could trigger a drop below 3.0 m).
        Z_authority   = current_altitude if current_altitude is not None else Z_est
        altitude_safe = DROP_ALTITUDE_MIN <= Z_authority <= DROP_ALTITUDE_MAX

        if altitude_safe and aligned and self._drop_cooldown == 0:
            self.phase = Phase.DROPPING
            self._execute_drop(target_label, target_pts, Z_authority, cx=cx, cy=cy)
        elif Z_authority < DROP_ALTITUDE_MIN:
            self.phase = Phase.TARGETING   # too low — continue controlling while ascending
        else:
            self.phase = Phase.TARGETING   # descending / aligning

        # ── Draw ─────────────────────────────────────────────────────────────
        cv2.drawContours(frame, [cnt_frame], -1, color_bgr, 2)
        cv2.circle(frame, (cx, cy), 5, color_bgr, -1)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color_bgr, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color_bgr, 1)

        fc_x, fc_y = frame_w // 2, frame_h // 2
        cv2.line(frame, (fc_x-15, fc_y), (fc_x+15, fc_y), (255, 255, 255), 1)
        cv2.line(frame, (fc_x, fc_y-15), (fc_x, fc_y+15), (255, 255, 255), 1)

        # ── FlightCommand from PID + guarded feedforward ──────────────────────
        cmd_vx = float(np.clip(-vel_y * PX_TO_MS_FIRE, -1.0, 1.0))
        cmd_vy = float(np.clip( vel_x * PX_TO_MS_FIRE, -1.0, 1.0))
        # NED convention: vz > 0 = descend (down), vz < 0 = ascend (up).
        # Too HIGH (Z > 4.5m) → descend → +vz.  Too LOW (Z < 3.05m) → ascend → -vz.
        cmd_vz = (0.0   if altitude_safe else
                   0.6  if Z_authority > DROP_ALTITUDE_MAX else   # too high → descend
                  -0.5)                                            # too low  → ascend

        # Feedforward: anticipate moving F2 target (Finals only).
        # GUARD: suppress feedforward when the drone is actively manoeuvring
        # (large cmd_vx/vy). In that state, camera tilt causes apparent pixel
        # movement that is drone-motion, not target-motion. Applying feedforward
        # on top creates a positive-feedback loop (pitch → pixel shift → more
        # pitch → more pixel shift → crash). The threshold 0.35 m/s means the
        # drone is executing a significant correction — at that point we trust
        # the PID alone and zero the feedforward contribution.
        FF_ACTIVE_THRESHOLD = 0.35   # m/s — above this, feedforward is unsafe
        if self.is_finals:
            drone_moving = (abs(cmd_vx) > FF_ACTIVE_THRESHOLD or
                            abs(cmd_vy) > FF_ACTIVE_THRESHOLD)
            if not drone_moving:
                ff_vx_px, ff_vy_px = self._feedforward_velocity()
                FF_SCALE = 0.003   # px/s → m/s (tune on field)
                cmd_vx  += float(np.clip(-ff_vy_px * FF_SCALE, -0.4, 0.4))
                cmd_vy  += float(np.clip( ff_vx_px * FF_SCALE, -0.4, 0.4))
                cmd_vx   = float(np.clip(cmd_vx, -1.0, 1.0))
                cmd_vy   = float(np.clip(cmd_vy, -1.0, 1.0))

        action = ("DROP"    if (altitude_safe and aligned) else
                  "ALIGN"   if altitude_safe else
                  "DESCEND" if Z_authority > DROP_ALTITUDE_MAX else "ASCEND")

        self.last_command = FlightCommand(
            vx=cmd_vx, vy=cmd_vy, vz=cmd_vz,
            yaw_rate=0.0, yaw_lock=True,
            drop_ball=(self.phase == Phase.DROPPING),
            phase="FIRE", action=action,
        )

        # HUD label — show which altitude source is active
        if current_altitude is not None:
            alt_str = f"Z={current_altitude:.2f}m[SNS]"   # sensor = authoritative
        elif Z_est < 99:
            alt_str = f"Z={Z_est:.2f}m[VIS]"              # vision = fallback
        else:
            alt_str = "Z=?"
        label   = f"{target_label} [{color}] | {alt_str} | fr={fr:.2f} | {action}"
        label_y = max(20, cy - radius - 12)
        cv2.putText(frame, label, (max(0, cx - 80), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        _draw_altitude_bar(frame, Z_authority, frame_w, frame_h, color_bgr)
        return frame

    def _execute_drop(self, label: str, points: int, Z_est: float = 0.0,
                      cx: int = -1, cy: int = -1):
        """Register a successful ball drop and record centroid for exclusion."""
        self.balls_dropped += 1
        self.score         += points
        # Time-based cooldown (2 seconds) instead of hardcoded 60 frames.
        # At 20 FPS: 40 frames.  At 30 FPS: 60 frames.
        self._drop_cooldown = int(2.0 * self._fps_hint)
        self._pid_x.reset()
        self._pid_y.reset()
        self._pos_history.clear()

        # Record centroid so Filter 4 can suppress this circle going forward
        if cx >= 0 and cy >= 0:
            self._scored_positions.append((cx, cy))
            print(f"[MISSION] Scored position recorded: ({cx},{cy}) "
                  f"— exclusion radius {SCORED_RADIUS_PX}px")

        print(f"[MISSION] DROP {label} at Z={Z_est:.2f}m "
              f"(+{points} pts) | {self.balls_dropped}/5 | score={self.score}")
        # Stub: replace with servo / MAVLink drop command
        # e.g. mavlink_conn.send_ball_drop()

        if self.balls_dropped >= 5:
            self._transition_to_helipad()
        else:
            self.current_target_idx += 1
            self.phase = Phase.SEARCHING


    def _transition_to_helipad(self):
        print(f"[MISSION] All balls dropped! Score so far: {self.score}")
        print("[MISSION] Switching to FIND_HELIPAD mode...")
        self.phase = Phase.FIND_HELIPAD
        # BUG FIX: reset the shared counter so helipad phase starts clean.
        # Without this, any residual no_detect count from the fire phase could
        # cause the helipad scan to start immediately instead of waiting the
        # correct _max_no_detect frames.
        self._no_detect_frames = 0
        self._smooth_cx = self._smooth_cy = self._smooth_ratio = None

    # ── Helipad phase logic ───────────────────────────────────────────────────

    def _helipad_phase(self, frame, result, frame_w, frame_h):
        names = result.names if hasattr(result, 'names') else {}
        helipad_color = (255, 128, 0)   # orange for H_landing

        found = False
        if result.masks is not None:
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id     = int(box.cls[0].item())
                class_name = names.get(cls_id, "")
                if class_name != "H_landing":
                    continue

                found = True
                x1, y1, x2, y2 = (int(box.xyxy[0][k].item()) for k in range(4))
                pad_w  = x2 - x1
                pad_cx = (x1 + x2) // 2
                pad_cy = (y1 + y2) // 2

                d_ratio = float(pad_w) / float(frame_w)
                action  = _decide_action(d_ratio)

                # Helipad approach velocity — proportional to offset from centre
                pad_offset_x = pad_cx - frame_w // 2
                pad_offset_y = pad_cy - frame_h // 2
                # PX_TO_MS_HELIPAD tuned for precise final approach (FC ATC_RAT P=0.135)
                h_vx = float(np.clip(-pad_offset_y * PX_TO_MS_HELIPAD, -0.5, 0.5))
                h_vy = float(np.clip( pad_offset_x * PX_TO_MS_HELIPAD, -0.5, 0.5))
                h_vz = 0.3 if d_ratio < DROP_THRESHOLD else 0.0  # descend

                if d_ratio >= DROP_THRESHOLD:
                    if self.phase != Phase.LANDING:
                        print(f"[MISSION] Helipad in range (d={d_ratio:.2f}) — LANDING")
                    self.phase = Phase.LANDING
                    h_vz = 0.4   # final descent
                    # Stub: e.g. mavlink_conn.set_mode("LAND")
                else:
                    self.phase = Phase.FIND_HELIPAD

                self.last_command = FlightCommand(
                    vx=h_vx, vy=h_vy, vz=h_vz,
                    yaw_rate=0.0, yaw_lock=True,
                    phase=self.phase.name, action=action,
                )

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), helipad_color, 2)
                cv2.circle(frame, (pad_cx, pad_cy), 6, helipad_color, -1)
                cv2.line(frame, (pad_cx - 20, pad_cy), (pad_cx + 20, pad_cy), helipad_color, 1)
                cv2.line(frame, (pad_cx, pad_cy - 20), (pad_cx, pad_cy + 20), helipad_color, 1)

                label = f"H_landing | d={d_ratio:.2f} | {action}"
                cv2.putText(frame, label, (x1, max(14, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, helipad_color, 2)

                _draw_ratio_bar(frame, d_ratio, frame_w, frame_h, helipad_color)

                # Landing complete when drone is very close AND on ground
                # In practice: check altitude sensor, not just camera ratio.
                # Set COMPLETE externally when land command confirms touchdown.
                break

        if not found:
            self._no_detect_frames += 1
            if self._no_detect_frames >= self._max_no_detect:
                self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
                self.phase = Phase.FIND_HELIPAD  # keep searching, don't land blindly
                # Yaw scan — same sweep pattern as RECOVERY phase
                scan_frame  = self._no_detect_frames - self._max_no_detect
                step_idx    = min(scan_frame // self._scan_step_frames,
                                  len(self._scan_yaw_steps) - 1)
                yaw_rate    = float(self._scan_yaw_steps[step_idx])
                self.last_command = FlightCommand(
                    vx=0.0, vy=0.0, vz=0.0,
                    yaw_rate=yaw_rate, yaw_lock=False,
                    phase="FIND_HELIPAD", action=f"SCAN_STEP_{step_idx}",
                )
                cv2.putText(frame,
                            f"HELIPAD SCAN — step {step_idx+1} "
                            f"yaw={yaw_rate:+.0f} deg/s",
                            (10, frame_h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
        else:
            self._no_detect_frames = 0

        return frame

    def signal_landed(self):
        """
        Call when flight controller confirms touchdown (MAVLink LAND_COMPLETE).
        Awards landing bonus score (§VI — 30 pts Qualifier, 20 pts Finals).
        """
        if not self._landing_scored:
            bonus = SCORE_LANDING_FINALS if self.is_finals else SCORE_LANDING_QUAL
            self.score += bonus
            self._landing_scored = True
            print(f"[MISSION] Landing bonus +{bonus} pts -> score={self.score}")
        self.phase = Phase.COMPLETE
        self.last_command = FlightCommand(phase="COMPLETE", action="LANDED")
        print(f"[MISSION] LANDED. Final score: {self.score} pts  "
              f"(penalties: {self.penalties} pts  "
              f"net: {self.score + self.penalties} pts)")

    def reset(self):
        """
        Full mission reset for RETRY (rulebook §VII).

        Teams may retry once per round. Clock keeps running. All scores
        from before the retry are reset to zero — nothing carries over.

        Usage:
            mission.reset()
            # Fly drone back to start manually, then resume or call skip_takeoff()
        """
        self.phase              = Phase.TAKEOFF
        self.balls_dropped      = 0
        self.current_target_idx = 0
        self.score              = 0
        self.penalties          = 0
        self._drop_cooldown     = 0
        self._takeoff_scored    = False
        self._landing_scored    = False
        self._no_detect_frames  = 0
        self._recovery_frames   = 0
        self._scan_step         = 0
        self._low_alt_frames    = 0
        self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
        self._pos_history.clear()
        self._scored_positions.clear()
        self._pid_x.reset()
        self._pid_y.reset()
        self.last_command = FlightCommand(phase="TAKEOFF", action="RETRY_RESET")
        print("[MISSION] RETRY — all scores reset. Return drone to start position.")

    def skip_takeoff(self):
        """
        Manually advance past TAKEOFF to SEARCHING.
        Use when testing on pre-recorded video or when FC handles takeoff externally.
        NOTE: takeoff bonus NOT awarded (no auto takeoff confirmed).
        """
        if self.phase == Phase.TAKEOFF:
            self.phase = Phase.SEARCHING
            print("[MISSION] Takeoff phase skipped -> SEARCHING")


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------

def draw_mission_hud(frame: np.ndarray, mission: MissionStateMachine) -> np.ndarray:
    """
    Draw the mission status HUD in the top-left corner.
    Shows: phase, balls, score, next target, altitude window, finals mode.
    """
    h, w = frame.shape[:2]
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 280, 132
    target = mission.current_target

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.48
    lh   = 22

    phase_colors = {
        Phase.TAKEOFF:      (255, 220,  50),
        Phase.SEARCHING:    (180, 180, 180),
        Phase.TARGETING:    (30,  200, 255),
        Phase.DROPPING:     (0,   80,  255),
        Phase.RECOVERY:     (0,   165, 255),
        Phase.FIND_HELIPAD: (255, 180,  60),
        Phase.LANDING:      (60,  255, 180),
        Phase.COMPLETE:     (0,   255,  80),
    }
    pc = phase_colors.get(mission.phase, (200, 200, 200))

    round_tag = "[FINALS]" if mission.is_finals else "[QUAL]"
    lines = [
        (f"Phase: {mission.phase.name}  {round_tag}", pc),
        (f"Balls: {mission.balls_dropped}/5  |  Score: {mission.score} pts",
         (200, 200, 200)),
        (f"Penalties: {mission.penalties} pts  "
         f"Net: {mission.score + mission.penalties} pts",
         (200, 80, 80) if mission.penalties < 0 else (140, 140, 140)),
        (f"Drop Z: {DROP_ALTITUDE_MIN:.1f}–{DROP_ALTITUDE_MAX:.1f}m  "
         f"(align ±{ALIGN_PIXEL_THRESH}px)", (140, 140, 140)),
    ]

    if target and mission.phase not in (Phase.FIND_HELIPAD, Phase.LANDING,
                                         Phase.COMPLETE):
        color_name, label, pts, D_real = target   # 4-element tuple
        color_indicator = _color_bgr(color_name)
        lines.append((f"Next: {label} {color_name} D={D_real:.1f}m (+{pts}pt)",
                      color_indicator))
    elif mission.phase in (Phase.FIND_HELIPAD, Phase.LANDING):
        lines.append(("Target: H_landing pad", (255, 128, 0)))
    elif mission.phase == Phase.COMPLETE:
        lines.append(("MISSION COMPLETE", (0, 255, 80)))

    for i, (text, color) in enumerate(lines):
        y = panel_y + 20 + i * lh
        cv2.putText(frame, text, (panel_x + 8, y), font, fs, color, 1,
                    cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_bgr(color_name: str) -> tuple:
    return {
        "red":    (0,   40,  220),
        "yellow": (0,   220, 220),
        "blue":   (220, 60,  0),
    }.get(color_name, (180, 180, 180))


# _decide_action is kept for H_landing approach only.
# Fire-target DROP is now altitude-based, not ratio-based.
def _decide_action(d_ratio: float) -> str:
    if d_ratio >= DROP_THRESHOLD: return "LAND"
    if d_ratio >= 0.40:           return "APPROACH"
    if d_ratio >= 0.15:           return "ALIGN"
    return "SEARCH"


def _draw_ratio_bar(frame, ratio, frame_w, frame_h, color_bgr):
    """Approach bar for H_landing (diameter_ratio 0→1)."""
    bx, by = 10, frame_h - 36
    bw, bh = 180, 14
    fill   = int(min(ratio, 1.0) * bw)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color_bgr, -1)
    tx = bx + int(DROP_THRESHOLD * bw)
    cv2.line(frame, (tx, by - 2), (tx, by + bh + 2), (255, 255, 255), 1)
    cv2.putText(frame, f"d={ratio:.2f}", (bx + bw + 6, by + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def _draw_altitude_bar(frame, Z_est: float, frame_w: int, frame_h: int,
                       color_bgr: tuple) -> None:
    """
    Altitude bar for fire targets (replaces diameter_ratio bar).

    Shows estimated Z against the safe drop window [3.0 m – 5.0 m].
    Bar fills green in the safe window, red outside it.

      |----[====DROP_MIN===DROP_MAX====]--------|
      0m                                      6m
    """
    bx, by = 10, frame_h - 36
    bw, bh = 200, 14
    Z_MAX_DISPLAY = 6.0   # bar represents 0 → 6 m

    # Background
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)

    # Safe-window highlight (green band)
    safe_x1 = bx + int(DROP_ALTITUDE_MIN / Z_MAX_DISPLAY * bw)
    safe_x2 = bx + int(DROP_ALTITUDE_MAX / Z_MAX_DISPLAY * bw)
    cv2.rectangle(frame, (safe_x1, by), (safe_x2, by + bh), (0, 80, 0), -1)

    # Current altitude marker
    Z_clamped = max(0.0, min(Z_est, Z_MAX_DISPLAY))
    marker_x  = bx + int(Z_clamped / Z_MAX_DISPLAY * bw)
    in_safe   = DROP_ALTITUDE_MIN <= Z_est <= DROP_ALTITUDE_MAX
    marker_col = color_bgr if in_safe else (0, 0, 200)
    cv2.line(frame, (marker_x, by - 2), (marker_x, by + bh + 2), marker_col, 2)

    # Min/max threshold lines
    cv2.line(frame, (safe_x1, by - 3), (safe_x1, by + bh + 3), (0, 220, 0), 1)
    cv2.line(frame, (safe_x2, by - 3), (safe_x2, by + bh + 3), (0, 180, 0), 1)

    label = f"Z={Z_est:.2f}m" if Z_est < 99 else "Z=?"
    cv2.putText(frame, label, (bx + bw + 6, by + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"{DROP_ALTITUDE_MIN:.0f}m",
                (safe_x1 - 4, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 200, 0), 1)
    cv2.putText(frame, f"{DROP_ALTITUDE_MAX:.0f}m",
                (safe_x2 - 4, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 160, 0), 1)
