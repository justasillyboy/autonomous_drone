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
  4. Once the drone is in the safe drop window (3.05–4.50 m altitude) AND
     the target centre is within ALIGN_PIXEL_THRESH pixels of frame centre AND
     attitude is level (|pitch|,|roll| < 8°), it signals a ball drop.
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

import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mission constants
# ---------------------------------------------------------------------------

# H_landing approach threshold: helipad bbox width must reach this fraction of
# frame width before switching from APPROACH → LAND descent mode.
# This is NOT used for fire-circle drops (those are altitude-based).
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

# Edge-clip detection margin: if a contour's scaled bounding box is within this many
# pixels of any frame boundary, the visible arc is a partial circle — the radius from
# cv2.minEnclosingCircle is unreliable and the derived altitude Z_est is untrustworthy.
EDGE_CLIP_MARGIN_PX = 5   # pixels

# ---------------------------------------------------------------------------
# Competition fire target specs  (verified against rulebook 27/02/2026)
# Each entry: (color_name, label, points, real_diameter_m)
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

# Real physical diameters (m) keyed by colour — used for altitude estimation.
# Derived from FIRE_TARGETS_QUALIFIER so the per-target tuple is the single
# source of truth; hand-editing this dict is therefore impossible.
FIRE_REAL_DIAMETERS: dict[str, float] = {
    color: diameter for (color, _label, _pts, diameter) in FIRE_TARGETS_QUALIFIER
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
SCORED_RADIUS_PX     = 80     # pixels  (kept for legacy reference)

# Exclusion zone for scored circles expressed in WORLD space (metres).
# Using world coordinates makes the zone altitude-invariant: the same physical
# 0.60 m radius is enforced whether the drone is at 3.0 m or 4.5 m.
# 0.60 m ≈ the radius of the 1.0 m red circles (F1/F2) plus a 10 cm margin.
# Also applied in _recovery_phase to prevent re-locking on scored targets
# during the yaw scan (which would cause an infinite RECOVERY ↔ TARGETING loop).
SCORED_WORLD_RADIUS_M = 0.60   # metres

# ---------------------------------------------------------------------------
# Competition rule constants  (rulebook 27/02/2026 — DO NOT EDIT)
# ---------------------------------------------------------------------------

# Timing
MATCH_TIME_S           = 300  # 5 minutes total for Round 1 + Round 2 combined
# PREP_TIME_S = 60                # setup time at the field before clock starts (not enforced in code)
ALT_PENALTY_TIME_S     =   5  # seconds below 3.0 m before −5 pt altitude penalty
LAND_EARLY_AT_S        = 285.0  # force FIND_HELIPAD transition at this elapsed time (15 s margin before T=300 s)
HELIPAD_FORCE_LAND_S   = 5.0   # remaining time threshold (s) to trigger emergency 1.0 m/s descent

# Penalties (§VII)
# Only PENALTY_ALTITUDE is self-tracked in code (altitude reading is onboard).
# TZ-gate collisions (−5 pts each) and field-boundary violations (−5 pts per 5 s)
# are rulebook penalties the drone cannot self-detect without sensors it doesn't
# carry (impact sensor / GPS fence) — they are referee-applied only and therefore
# intentionally NOT defined here. See config.yaml for rulebook reference values.
PENALTY_ALTITUDE     = -5   # pts — below 3.0 m during auto flight > 5 seconds

# Bonus scores (added to self.score on successful task)
SCORE_TAKEOFF_QUAL   = 20   # auto takeoff ≥ 3.0 m — Qualifier / Semi-final
SCORE_TAKEOFF_FINALS = 10   # auto takeoff ≥ 3.0 m — Finals
SCORE_LANDING_QUAL   = 30   # auto land in helipad — Qualifier / Semi-final
SCORE_LANDING_FINALS = 20   # auto land in helipad — Finals

# Field geometry (metres) — for geofencing (§IV).
# Only the three values actually consumed by the state machine are kept here;
# TZ-ring dimensions and fire-circle mount height are rulebook facts documented
# in config.yaml and README.md but never referenced from code.
FIELD_LENGTH    = 24.0
FIELD_WIDTH     = 12.0
SAFE_ZONE_DEPTH =  3.5   # along field length axis

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
# SensorData — unified input from MTF-01 + BNO080
#
# Populate this once per frame in your MAVLink/I2C loop and pass it to
# MissionStateMachine.update().  All fields are optional (None = not available).
#
# Wiring guide:
#   MTF-01  → ArduPilot serial port, SERIAL_PROTOCOL=1 (MAVLink).
#             ArduPilot then re-broadcasts DISTANCE_SENSOR and OPTICAL_FLOW_RAD
#             over the companion-computer telemetry stream (SERIAL2 etc.).
#             Read with pymavlink:
#               msg = conn.recv_match(type='DISTANCE_SENSOR', blocking=False)
#               tof_alt = msg.current_distance / 100.0  # cm → m
#               msg = conn.recv_match(type='OPTICAL_FLOW_RAD', blocking=False)
#               flow_vx = msg.integrated_x / msg.integration_time_us * 1e6  # rad/s
#               # Convert rad/s → m/s: v = angular_rate * altitude
#               of_vx_ms = flow_vx * tof_alt
#
#   BNO080  → I2C (address 0x4A default) or UART.
#             Use Adafruit CircuitPython BNO08x library:
#               from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR, \
#                                          BNO_REPORT_GYROSCOPE
#               pitch, roll, yaw = euler_from_quaternion(bno.game_quaternion)
#               gx, gy, gz = bno.gyro  # rad/s body frame
# ---------------------------------------------------------------------------

@dataclass
class SensorData:
    """
    Per-frame sensor readings from MTF-01 and BNO080.
    All fields are optional — None means sensor not connected or data not fresh.
    """
    # ── MTF-01 Time-of-Flight laser rangefinder ────────────────────────────
    # Direct laser ranging to the floor.  More accurate than barometer at
    # 3–5 m, immune to air-pressure transients during forward flight.
    # Priority order for altitude: fc_altitude > tof_altitude > Z_est_vision
    tof_altitude_m: float | None = None    # metres above floor, ±1 cm accuracy

    # ── MTF-01 Optical flow ────────────────────────────────────────────────
    # Body-frame drone velocity in m/s (NOT the target velocity).
    # Used to remove drone-motion contribution from the pixel history so that
    # _feedforward_velocity() estimates pure TARGET motion only.
    of_vx_ms: float | None = None    # forward  (body X)
    of_vy_ms: float | None = None    # lateral  (body Y)

    # ── BNO080 attitude (euler angles, degrees) ────────────────────────────
    pitch_deg: float | None = None   # nose up = positive
    roll_deg:  float | None = None   # right wing down = positive
    yaw_deg:   float | None = None   # clockwise from north = positive

    # ── BNO080 angular velocity (deg/s, body frame) ────────────────────────
    # Used to detect active maneuvering for feedforward guard — more reliable
    # than using cmd_vx/vy as a proxy because it measures ACTUAL drone motion,
    # not just our intended command.
    pitch_rate_dps: float | None = None   # deg/s pitch rate
    roll_rate_dps:  float | None = None   # deg/s roll rate


# ---------------------------------------------------------------------------
# Attitude-gated drop thresholds  (BNO080)
# ---------------------------------------------------------------------------

# Maximum tilt allowed when releasing a ball.
# At 8° pitch, a ball released at 3.5 m altitude lands ~49 cm off-centre —
# already close to the edge of the 50 cm radius F5 blue circle.
# At 12° the miss distance grows to ~74 cm — outside any target.
DROP_MAX_PITCH_DEG = 8.0   # degrees — refuse drop if |pitch| > this
DROP_MAX_ROLL_DEG  = 8.0   # degrees — refuse drop if |roll|  > this

# Feedforward guard: actual angular rate threshold (BNO080) for suppressing
# feedforward.  Replaces the cmd_vx > 0.35 m/s proxy.
FF_ANGULAR_RATE_GATE_DPS = 12.0  # deg/s — above this the camera is tilting too fast


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
    Identify the dominant colour inside a segmentation mask.

    Uses HSV masking instead of raw BGR so it is robust to lighting changes.

    Args:
        mask_binary : uint8 binary mask (H, W) — 255 inside object, 0 outside.
                      Must have the same (H, W) as frame_bgr.
        frame_bgr   : BGR image region corresponding to the mask area.
                      Callers typically pass a tight ROI crop, NOT the full frame.

    Returns:
        'red' | 'yellow' | 'blue' | None (if no colour dominates).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    roi = cv2.bitwise_and(hsv, hsv, mask=mask_binary)

    total_pixels = max(1, int(mask_binary.sum() / 255))
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

# Feedforward scale factor (Finals F2): target-velocity (px/s) → drone additive velocity (m/s).
# Decrease if the drone overshoots the intercept point; increase if tracking lags.
FF_SCALE = 0.003

# Feedforward velocity clamp: maximum pixel/s velocity accepted from the 8-frame linear fit.
# Values above this indicate camera tilt or optical-flow noise, not real target motion.
MAX_FF_PX_S = 60.0   # px/s

# Feedforward suppression proxy — used when BNO080 angular-rate data is unavailable.
# If the estimated command speed in either axis exceeds this value, the drone is
# actively maneuvering and the pixel history reflects camera tilt, not target motion.
FF_CMD_THRESHOLD_MS = 0.35   # m/s

# Tilt gate relaxed threshold — applied after TILT_RELAX_AFTER_S of continuous tilt-block.
# At 15° pitch/roll from 3.5 m the ball lands ~93 cm off-centre (misses F5, 40 cm radius),
# but is still preferable to a "never-drop" deadlock caused by sustained crosswind.
TILT_MAX_RELAXED_DEG = 15.0   # degrees

# Drop cooldown duration (seconds). Blocks a second release after a ball drop to prevent
# double-drop from servo bounce or a stale YOLO detection on the same frame.
DROP_COOLDOWN_S = 2.0   # seconds — converted to frames in _execute_drop()

# ToF median filter — promoted to module-level so load_mission_config_yaml() can override them.
TOF_SPIKE_THRESHOLD_M = 1.5  # max deviation from median before reading is rejected as a spike
TOF_SPIKE_RESET_COUNT = 4    # consecutive spike rejections before assuming genuine step-change

# Feedforward history — promoted to module-level for the same reason.
FF_HISTORY_LEN = 8   # frames of position history for linear-velocity fit (Finals F2)


def load_mission_config_yaml(yaml_path: str) -> None:
    """
    Load tunable mission constants from config.yaml into module-level globals.

    Call this ONCE at startup, BEFORE constructing MissionStateMachine(), so that
    instance variables initialized from these globals pick up the YAML values.

    Falls back to hardcoded defaults silently if the file is absent or a key is
    missing — the drone is always flyable without a config override.

    Safety invariant: DROP_ALTITUDE_MIN is hard-floored at 3.0 m regardless of
    what config.yaml says (competition rule §IV — never lower this floor).
    """
    import yaml as _yaml  # inline import: avoids a hard dependency when not used

    # Maps config.yaml key → module-level global name.
    # Only operator-tunable parameters are listed.
    # READ-ONLY competition-rule constants (ball sequence, match time) are excluded.
    _TUNABLE_MAP: dict = {
        # Detection filters
        "fill_ratio_min_circle":    "FILL_RATIO_MIN_CIRCLE",
        "max_fire_diameter_ratio":  "MAX_FIRE_DIAMETER_RATIO",
        "h_landing_overlap_thresh": "H_LANDING_OVERLAP_THRESH",
        "drop_threshold":           "DROP_THRESHOLD",
        "color_match_threshold":    "COLOR_MATCH_THRESHOLD",
        "edge_clip_margin_px":      "EDGE_CLIP_MARGIN_PX",
        # Altitude gate
        "drop_altitude_min":        "DROP_ALTITUDE_MIN",
        "drop_altitude_target":     "DROP_ALTITUDE_TARGET",
        "drop_altitude_max":        "DROP_ALTITUDE_MAX",
        "default_f_px":             "DEFAULT_F_PX",
        # Attitude gate
        "align_pixel_thresh":       "ALIGN_PIXEL_THRESH",
        "drop_max_pitch_deg":       "DROP_MAX_PITCH_DEG",
        "drop_max_roll_deg":        "DROP_MAX_ROLL_DEG",
        "ff_angular_rate_gate_dps": "FF_ANGULAR_RATE_GATE_DPS",
        # Exclusion zone
        "scored_world_radius_m":    "SCORED_WORLD_RADIUS_M",
        # Velocity gains
        "px_to_ms_fire":            "PX_TO_MS_FIRE",
        "px_to_ms_helipad":         "PX_TO_MS_HELIPAD",
        # Feedforward (Finals F2)
        "ff_scale":                 "FF_SCALE",
        "max_ff_px_s":              "MAX_FF_PX_S",
        "ff_cmd_threshold_ms":      "FF_CMD_THRESHOLD_MS",
        "ff_history_len":           "FF_HISTORY_LEN",
        # Tilt gate
        "tilt_max_relaxed_deg":     "TILT_MAX_RELAXED_DEG",
        # Drop timing
        "drop_cooldown_s":          "DROP_COOLDOWN_S",
        # Match timing (operators need to see these, but changing them is risky)
        "land_early_at_s":          "LAND_EARLY_AT_S",
        "helipad_force_land_s":     "HELIPAD_FORCE_LAND_S",
        # ToF filter
        "tof_spike_threshold_m":    "TOF_SPIKE_THRESHOLD_M",
        "tof_spike_reset_count":    "TOF_SPIKE_RESET_COUNT",
    }

    try:
        with open(yaml_path, "r") as _f:
            cfg = _yaml.safe_load(_f)
    except FileNotFoundError:
        print(f"[CONFIG] WARNING: {yaml_path} not found — using built-in defaults")
        return
    except _yaml.YAMLError as exc:
        print(f"[CONFIG] WARNING: YAML parse error in {yaml_path}: {exc} — using built-in defaults")
        return

    globs = globals()
    loaded: list[str] = []
    for yaml_key, global_name in _TUNABLE_MAP.items():
        if yaml_key in cfg and cfg[yaml_key] is not None:
            old = globs.get(global_name)
            # Preserve the original type (float stays float, int stays int)
            globs[global_name] = type(old)(cfg[yaml_key]) if old is not None else cfg[yaml_key]
            loaded.append(f"  {global_name} = {globs[global_name]}  (was {old})")

    print(f"[CONFIG] Loaded {len(loaded)} constants from {yaml_path}:")
    for _line in loaded:
        print(_line)

    # Safety hard-floor: DROP_ALTITUDE_MIN must never go below 3.0 m.
    # Competition rule §IV: autonomous flight below 3.0 m → −5 pts per 5 s.
    if globs["DROP_ALTITUDE_MIN"] < 3.0:
        print(
            f"[CONFIG] SAFETY OVERRIDE: DROP_ALTITUDE_MIN {globs['DROP_ALTITUDE_MIN']:.3f} m "
            f"< 3.0 m — reset to 3.05 m (competition rule §IV-4)"
        )
        globs["DROP_ALTITUDE_MIN"] = 3.05


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
# Outer-loop PID controller
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

    def update(self, error: float, timestamp: float | None = None,
               freeze_integral: bool = False) -> float:
        """
        Feed one error sample, get one velocity command.

        Args:
            error            : Signed pixel offset (positive = target is right/below).
            timestamp        : Current time in seconds. Uses 1/30 s if None.
            freeze_integral  : If True, skip integral accumulation this frame.
                               P and D terms still run normally so the controller
                               continues to provide corrective output.
                               Use this when the drone is in a WAIT_LEVEL state
                               (attitude gate blocking drop): the drone is hovering
                               over the target with near-zero error, but the
                               integrator would still wind up over the 1.5 s tilt
                               timeout.  Releasing the freeze when attitude clears
                               produces a clean, overshoot-free drop command.

        Returns:
            Velocity command in [-output_limit, +output_limit].
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        dt  = (now - self._prev_time) if self._prev_time is not None else (1.0 / 30.0)
        dt  = max(dt, 1e-4)

        if not freeze_integral:
            self._integral += error * dt
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

    EDGE-CLIP FIX (Threat 1):
    When a fire circle is at the edge of the frame, the visible portion is
    a semicircle or arc. cv2.minEnclosingCircle() fits a circle TIGHTLY around
    that arc, producing a radius much smaller than the real target radius.
    Plugging this into Z = f_px * D_real / (2*r) gives a phantom altitude of
    6-7 m when the drone is actually at 3.5 m — the control law then commands
    a violent dive (cmd_vz = +0.6 m/s) risking a −5 pt altitude penalty and
    potential crash.

    Fix: detect whether the contour's bounding box touches the frame boundary.
    If so, set edge_clipped=True. The caller (_fire_phase) then:
      - Still uses (cx, cy) for XY alignment PID (these remain valid)
      - Falls back to the last known good Z_est instead of computing Z from radius
      - Does NOT allow a DROP trigger (Z is untrustworthy)

    Returns:
        (cx, cy, radius, fill_ratio, cnt_frame, edge_clipped) in frame-pixel
        coordinates, or None if no contour found.
        edge_clipped : True if the circle is partially outside the frame boundary.
    """
    mask_np  = mask_obj.cpu().numpy()
    h_m, w_m = mask_np.shape
    mask_small = (mask_np > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt_small = max(contours, key=cv2.contourArea)

    # ── Fix #6: fit circle in mask space, then scale ─────────────────────────
    # Previous code scaled the contour to frame space BEFORE fitting
    # minEnclosingCircle.  Integer rounding of the scaled points creates heavily
    # aliased (jagged) edges — the fitted radius jumps ±several pixels
    # frame-to-frame even on a static target.  Those noise spikes feed directly
    # into Z_est = f*D/(2r), causing cmd_vz to stutter.
    #
    # Fix: fit on the low-res mask contour (smooth integer points), then scale
    # cx / cy / radius into frame space analytically.  The resulting radius is
    # stable to sub-pixel accuracy across frames.
    sx = frame_w / w_m
    sy = frame_h / h_m

    # ── Fix #6 (updated): fit circle in MASK space for fill ratio only ───────
    # Fill ratio still uses the mask-space radius (no integer aliasing).
    # cx/cy/radius are now computed by a second fit on the FRAME-space float
    # contour (see below) — this eliminates the old (sx+sy)/2 bias.
    (cx_s, cy_s), radius_s = cv2.minEnclosingCircle(cnt_small)

    # ── Fix #9: frame-space radius via float-contour minEnclosingCircle ──────
    # Previous approach:  radius = radius_s * (sx + sy) * 0.5
    # This averages the X and Y scale factors and implicitly assumes the YOLO
    # bounding box is always square (sx ≈ sy).
    #
    # Problem: when the drone banks at 10–15°, the projected circle becomes an
    # ellipse.  YOLO's bounding box stretches on the long axis → sx ≠ sy by up
    # to 15%.  Averaging them biases the radius by ~7-8%, which propagates into
    # Z_est = f·D/(2r): a 7% smaller radius → 7% higher Z_est → a 0.22 m
    # phantom altitude spike at 3.25 m → cmd_vz stutters between 0.0 and 0.6.
    #
    # Fix: scale the contour to frame coordinates as float32 and call
    # minEnclosingCircle directly.  The fitter accounts for the non-square
    # scaling automatically, returning the correct enclosing circle for the
    # scaled shape rather than for an averaged isotropic approximation.
    # CPU cost: fitting N float32 points is O(N) and <0.1 ms — negligible.
    cnt_frame_float = (cnt_small.reshape(-1, 2).astype(np.float32)
                       * np.array([sx, sy], dtype=np.float32)).reshape(-1, 1, 2)
    (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt_frame_float)
    cx     = float(cx_f)
    cy     = float(cy_f)
    radius = float(radius)

    # Integer contour for drawing / edge-clip bbox check.
    cnt_frame = cnt_frame_float.astype(np.int32)

    # ── Edge-clip detection ───────────────────────────────────────────────────
    # If the bounding rect of the (scaled) contour is within EDGE_CLIP_MARGIN_PX
    # pixels of any frame boundary, the circle is partially off-screen and
    # the radius returned by minEnclosingCircle is unreliable for altitude.
    bx, by, bw, bh = cv2.boundingRect(cnt_frame)
    edge_clipped = (
        bx                 <= EDGE_CLIP_MARGIN_PX or
        by                 <= EDGE_CLIP_MARGIN_PX or
        bx + bw >= frame_w - EDGE_CLIP_MARGIN_PX or
        by + bh >= frame_h - EDGE_CLIP_MARGIN_PX
    )

    # Fill ratio computed directly on the small mask using radius_s (mask space).
    # radius_s is accurate (no integer-aliasing) so the fill ratio is stable.
    filled    = int(mask_small.sum())
    full_area = max(1.0, np.pi * radius_s ** 2)
    fr        = min(filled / full_area, 1.0)

    return int(cx), int(cy), int(radius), fr, cnt_frame, edge_clipped


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

        # Scored-circle exclusion zone — prevents re-locking on the same circle.
        # Stored as (world_x_m, world_y_m): the camera-relative offset of the
        # scored centroid in metres at the moment of drop.  Using world space
        # (not pixels) makes the exclusion altitude-invariant: a 0.60 m radius
        # covers the full F1/F2 target regardless of current flight altitude.
        # Applied in both _fire_phase and _recovery_phase.
        self._scored_positions: list = []   # list of (world_x_m, world_y_m)

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

        # Separate PID for helipad approach — slower, more conservative gains.
        # The helipad is larger and slower-moving than fire targets, so we
        # prioritise precision over speed: lower kp prevents overshoot near
        # landing, higher ki eliminates the steady-state hover offset that
        # P-only control leaves behind.
        self._pid_hx = PIDController(kp=0.35, ki=0.03, kd=0.10, output_limit=100.0)
        self._pid_hy = PIDController(kp=0.35, ki=0.03, kd=0.10, output_limit=100.0)

        # Temporal smoothing — EMA with alpha as weight on the OLD (historical) value.
        # alpha=0.0 → always take the raw measurement (no smoothing).
        # alpha=1.0 → never update (full smoothing / frozen).
        # NOTE: this is INVERTED from the textbook "alpha = weight on new sample"
        # convention.  Keep it: lower alpha → more responsive to new measurements.
        # alpha=0.4 for static targets; 0.15 for Finals F2 (moving 10 cm/s).
        if smoothing_alpha == 0.4 and is_finals:
            smoothing_alpha = 0.15
            print("[Mission] Finals mode: EMA alpha lowered to 0.15 (less F2 lag)")
        self._smooth_alpha  = smoothing_alpha
        self._smooth_cx:    float | None = None
        self._smooth_cy:    float | None = None
        self._smooth_ratio: float | None = None

        # Position history for feedforward velocity estimation (F2 tracking).
        # deque(maxlen) auto-discards oldest entries in O(1) vs list.pop(0) O(n).
        self._ff_history_len = FF_HISTORY_LEN
        self._pos_history: deque = deque(maxlen=self._ff_history_len)

        # No-detection fallback
        self._no_detect_frames = 0
        # _max_no_detect: frames before entering RECOVERY / helipad scan.
        # 0.33 s at 30 fps = 10 frames; recalculated in update() when fps_hint changes.
        self._max_no_detect    = 10

        # Recovery / scan state
        self._recovery_frames  = 0
        # _recovery_max: 3 s of scan before giving up on a target.
        self._recovery_max     = 90          # recalculated in update()
        self._scan_yaw_steps   = [30, -60, 30, 0]   # deg/s per step (clockwise sweep)
        # _scan_step_frames: duration of each yaw step.  0.67 s at 30 fps = 20 frames.
        self._scan_step_frames = 20          # recalculated in update()

        # Structured command output — read this in your MAVLink integration
        self.last_command: FlightCommand = FlightCommand(phase="TAKEOFF",
                                                          action="ASCEND")

        # Most recently resolved altitude authority (set in update(), read by phases)
        self._last_alt_authority: float | None = None
        self._last_alt_source: str = "VIS"

        # ── IMU time-sync buffer ──────────────────────────────────────────────
        # Stores (timestamp, SensorData) at IMU poll rate.
        # _get_synced_imu() looks up the reading closest to frame-capture time.
        self._imu_buffer: deque = deque(maxlen=60)   # ~1 s at 60 Hz
        # Expected pipeline latency: camera capture → undistort → YOLO → update()
        # Measure once on your hardware and set accordingly.
        #
        # WHY 0.070 and not 0.045:
        #   Even with Pi 5 + Hailo-8L doing raw inference in 15-20 ms, the full
        #   pipeline adds:
        #     Mask decode + cv2.findContours  : ~15-20 ms
        #     UART / MAVLink round-trip       : ~10-15 ms
        #     Camera ISP + USB/CSI transfer   : ~10 ms
        #   Total realistic end-to-end: 60-80 ms.
        #   Using 45 ms would pull the wrong IMU historical slot, causing
        #   _feedforward_velocity() to over-correct and oscillate on the moving F2.
        self.camera_latency_s: float = 0.070

        # ── ToF median filter ─────────────────────────────────────────────────
        # 5-sample median removes single-frame spikes (TZ gate, helipad edge).
        self._tof_buffer: deque = deque(maxlen=5)
        # Max deviation from median before a reading is treated as an obstacle
        # spike and discarded.  1.5 m covers the TZ gate at 2.6 m when flying
        # at 3.5 m (spike would read ~0.9 m, delta = 2.6 m > 1.5 m → rejected).
        self._tof_spike_threshold_m: float = TOF_SPIKE_THRESHOLD_M
        # Bug 2 fix (ToF Median Filter Lockup): counts how many consecutive
        # frames exceeded the spike threshold.  If the altitude changes
        # legitimately (e.g. drone flies off a box onto the floor), every new
        # reading keeps being rejected and the filter freezes.
        # After _TOF_SPIKE_RESET_COUNT consecutive spikes we assume it is a
        # real step-change, clear the buffer, and let the new value through.
        self._tof_consecutive_spikes: int = 0
        self._TOF_SPIKE_RESET_COUNT:  int = TOF_SPIKE_RESET_COUNT

        # ── Zero-Order-Hold coasting on temporary target loss ─────────────────
        # When target is lost for a few frames, hold the EXACT last-known
        # inertial velocity vector unchanged (Zero-Order Hold).
        #
        # WHY NOT decaying velocity:
        #   A decaying setpoint (×0.75/frame) commands the FC to measure a velocity
        #   error (e.g. 1.0→0.75 m/s = Δ0.25 m/s in one frame) and pitch the nose
        #   back to decelerate — the camera tilts up, target exits the frame,
        #   and the system enters RECOVERY even though the occlusion was only glare.
        #
        # WHY Zero-Order Hold works:
        #   At 30 fps, _max_no_detect ≈ 10 frames = 0.33 s of glare/occlusion.
        #   Flying flat and level at constant velocity for 0.33 s keeps the camera
        #   stable so YOLO can reacquire without camera-induced perturbation.
        self._coast_vx: float = 0.0
        self._coast_vy: float = 0.0
        # Fix #4b (ZOH "Handbrake" — altitude axis):
        # Storing the last cmd_vz prevents the Zero-Order-Hold from issuing a
        # sudden vz=0.0 command when the target is briefly lost.  Without this,
        # the FC interprets the step change as "brake altitude immediately", which
        # throttles up, tilts the drone, and turns a 10-frame glare event into a
        # full RECOVERY cycle.  Initialised to 0.0 (altitude hold) as the safe
        # default before any detection occurs.
        self._coast_vz: float = 0.0

        # Last known reliable vision-based altitude estimate.
        # Updated only when _extract_geometry returns a NON-edge-clipped circle.
        # Used as fallback Z_est when the circle is partially off-frame.
        self._last_z_est: float | None = None

        # ── Fix #4: wind-compensation deadlock counter ────────────────────────
        # If we are aligned + altitude-safe but the attitude gate keeps blocking
        # the drop (FC must tilt to hold position in crosswind), increment this
        # counter.  After TILT_RELAX_AFTER_S the gate relaxes to _TILT_MAX_RELAXED.
        # Hard cap at 15°: better to drop with slight miss than never drop at all.
        self._tilt_block_frames: int  = 0
        # _TILT_RELAX_FRAMES is set by _refresh_fps_thresholds() on first update().
        self._TILT_MAX_RELAXED: float = TILT_MAX_RELAXED_DEG

        # ── Fix #7: dead-reckoning position for geofencing ────────────────────
        # Integrate optical-flow velocity to estimate drone displacement from the
        # takeoff point.  Not globally accurate (gyro drift, no GPS) but sufficient
        # to detect a rogue flight outside the 24×12 m field boundaries.
        # Reset to (0, 0) on every mission.reset().
        self._est_pos_x_m: float = 0.0   # forward   (body X integrated)
        self._est_pos_y_m: float = 0.0   # lateral   (body Y integrated)
        self._geofence_violation_frames: int = 0
        # Previous-frame yaw for Δψ scored-position rotation (Bug 1 fix).
        # None until the first frame with BNO080 yaw data.
        self._prev_yaw_deg: float | None = None
        # Bug 1 fix (Magnetic Compass Misalignment): absolute magnetic heading at
        # the moment the drone first becomes active (first SEARCHING frame).
        # All geofence body→world rotations use (yaw_deg − _yaw_0) so that the
        # world X-axis always points in the drone's initial nose direction,
        # regardless of how the arena is oriented relative to magnetic north.
        self._yaw_0: float | None = None
        # Wall-clock time of the last update() call for real-dt odometry (Bug 2 fix).
        # None on first frame; set at the top of every subsequent update() call.
        self._last_frame_time: float | None = None
        # Trigger geofence return after 1 s of continuous out-of-bounds reading.
        # Short enough to prevent collision with netting; long enough to ignore
        # momentary GPS/flow glitches at the field boundary.
        # _GEOFENCE_WARN_FRAMES is set by _refresh_fps_thresholds() on first update().
        # Soft boundary: 1 m inside each field edge.
        #
        # Bug 1 fix (Invisible Wall Mid-Field):
        # The old code used a SYMMETRIC ±11 m limit on the X axis, which implicitly
        # assumes the odometry origin (0,0) is at the CENTRE of the 24 m field.
        # In reality, the drone takes off from the Safe Zone at one END of the field.
        # The odometry origin IS the takeoff point.  The full field extends forward
        # from 0 m to 24 m.
        #
        # With the symmetric ±11 m limit, the geofence fired "VIOLATION!" the moment
        # the drone passed the halfway point (~11 m), yanking it back before it could
        # ever reach F3, F4, or F5 which are in the second half of the arena.
        #
        # Correct asymmetric limits (1 m buffer from each physical net):
        #   X_MIN = -1.0 m  (allow up to 1 m behind the start line)
        #   X_MAX = +23.0 m (stop 1 m before the far net at 24 m)
        # Y axis remains symmetric because takeoff is centred on the 12 m width.
        self._GEOFENCE_X_MIN: float = -1.0              # 1 m behind start (backward)
        self._GEOFENCE_X_MAX: float = FIELD_LENGTH - 1.0  # 1 m before far net = 23.0 m
        self._GEOFENCE_Y_M:   float = FIELD_WIDTH / 2.0 - 1.0   # ±5 m lateral

        # ── Match timer ───────────────────────────────────────────────────────
        # Rulebook §XI: 5 minutes total.  Tasks after 300 s are not scored.
        # Force FIND_HELIPAD 15 s before expiry to secure the landing bonus.
        self._start_time: float | None = None
        self.time_elapsed_s: float = 0.0
        self._land_early_at_s: float = LAND_EARLY_AT_S
        self._timer_forced_land: bool = False   # latched once to avoid repeated prints

    # ── Properties ───────────────────────────────────────────────────────────

    # ── fps_hint public property ──────────────────────────────────────────────
    # The README and test_video.py say `mission.fps_hint = cap.get(FPS)`.
    # The backing store is _fps_hint (private).  Without a property, that
    # assignment silently creates a NEW public attribute while _fps_hint stays
    # frozen at 30 — all frame-count thresholds would be wrong.
    @property
    def fps_hint(self) -> float:
        return self._fps_hint

    @fps_hint.setter
    def fps_hint(self, value: float) -> None:
        self._fps_hint = float(max(1.0, value))
        self._refresh_fps_thresholds()

    @property
    def current_target(self):
        """Return (color_name, label, points, real_diameter_m) for current fire target."""
        if self.current_target_idx < len(self._fire_sequence):
            return self._fire_sequence[self.current_target_idx]
        return None

    @property
    def mission_complete(self):
        return self.phase == Phase.COMPLETE

    def _get_filtered_tof(self, raw_tof: float | None) -> float | None:
        """
        ToF median filter + spike rejection.

        The MTF-01 ToF laser fires a single point straight down.  Flying over a
        TZ gate (2.6 m tall) when the drone is at 3.5 m produces a sudden ~0.9 m
        reading — a delta of 2.6 m from the true floor distance.  Without
        filtering this trips the altitude-penalty check and commands a violent
        ascent that ruins the drop sequence.

        Strategy:
          1. If the buffer has ≥ 3 readings, compute the running median.
          2. Reject any incoming reading that deviates from the median by more
             than _tof_spike_threshold_m (default 1.5 m) — return None so the
             caller falls through to the vision Z_est fallback.  The spike is
             NOT added to the buffer so it cannot corrupt future medians.
          3. Accept the reading, append it, and return the updated median.
        """
        if raw_tof is None:
            return None

        # Check for spike against current buffer BEFORE appending.
        # This avoids the old append-then-pop anti-pattern and keeps the
        # buffer clean: a rejected spike never enters _tof_buffer.
        if len(self._tof_buffer) >= 3:
            current_median = float(np.median(list(self._tof_buffer)))
            if abs(raw_tof - current_median) > self._tof_spike_threshold_m:
                self._tof_consecutive_spikes += 1
                # Bug 2 fix (Step-Response Lockup):
                # If the buffer has been rejecting every reading for
                # _TOF_SPIKE_RESET_COUNT consecutive frames, this is NOT a
                # transient spike (e.g. TZ gate) — it is a genuine terrain
                # step-change (e.g. drone flew off a startup box onto the floor,
                # or the competition surface dropped).  Clearing the buffer
                # allows the filter to re-converge on the new true altitude.
                # Without this, the filter freezes permanently at the old median.
                if self._tof_consecutive_spikes >= self._TOF_SPIKE_RESET_COUNT:
                    self._tof_buffer.clear()
                    self._tof_consecutive_spikes = 0
                    print(f"[TOF] Step-change detected — buffer cleared, "
                          f"re-seeding with {raw_tof:.2f}m")
                    # Fall through to append and return below
                else:
                    return None   # obstacle spike — discard without contaminating buffer
        else:
            self._tof_consecutive_spikes = 0   # reset on a clean (unfiltered) reading

        self._tof_buffer.append(raw_tof)
        self._tof_consecutive_spikes = 0   # reading accepted — clear streak counter

        if len(self._tof_buffer) < 3:
            return raw_tof   # not enough history yet, pass raw value through

        return float(np.median(list(self._tof_buffer)))

    def _already_scored(self, cx: int, cy: int, Z_est: float,
                         frame_cx: int, frame_cy: int) -> bool:
        """
        Return True if (cx, cy) is within SCORED_WORLD_RADIUS_M of any
        previously scored circle.

        Converts the pixel offset from frame centre to world-space metres using
        the pinhole equation (world_x = dx_px * Z / f_px), then compares the
        Euclidean world distance against SCORED_WORLD_RADIUS_M.

        WHY world space, not pixels:
          At 3.0 m altitude 80 px ≈ 0.48 m; at 4.5 m the same 80 px ≈ 0.72 m.
          A pixel-based radius oscillates ±25% across the drop window.
          A metre-based radius is constant regardless of altitude.

        LIMITATION: this does not account for drone lateral translation between
        the score event and the current frame.  At typical hover accuracy
        (< 0.3 m drift) the 0.60 m radius still covers the full F1/F2 target.
        """
        if not self._scored_positions:
            return False
        f = self.alt_est.f_px
        Z = max(Z_est, 0.5)
        curr_wx = (cx - frame_cx) * Z / f
        curr_wy = (cy - frame_cy) * Z / f
        R2 = SCORED_WORLD_RADIUS_M ** 2
        return any(
            (curr_wx - sx) ** 2 + (curr_wy - sy) ** 2 < R2
            for sx, sy in self._scored_positions
        )

    def _refresh_fps_thresholds(self) -> None:
        """
        Recompute all frame-count thresholds from _fps_hint.

        Called at the start of update() so that any change to _fps_hint
        (e.g. set by test_video.py after reading cap.get(FPS)) takes effect
        immediately without requiring a restart.

        Time targets:
          _max_no_detect    : 0.33 s  → 10 fr @ 30 fps, 7 fr @ 20 fps
          _recovery_max     : 3.0  s  → 90 fr @ 30 fps, 60 fr @ 20 fps
          _scan_step_frames : 0.67 s  → 20 fr @ 30 fps, 13 fr @ 20 fps
        """
        fps = max(self._fps_hint, 1.0)
        self._max_no_detect        = max(3,  int(0.33 * fps))
        self._recovery_max         = max(10, int(3.00 * fps))
        self._scan_step_frames     = max(5,  int(0.67 * fps))
        # Bug fix: these two were computed once in __init__ at fps=30 and never
        # updated.  A user who sets mission.fps_hint = 20 would get a tilt-relax
        # timer of 45 frames (2.25 s) instead of 30 frames (1.5 s), and a
        # geofence trigger after 30 frames (1.5 s) instead of 20 frames (1.0 s).
        self._TILT_RELAX_FRAMES    = max(5,  int(1.5 * fps))   # 1.5 s
        self._GEOFENCE_WARN_FRAMES = max(3,  int(1.0 * fps))   # 1.0 s

    # ── Temporal smoothing helper ─────────────────────────────────────────────

    def _smooth(self, cx: int, cy: int, Z_drop: float) -> tuple:
        """
        Exponential moving average on target centre position and altitude.

        Alpha weights the OLD (historical) value — see __init__ for the
        convention note.  Also appends to _pos_history for feedforward.

        Returns:
            (smoothed_cx, smoothed_cy, smoothed_Z_drop) as (int, int, float).
        """
        a = self._smooth_alpha
        if self._smooth_cx is None:
            self._smooth_cx    = float(cx)
            self._smooth_cy    = float(cy)
            self._smooth_ratio = Z_drop
        else:
            self._smooth_cx    = a * self._smooth_cx    + (1 - a) * cx
            self._smooth_cy    = a * self._smooth_cy    + (1 - a) * cy
            self._smooth_ratio = a * self._smooth_ratio + (1 - a) * Z_drop

        # Record smoothed position for feedforward estimation
        self._pos_history.append(
            (self._smooth_cx, self._smooth_cy, time.perf_counter())
        )

        return int(self._smooth_cx), int(self._smooth_cy), self._smooth_ratio

    def push_sensor_data(self, sd: "SensorData") -> None:
        """
        Buffer a SensorData reading with a timestamp.

        Call this at IMU polling rate (up to 400 Hz for BNO080) in a background
        thread or tight loop.  update() will then fetch the reading whose
        timestamp is closest to the frame capture time (≈ now - camera_latency_s).

        This resolves the time-sync asymmetry: camera frames have 30-60 ms
        latency (capture → undistort → YOLO → update()), while IMU data is
        near-realtime.  Without buffering, we would be compensating a 30-60ms-old
        frame with the current (too-fresh) IMU reading, over-correcting any fast
        manoeuvres (e.g. braking produces 2+ extra degrees of tilt vs the frame).
        """
        self._imu_buffer.append((time.perf_counter(), sd))

    def _get_synced_imu(self, sensor_data: "SensorData | None",
                        frame_timestamp: float | None = None) -> "SensorData | None":
        """
        Return the buffered SensorData reading whose timestamp best matches the
        frame capture time.

        Priority for target lookup time:
          1. frame_timestamp  — caller-supplied timestamp from time.perf_counter()
                                at the moment the camera frame was *grabbed*.
                                Use this whenever possible: it is exact and immune
                                to pipeline-latency jitter (thermal throttling,
                                variable YOLO inference time, etc.).
          2. time.perf_counter() − camera_latency_s  — static fallback when no
                                frame timestamp is available.  Accurate only if
                                the true pipeline latency matches camera_latency_s.

        Fix #2 (Dynamic Camera Latency):
        The hardcoded camera_latency_s = 0.070 assumes a stable 70 ms pipeline.
        Under Raspberry Pi 5 thermal throttling the pipeline can stretch to 100 ms+
        or collapse to 40 ms.  Fetching the IMU reading at the wrong historical
        slot causes _feedforward_velocity() to over-correct the F2 intercept and
        oscillate.  Attaching a frame timestamp at grab time and passing it here
        makes the sync exact regardless of pipeline variability.

        Usage in test_video.py:
            frame_ts = time.perf_counter()
            ret, frame = cap.read()               # or frame = cam.grab()
            results = model(frame, verbose=False)[0]
            frame = mission.update(frame, results,
                                   current_altitude=alt,
                                   sensor_data=sd,
                                   frame_timestamp=frame_ts)

        Falls back to `sensor_data` (the caller-supplied current reading) when:
          - The buffer is empty (push_sensor_data() was never called), or
          - sensor_data itself is None (sensor not connected)
        """
        if sensor_data is None:
            return None
        if not self._imu_buffer:
            return sensor_data   # no buffer yet — use current reading as-is
        target_t = (frame_timestamp
                    if frame_timestamp is not None
                    else time.perf_counter() - self.camera_latency_s)
        # Linear search is fine: buffer is at most ~60 entries (1 s at 60 Hz)
        synced = min(self._imu_buffer, key=lambda x: abs(x[0] - target_t))
        return synced[1]

    def _feedforward_velocity(self) -> tuple[float, float]:
        """
        Estimate TARGET pixel velocity from the smoothed, drone-motion-corrected
        position history using least-squares linear regression.

        Drone-motion correction is now applied INCREMENTALLY in update() every
        frame by shifting all existing _pos_history entries by one frame's worth
        of optical-flow pixels (Bug 3 fix — Phantom Velocity).  This means the
        history already contains only pure TARGET motion by the time this function
        runs, so a simple polyfit suffices with no additional correction here.

        Returns:
            (vx_px_per_s, vy_px_per_s) — predicted target velocity in image-pixel
            coordinates, clamped to ±MAX_FF_PX_S.
        """
        if len(self._pos_history) < 4:
            return 0.0, 0.0

        xs  = np.array([p[0] for p in self._pos_history])
        ys  = np.array([p[1] for p in self._pos_history])
        ts  = np.array([p[2] for p in self._pos_history])

        # Bug 3 fix (Phantom Velocity):
        # Drone-motion drift has already been subtracted from each entry in
        # _pos_history incrementally in update() (one real_dt step per frame),
        # so no retroactive correction is needed or correct here.
        # Running polyfit directly on the corrected history gives pure target
        # velocity without the constant-velocity assumption artifact.

        # Normalise timestamps and fit
        ts_n = ts - ts[0]
        if ts_n[-1] < 1e-4:
            return 0.0, 0.0

        # Bug 4 fix (Unweighted Lag):
        # An unweighted polyfit treats a sample from 8 frames ago (~0.27 s at
        # 30 fps) identically to the current frame.  When the 10 cm/s F2 target
        # curves or stops, the stale samples bias the slope estimate and the
        # feedforward continues driving toward the OLD trajectory for another
        # 0.25 s — causing the drone to overshoot and drop too late.
        #
        # Fix: linearly increasing weights 0.2 → 1.0.  The oldest sample gets
        # weight 0.2 (5× less influence than the current frame).  The velocity
        # estimate converges ~3× faster to trajectory changes while remaining
        # smooth enough to resist single-frame outliers.
        weights = np.linspace(0.2, 1.0, len(ts_n))
        vx_px = float(np.polyfit(ts_n, xs, 1, w=weights)[0])
        vy_px = float(np.polyfit(ts_n, ys, 1, w=weights)[0])

        return (float(np.clip(vx_px, -MAX_FF_PX_S, MAX_FF_PX_S)),
                float(np.clip(vy_px, -MAX_FF_PX_S, MAX_FF_PX_S)))

    # ── Main per-frame update ─────────────────────────────────────────────────

    def update(self, frame: np.ndarray, yolo_result,
               current_altitude: float | None = None,
               sensor_data: "SensorData | None" = None,
               frame_timestamp: float | None = None) -> np.ndarray:
        """
        Process one YOLO result and update mission state.

        Args:
            frame            : BGR frame (annotated in-place).
            yolo_result      : Single Ultralytics Results object.
            current_altitude : Altitude in metres from FC barometer/rangefinder
                               via MAVLink.  Highest priority if provided.
            sensor_data      : SensorData from MTF-01 + BNO080.  Used for:
                                 - tof_altitude_m  : fallback altitude when FC
                                                     data unavailable
                                 - of_vx/vy_ms     : optical flow for feedforward
                                                     target-motion isolation
                                 - pitch/roll_deg  : tilt-corrected Z_est,
                                                     level-attitude drop gate
                                 - pitch/roll_rate : IMU-based feedforward guard
            frame_timestamp  : time.perf_counter() value recorded at the moment
                               the camera frame was grabbed (BEFORE YOLO inference).
                               Enables exact IMU time-sync without relying on the
                               hardcoded camera_latency_s fallback.  Pass this
                               whenever possible — see _get_synced_imu() for details.

        After calling, read mission.last_command for the structured FlightCommand.

        Altitude priority (highest → lowest):
          1. current_altitude  (FC barometer / rangefinder via MAVLink)
          2. sensor_data.tof_altitude_m  (MTF-01 ToF laser)
          3. Z_est  (pinhole vision estimate — tilt-corrected if BNO080 available)
        """
        h, w = frame.shape[:2]

        # Recompute frame-count thresholds in case _fps_hint was updated externally.
        self._refresh_fps_thresholds()

        # ── Real inter-frame Δt for odometry (Bug 2 fix) ─────────────────────
        # The old code used 1/fps_hint as Δt everywhere.  Under Raspberry Pi
        # thermal throttling, real FPS can halve (30 → 15) while fps_hint stays
        # at 30 — causing position estimates to accumulate at half the true rate.
        # A drone flying 2 m into the boundary only registers 1 m; the geofence
        # stays silent.
        #
        # Fix: capture the actual wall-clock time at the top of update() and
        # compute the TRUE elapsed time since the previous frame.
        # Clamped to [1 ms, 500 ms] to survive debugger breakpoints or
        # a one-off camera stutter without corrupting the integrator.
        _now = time.perf_counter()
        _real_dt = ((_now - self._last_frame_time)
                    if self._last_frame_time is not None
                    else (1.0 / max(self._fps_hint, 1.0)))
        _real_dt = float(np.clip(_real_dt, 1e-3, 0.5))
        self._last_frame_time = _now

        # ── Match timer ───────────────────────────────────────────────────────
        # Start the clock when the drone first enters SEARCHING (confirmed
        # takeoff / skip_takeoff). NOT during TAKEOFF — the system may be
        # running for seconds during pre-flight checks before liftoff.
        if (self._start_time is None
                and self.phase not in (Phase.TAKEOFF, Phase.COMPLETE)):
            self._start_time = time.perf_counter()
        if self._start_time is not None:
            self.time_elapsed_s = time.perf_counter() - self._start_time
            # Forced helipad transition at _land_early_at_s (285 s default).
            # Securing the 30-pt landing bonus is always better than gambling
            # on completing one more ball drop in the last 15 seconds.
            if (not self._timer_forced_land
                    and self.time_elapsed_s >= self._land_early_at_s
                    and self.phase not in (Phase.FIND_HELIPAD, Phase.LANDING,
                                           Phase.COMPLETE)):
                remaining = MATCH_TIME_S - self.time_elapsed_s
                print(f"[TIMER] {self.time_elapsed_s:.0f}s elapsed "
                      f"({remaining:.0f}s remaining) — forcing FIND_HELIPAD "
                      f"to secure landing bonus")
                self._timer_forced_land = True
                self._transition_to_helipad()
                # Still fall through to draw HUD (helipad phase handles itself)

        # ── Sync sensor_data to frame capture timestamp ───────────────────────
        # Fetch the buffered IMU reading whose timestamp ≈ when this frame was
        # captured (frame_timestamp if provided, else now - camera_latency_s).
        # Falls back to current sensor_data if push_sensor_data() was never called.
        synced_sd = self._get_synced_imu(sensor_data, frame_timestamp)

        # ── Apply ToF median filter ───────────────────────────────────────────
        # Replace the raw ToF reading with the filtered value before it enters
        # the altitude priority chain.  Returns None on spike (obstacle).
        if synced_sd is not None and synced_sd.tof_altitude_m is not None:
            filtered_tof = self._get_filtered_tof(synced_sd.tof_altitude_m)
            # Rebuild SensorData with filtered ToF so all downstream code
            # uses the clean value automatically.
            if filtered_tof != synced_sd.tof_altitude_m:
                synced_sd = dataclasses.replace(synced_sd, tof_altitude_m=filtered_tof)

        # ── Resolve authoritative altitude from priority chain ────────────────
        if current_altitude is not None:
            _alt_authority = current_altitude
            _alt_source    = "FC"
        elif synced_sd is not None and synced_sd.tof_altitude_m is not None:
            _alt_authority = synced_sd.tof_altitude_m
            _alt_source    = "TOF"
        else:
            _alt_authority = None
            _alt_source    = "VIS"
        self._last_alt_authority = _alt_authority
        self._last_alt_source    = _alt_source

        if self._drop_cooldown > 0:
            self._drop_cooldown -= 1

        # ── Fix #1 + Yaw-rotation: drift-correct scored positions ────────────
        # _scored_positions are stored in the CAMERA-HEADING-RELATIVE frame:
        #   World X = right of camera,  World Y = forward of camera.
        #
        # Two independent corrections must be applied every frame:
        #
        #  A) TRANSLATION (already present): as the drone translates, stored
        #     world offsets become stale.  Subtract the optical-flow contribution
        #     to keep each point at its true camera-relative position.
        #       Body lateral (+vy) → objects shift LEFT    → wx decreases
        #       Body forward (+vx) → objects shift TOWARD  → wy decreases
        #
        #  B) ROTATION (Bug 1 fix): as the drone yaws by Δψ, the camera-heading
        #     axis rotates.  Every stored point must be rotated by +Δψ so it
        #     stays at the correct physical position in the NEW camera frame.
        #     Without this, recovery yaw sweeps cause all exclusion zones to
        #     "spin away" from their real locations — F1 slips out of the 0.6 m
        #     exclusion radius and the drone re-locks and drops a second ball.
        #
        #     Formula (positive rotation by Δψ, where Δψ = yaw_now − yaw_prev):
        #       wx_new = wx·cos(Δψ) − wy·sin(Δψ)
        #       wy_new = wx·sin(Δψ) + wy·cos(Δψ)
        #
        # Both corrections use _real_dt (Bug 2 fix) so they remain accurate
        # under CPU thermal throttling.
        #
        # Round 6 fix (Bug 1 — Interpolation Memory Vertigo):
        # Δψ is pre-computed ONCE here (before _prev_yaw_deg is updated) and
        # shared by BOTH _scored_positions (world-space) and _pos_history
        # (pixel-space).  The old code embedded the Δψ computation inside the
        # _scored_positions block, so _prev_yaw_deg was already stale by the
        # time the _pos_history block ran — giving it a delta of 0 every frame
        # and silently skipping the rotation entirely.

        # ── Pre-compute this frame's Δψ rotation coefficients ────────────────
        # MUST happen before _prev_yaw_deg is updated.  Both correction blocks
        # below consume _frame_cd / _frame_sd directly.
        _frame_yaw_now = (synced_sd.yaw_deg
                          if synced_sd is not None and synced_sd.yaw_deg is not None
                          else None)
        _frame_has_rotation = False
        _frame_cd = 1.0   # cos(Δψ) — identity when there is no rotation
        _frame_sd = 0.0   # sin(Δψ)
        if _frame_yaw_now is not None and self._prev_yaw_deg is not None:
            _frame_d_psi = ((_frame_yaw_now - self._prev_yaw_deg + 180.0) % 360.0) - 180.0
            if abs(_frame_d_psi) > 0.05:   # skip sub-0.05° rounding noise
                _frame_has_rotation = True
                _frame_cd = float(np.cos(np.deg2rad(_frame_d_psi)))
                _frame_sd = float(np.sin(np.deg2rad(_frame_d_psi)))

        if self._scored_positions:
            # A) Translation correction — needs optical flow
            if (synced_sd is not None
                    and synced_sd.of_vx_ms is not None
                    and synced_sd.of_vy_ms is not None):
                # Fix J3 — "Chasing Shadows" (Y-axis sign error):
                # Camera: TOP = nose, BOTTOM = tail. Forward motion (+vx) pushes scored
                # targets toward the BOTTOM (larger cy → larger world_y). So _drift_wy
                # must be POSITIVE. The old minus sign dragged exclusion zones forward
                # with the drone — F1's shadow followed it to F2 every run.
                # Lateral axis (wx) sign is physically correct, unchanged.
                _drift_wx = -(synced_sd.of_vy_ms * _real_dt)   # lateral → world X ✓
                _drift_wy = +(synced_sd.of_vx_ms * _real_dt)   # forward → world Y (fixed: was −)
                self._scored_positions = [
                    (wx + _drift_wx, wy + _drift_wy)
                    for wx, wy in self._scored_positions
                ]

            # B) Yaw rotation correction — use pre-computed frame Δψ
            if _frame_has_rotation:
                self._scored_positions = [
                    (wx * _frame_cd - wy * _frame_sd,
                     wx * _frame_sd + wy * _frame_cd)
                    for wx, wy in self._scored_positions
                ]

        # Update _prev_yaw_deg even when _scored_positions is empty so the
        # first ball-drop frame has a correct Δψ baseline.
        if _frame_yaw_now is not None:
            self._prev_yaw_deg = _frame_yaw_now

        # ── Bug 1 fix: latch takeoff heading reference (yaw_0) ───────────────
        # Save the absolute magnetic heading the FIRST time the drone becomes
        # active (i.e. exits TAKEOFF).  All subsequent geofence world-frame
        # rotations use (yaw_deg − _yaw_0) so that world-X always aligns with
        # the drone's initial nose direction, regardless of where Magnetic North
        # is relative to the arena.  Works for both the normal takeoff path and
        # skip_takeoff() (which bypasses _takeoff_phase entirely).
        if (self._yaw_0 is None
                and self.phase not in (Phase.TAKEOFF, Phase.COMPLETE)
                and synced_sd is not None
                and synced_sd.yaw_deg is not None):
            self._yaw_0 = synced_sd.yaw_deg
            print(f"[GEOFENCE] yaw_0 latched = {self._yaw_0:.1f}° "
                  f"(arena X-axis = magnetic {self._yaw_0:.1f}°)")

        # ── Bug 3 fix: real-time drift correction for _pos_history ──────────
        # Problem with the old approach: _feedforward_velocity() applied the
        # CURRENT optical-flow velocity retroactively across ALL past timestamps.
        # This assumed constant velocity — catastrophically wrong when the PID is
        # accelerating the drone, generating a "phantom" target velocity in the
        # opposite direction that made the prediction wildly unstable.
        #
        # Fix: apply the EXACT SAME per-frame pixel drift that was computed for
        # _scored_positions (but expressed in pixel space) to every existing entry
        # in _pos_history RIGHT NOW, every frame.  Each historical point is shifted
        # by exactly one frame's worth of drone-motion pixels, so that by the time
        # _feedforward_velocity() runs its polyfit the drone translation has already
        # been subtracted incrementally and accurately — no retroactive assumption.
        if (self._pos_history
                and synced_sd is not None
                and synced_sd.of_vx_ms is not None
                and synced_sd.of_vy_ms is not None):
            _Z_ref   = max(self._smooth_ratio or DROP_ALTITUDE_TARGET, 0.5)
            _px_per_m = self.alt_est.f_px / _Z_ref          # px / m
            # Same body→image sign convention as _feedforward_velocity:
            #   lateral right  (+of_vy) → image X shifts LEFT  (−px)
            #   forward motion (+of_vx) → image Y shifts DOWN  (+py)
            _dpx = -synced_sd.of_vy_ms * _px_per_m * _real_dt
            _dpy = +synced_sd.of_vx_ms * _px_per_m * _real_dt
            self._pos_history = deque(
                ((x + _dpx, y + _dpy, t) for x, y, t in self._pos_history),
                maxlen=self._ff_history_len
            )
        # ── Round 6 Fix 1B: Rotate _pos_history by Δψ on yaw change ────────────
        # _pos_history stores pixel coordinates (cx, cy) in the camera frame.
        # When the drone yaws by Δψ, the camera rotates and each stored pixel
        # position must be rotated about the IMAGE CENTRE (w/2, h/2) — not the
        # world origin — to stay at the same physical world location in the new
        # camera heading.  Without this, np.polyfit sees a sudden coordinate jump
        # (equal to the full rotation of the circle's position) and interprets it
        # as the target moving at ~1000 px/s.  _pred_offset explodes and the PID
        # sends a maximum velocity command the instant the drone executes any yaw.
        #
        # Rotation formula (positive Δψ, rotating about image centre):
        #   dx = cx − w/2,  dy = cy − h/2
        #   cx_new = (w/2) + dx·cos(Δψ) − dy·sin(Δψ)
        #   cy_new = (h/2) + dx·sin(Δψ) + dy·cos(Δψ)
        #
        # Uses the same _frame_cd / _frame_sd pre-computed above so both
        # world-space (_scored_positions) and pixel-space (_pos_history) see
        # exactly the same rotation angle for this frame.
        if self._pos_history and _frame_has_rotation:
            _cx_img = w / 2.0
            _cy_img = h / 2.0
            # PHYSICAL SIGN CONVENTION (Bug 1 fix — Inverse Rotation):
            # The drone yaws +Δψ clockwise  →  the camera rotates +Δψ clockwise.
            # Static objects therefore appear to shift -Δψ (counter-clockwise) in
            # the image.  To keep the stored history at the correct world position
            # in the NEW camera frame, the pixel coordinates must be rotated by
            # -Δψ, NOT +Δψ.
            #
            # Rotation by -Δψ uses:
            #   cos(-Δψ) =  cos(Δψ) = _frame_cd   (unchanged)
            #   sin(-Δψ) = -sin(Δψ) = -_frame_sd
            #
            # Full formula rotating (x, y) about image centre by -Δψ:
            #   x_new = cx_img + (x - cx_img)·cos(Δψ) + (y - cy_img)·sin(Δψ)
            #   y_new = cy_img − (x - cx_img)·sin(Δψ) + (y - cy_img)·cos(Δψ)
            #
            # Compare with the OLD (wrong) +Δψ formula:
            #   x_new = cx_img + dx·_frame_cd − dy·_frame_sd   ← minus was wrong
            #   y_new = cy_img + dx·_frame_sd + dy·_frame_cd   ← plus  was wrong
            self._pos_history = deque(
                (
                    (
                        _cx_img + (x - _cx_img) * _frame_cd + (y - _cy_img) * _frame_sd,
                        _cy_img - (x - _cx_img) * _frame_sd + (y - _cy_img) * _frame_cd,
                        t
                    )
                    for x, y, t in self._pos_history
                ),
                self._ff_history_len
            )

        # Accumulate optical-flow velocity into a scalar displacement estimate.
        # Used ONLY for the soft geofence below — not for world-space exclusion
        # (that is already handled by the scored-position drift above).
        #
        # BUG FIX (Geofence Frame Disaster): optical-flow velocities are in the
        # BODY frame (of_vx = nose, of_vy = right-wing).  Blindly summing them
        # into (_est_pos_x_m, _est_pos_y_m) — which is a WORLD-frame accumulator
        # — ignores yaw completely.  After a 90° yaw manoeuvre the body X-axis
        # points in the world Y direction, so the estimate drifts 90° off and
        # the geofence fires false alarms (or stays silent at the real boundary).
        #
        # Fix: apply a 2-D rotation by the current yaw angle (BNO080 yaw_deg)
        # BEFORE integrating.  If yaw is unavailable, fall back to raw body-frame
        # values — less accurate but no worse than the old code.
        if (synced_sd is not None
                and synced_sd.of_vx_ms is not None
                and synced_sd.of_vy_ms is not None):
            # Use _real_dt (wall-clock) instead of 1/fps_hint — Bug 2 fix.
            if synced_sd.yaw_deg is not None:
                # Bug 1 fix: subtract _yaw_0 so world-X = drone's initial nose
                # direction, not Magnetic North.  If _yaw_0 is not yet set,
                # fall back to 0° (absolute heading) as a safe default.
                _yaw_rel  = synced_sd.yaw_deg - (self._yaw_0 or 0.0)
                _yaw_rad  = np.deg2rad(_yaw_rel)
                _cos_y    = float(np.cos(_yaw_rad))
                _sin_y    = float(np.sin(_yaw_rad))
                # Standard 2-D rotation: body → world
                #   world_x =  cos(ψ)·vx_body − sin(ψ)·vy_body
                #   world_y =  sin(ψ)·vx_body + cos(ψ)·vy_body
                _world_vx = _cos_y * synced_sd.of_vx_ms - _sin_y * synced_sd.of_vy_ms
                _world_vy = _sin_y * synced_sd.of_vx_ms + _cos_y * synced_sd.of_vy_ms
            else:
                # Yaw unavailable — use raw body-frame as best-effort fallback
                _world_vx = synced_sd.of_vx_ms
                _world_vy = synced_sd.of_vy_ms
            self._est_pos_x_m += _world_vx * _real_dt
            self._est_pos_y_m += _world_vy * _real_dt

        # ── Altitude penalty tracking (§VII — <3.0m >5s in auto = −5pts) ────
        # Use resolved altitude authority for the penalty check.
        _penalty_alt = _alt_authority   # None if vision-only
        if (self.phase not in (Phase.TAKEOFF, Phase.COMPLETE)
                and _penalty_alt is not None
                and _penalty_alt < 3.0):
            self._low_alt_frames += 1
            penalty_frame_thresh = max(1, int(ALT_PENALTY_TIME_S * self._fps_hint))
            if self._low_alt_frames % penalty_frame_thresh == 0:
                self.penalties += PENALTY_ALTITUDE
                print(f"[PENALTY] Altitude {_penalty_alt:.2f}m < 3.0m for "
                      f"{ALT_PENALTY_TIME_S}s → {PENALTY_ALTITUDE} pts  "
                      f"(total penalties: {self.penalties})")
        else:
            self._low_alt_frames = 0

        # ── Phase routing ─────────────────────────────────────────────────────
        if self.phase == Phase.TAKEOFF:
            frame = self._takeoff_phase(frame, _alt_authority, w, h)

        elif self.phase in (Phase.SEARCHING, Phase.TARGETING, Phase.DROPPING):
            frame = self._fire_phase(frame, yolo_result, w, h,
                                     _alt_authority, _alt_source, synced_sd)

        elif self.phase == Phase.RECOVERY:
            frame = self._recovery_phase(frame, yolo_result, w, h)

        elif self.phase in (Phase.FIND_HELIPAD, Phase.LANDING):
            frame = self._helipad_phase(frame, yolo_result, w, h, synced_sd)

        # ── Fix #7: software geofence override ───────────────────────────────
        # Apply AFTER all phase logic so this overrides any command that would
        # fly the drone outside the 24×12 m field boundary.
        #
        # Trigger condition: optical-flow dead-reckoning puts the drone outside
        # the soft boundary (1 m inside field edge) for more than 1 second.
        # This handles the critical failure mode of a false positive outside the
        # arena (e.g. spectator in a red shirt) pulling the drone through netting.
        #
        # Only active when optical flow is available (of_vx_ms not None).
        # Without flow data we cannot estimate position, so the geofence is
        # silent — the altitude penalty and competition rules are the last defence.
        if (self.phase not in (Phase.TAKEOFF, Phase.COMPLETE)
                and (self._est_pos_x_m < self._GEOFENCE_X_MIN or
                     self._est_pos_x_m > self._GEOFENCE_X_MAX or
                     abs(self._est_pos_y_m) > self._GEOFENCE_Y_M)):
            self._geofence_violation_frames += 1
            if self._geofence_violation_frames >= self._GEOFENCE_WARN_FRAMES:
                # Proportional return toward field centre.
                # Gain 0.3 s⁻¹ → 1 m error → 0.3 m/s; capped at 0.5 m/s.
                #
                # BUG FIX (Reversed RTH Arrow): even after Fix #1 stores
                # (_est_pos_x_m, _est_pos_y_m) in the WORLD frame, the RTH
                # velocity vector must be expressed in the BODY frame for
                # ArduPilot SET_POSITION_TARGET (vx = nose, vy = right-wing).
                # Without the inverse rotation, the return direction depends on
                # which way the nose is pointing — it can drive the drone into
                # the netting instead of away from it.
                #
                # Fix: rotate the world-frame proportional return vector through
                # the inverse (= transpose) of the yaw rotation matrix.
                _ret_wx = -self._est_pos_x_m * 0.3   # world-frame return X
                _ret_wy = -self._est_pos_y_m * 0.3   # world-frame return Y
                # Fix J2 — Geofence Vector Clipping (angle distortion):
                # Independent np.clip on X and Y axes snaps any diagonal return vector
                # to exactly 45° (e.g. vx=-1.5, vy=-0.3 → vx=-0.5, vy=-0.5 = 45°).
                # A drone in the corner then flies along the netting instead of away.
                #
                # Fix: treat the return vector as a 2-D direction.  Scale its magnitude
                # uniformly to the speed cap so the angle is always preserved.
                # speed = min(proportional_magnitude, 0.5 m/s)
                _MAX_GF_SPEED = 0.5
                if synced_sd is not None and synced_sd.yaw_deg is not None:
                    # Bug 1 fix: use relative yaw so the body→world rotation
                    # is referenced to the arena, not to Magnetic North.
                    _yaw_rad = np.deg2rad(synced_sd.yaw_deg - (self._yaw_0 or 0.0))
                    _cos_y   = float(np.cos(_yaw_rad))
                    _sin_y   = float(np.sin(_yaw_rad))
                    # Inverse rotation (transpose of rotation matrix): world → body
                    #   body_x =  cos(ψ)·wx + sin(ψ)·wy
                    #   body_y = −sin(ψ)·wx + cos(ψ)·wy
                    _body_x_raw = _cos_y * _ret_wx + _sin_y * _ret_wy
                    _body_y_raw = -_sin_y * _ret_wx + _cos_y * _ret_wy
                else:
                    # No yaw data — world-frame best-effort fallback
                    _body_x_raw = _ret_wx
                    _body_y_raw = _ret_wy
                # Uniform magnitude scaling — never distort direction
                _mag = (_body_x_raw ** 2 + _body_y_raw ** 2) ** 0.5 + 1e-9
                _scale = min(1.0, _MAX_GF_SPEED / _mag)
                _gf_vx = float(_body_x_raw * _scale)
                _gf_vy = float(_body_y_raw * _scale)
                self.last_command = FlightCommand(
                    vx=_gf_vx, vy=_gf_vy, vz=0.0,
                    yaw_lock=True,
                    phase=self.phase.name, action="GEOFENCE_RETURN",
                )
                cv2.putText(frame,
                            f"GEOFENCE! pos=({self._est_pos_x_m:+.1f},{self._est_pos_y_m:+.1f})m",
                            (w // 2 - 200, h // 2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            self._geofence_violation_frames = 0

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
                phase=self.phase.name, action="REACHED_ALT"
            )
            print(f"[TAKEOFF] Altitude {current_altitude:.2f}m reached → SEARCHING")
        else:
            # Gentle ascent command — FC executes, we just report intent
            ascent_vz = -0.8   # m/s upward in NED (negative = up)
            src = getattr(self, '_last_alt_source', 'FC')
            alt_str = f"{current_altitude:.2f}m[{src}]" if current_altitude is not None else "sensor=None"
            self.last_command = FlightCommand(
                vz=ascent_vz, yaw_lock=True,
                phase=self.phase.name, action="ASCEND"
            )
            # HUD overlay
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
        if target is not None and result.masks is not None and result.boxes is not None:
            target_color = target[0]
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id = int(box.cls[0].item())
                if names.get(cls_id, "") != "circle":
                    continue
                geo = _extract_geometry(mask_obj, frame_w, frame_h)
                if geo is None:
                    continue
                cx, cy, radius, fr, cnt, _edge_clipped = geo
                if fr < FILL_RATIO_MIN_CIRCLE:
                    continue
                if h_bboxes and overlaps_h_landing(cx, cy, radius, h_bboxes):
                    continue
                # Apply the same scored-position exclusion as _fire_phase.
                # Without this, F1 (already scored) can be re-detected as F2
                # during the yaw scan, causing an infinite RECOVERY↔TARGETING loop.
                Z_rec = self._last_z_est if self._last_z_est is not None else DROP_ALTITUDE_TARGET
                if self._already_scored(cx, cy, Z_rec, frame_w // 2, frame_h // 2):
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
            self._no_detect_frames = 0
            self._pid_x.reset()
            self._pid_y.reset()
            # ── Fix #3: clear EMA filter on reacquisition ──────────────────
            # Without this reset, _smooth_cx/_smooth_cy still hold the LAST
            # KNOWN position from before the target was lost (e.g. far left).
            # The EMA then "drags" this stale coordinate toward the newly
            # found position over many frames, creating a ghost trajectory.
            # The PID chases that ghost → violent overshoot on reacquisition.
            self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
            self._pos_history.clear()
            self.last_command = FlightCommand(
                yaw_lock=True, phase=self.phase.name, action="TARGET_REACQUIRED"
            )
            return frame

        # Full scan exhausted — skip this target
        if self._recovery_frames >= self._recovery_max:
            print(f"[RECOVERY] Scan exhausted — skipping target {self.current_target_idx}")
            self._recovery_frames  = 0
            self._no_detect_frames = 0
            # Always reset EMA so the next target doesn't inherit a stale centroid
            # (same reason _execute_drop resets it between successful drops).
            self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
            self._pos_history.clear()
            self._pid_x.reset()
            self._pid_y.reset()

            is_last_target = (self.current_target_idx >= len(self._fire_sequence) - 1)
            if is_last_target:
                # Exhausted the final target — count it as a missed drop and
                # move to helipad so we don't spin in SEARCHING forever.
                print("[RECOVERY] Last target missed — transitioning to FIND_HELIPAD")
                self._transition_to_helipad()
                self.last_command = FlightCommand(
                    yaw_lock=True, phase=self.phase.name, action="SKIP_LAST_TARGET"
                )
            else:
                self.current_target_idx += 1
                self.phase = Phase.SEARCHING
                self.last_command = FlightCommand(
                    yaw_lock=True, phase=self.phase.name, action="SKIP_TARGET"
                )
            return frame

        # Compute yaw-scan command for this frame
        step_idx  = min(self._recovery_frames // self._scan_step_frames,
                        len(self._scan_yaw_steps) - 1)
        yaw_rate  = float(self._scan_yaw_steps[step_idx])
        self.last_command = FlightCommand(
            vx=0.0, vy=0.0, vz=0.0,
            yaw_rate=yaw_rate, yaw_lock=False,
            phase=self.phase.name, action=f"SCAN_STEP_{step_idx}"
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
                    current_altitude: float | None = None,
                    alt_source: str = "VIS",
                    synced_sd: "SensorData | None" = None):
        """
        Altitude authority hierarchy (highest → lowest):
          1. current_altitude  (FC baro/rangefinder via MAVLink)  ← passed in
          2. tof_altitude      (MTF-01 ToF laser, safety gate)    ← from synced_sd
          3. Z_est_raw         (pinhole vision)                   ← computed here

        Sensor improvements applied:
          BNO080 attitude gate → drop refused if |pitch| > DROP_MAX_PITCH_DEG (8°) or
            |roll| > DROP_MAX_ROLL_DEG (8°).  At 8° pitch a ball released at 3.5 m
            lands ~49 cm off-centre — borderline for the smallest target (F5, r=20 cm).
          NOTE: tilt correction of Z_est via 1/cos(θ) was intentionally removed.
            cv2.minEnclosingCircle returns the major axis of the projected ellipse
            which equals D * f_px / Z regardless of small tilt angles, so dividing by
            cos(θ) was overcorrecting and caused unnecessary descents.  The attitude
            GATE above is still applied and is the correct mitigation.
          BNO080 angular rate → feedforward guard: angular rate > FF_ANGULAR_RATE_GATE_DPS
            (12°/s) suppresses feedforward.  More reliable than the old cmd_vx proxy
            because it measures ACTUAL drone motion with no lag.
          MTF-01 optical flow → feedforward target isolation: drone-motion contribution
            is subtracted from the pixel-history so _feedforward_velocity() estimates
            pure TARGET motion (Finals F2 moving at 10 cm/s).
          ToF safety gate: if tof_altitude < DROP_ALTITUDE_MIN, drop is blocked even
            when FC or vision says safe — guards against sensor disagreements.
        """
        target = self.current_target
        if target is None:
            self._transition_to_helipad()
            return frame

        target_color, target_label, target_pts, D_real = target
        names = result.names if hasattr(result, 'names') else {}

        # ── Extract sensor values (all optional, from time-synced IMU) ────────
        pitch_deg  = (synced_sd.pitch_deg      if synced_sd else None)
        roll_deg   = (synced_sd.roll_deg       if synced_sd else None)
        pitch_rate = abs(synced_sd.pitch_rate_dps) if (synced_sd and synced_sd.pitch_rate_dps is not None) else None
        roll_rate  = abs(synced_sd.roll_rate_dps)  if (synced_sd and synced_sd.roll_rate_dps  is not None) else None

        # ── Cross-class NMS prep ─────────────────────────────────────────────
        h_bboxes = collect_h_landing_bboxes(result, names)

        best_match = None

        # ── Bug fix: initialise Z_est_raw BEFORE the detection loop ──────────────
        # _already_scored() is called at the top of the loop body (before Z_est_raw
        # is actually computed from radius further down).  Without this seed value,
        # the very first iteration of the loop throws a NameError.
        # Use the last reliable vision estimate so the exclusion zone is still
        # altitude-correct even on the first frame after entering _fire_phase.
        Z_est_raw: float = (self._last_z_est
                            if self._last_z_est is not None
                            else DROP_ALTITUDE_TARGET)

        if result.masks is not None and result.boxes is not None:
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id     = int(box.cls[0].item())
                class_name = names.get(cls_id, "")
                if class_name != "circle":
                    continue

                geo = _extract_geometry(mask_obj, frame_w, frame_h)
                if geo is None:
                    continue
                cx, cy, radius, fr, cnt_frame, edge_clipped = geo

                if fr < FILL_RATIO_MIN_CIRCLE:
                    cv2.drawContours(frame, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                d_ratio = compute_diameter_ratio(radius, frame_w)
                if d_ratio > MAX_FIRE_DIAMETER_RATIO:
                    cv2.drawContours(frame, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                if h_bboxes and overlaps_h_landing(cx, cy, radius, h_bboxes):
                    continue

                if self._already_scored(cx, cy, Z_est_raw, frame_w // 2, frame_h // 2):
                    cv2.drawContours(frame, [cnt_frame], -1, (80, 80, 80), 1)
                    cv2.putText(frame, "SCORED", (cx - 25, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                    continue

                # ── Altitude resolution — vision Z_est for DROP ──────────────
                #
                # WHY vision Z_est is better than ToF for the DROP gate:
                #   Z_est measures the distance from the camera to THIS SPECIFIC
                #   circle on the floor.  ToF measures the distance to whatever
                #   is directly beneath the drone — which may be a TZ gate, the
                #   helipad edge, or the wrong floor section entirely.
                #
                # EDGE-CLIP GUARD (Threat 1 fix):
                #   If the circle is partially off-screen, radius is the enclosing
                #   radius of the VISIBLE ARC — dramatically smaller than the real
                #   circle radius.  Plugging this into Z = f*D/(2r) gives a phantom
                #   altitude 2-3× too high, causing a violent dive command.
                #   When edge_clipped=True we therefore fall back to the last
                #   known GOOD Z_est (from when the circle was fully in frame).
                #   We still use (cx, cy) for XY PID — those remain valid.
                #   A DROP is blocked while edge-clipped: Z is untrustworthy.
                Z_est_raw = self.alt_est.estimate(radius * 2.0, D_real)
                if edge_clipped:
                    # Use last reliable Z, or DROP_ALTITUDE_TARGET as safe default.
                    Z_est_raw = (self._last_z_est
                                 if self._last_z_est is not None
                                 else DROP_ALTITUDE_TARGET)
                else:
                    # Full circle is visible — this Z_est is trustworthy.
                    self._last_z_est = Z_est_raw

                if current_altitude is not None:
                    Z_drop = current_altitude
                    z_src  = alt_source   # "FC" or "TOF" — resolved correctly in update()
                else:
                    Z_drop = Z_est_raw
                    z_src  = "VIS"

                # ToF safety gate: if ToF says too low, block drop regardless.
                tof_blocks_drop = (
                    synced_sd is not None
                    and synced_sd.tof_altitude_m is not None
                    and synced_sd.tof_altitude_m < DROP_ALTITUDE_MIN
                )
                # Edge-clip gate: also block drop when radius is untrustworthy.
                # We cannot confirm safe altitude from a partial circle.
                if edge_clipped:
                    tof_blocks_drop = True   # re-use existing drop-block path

                # Color check
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

                # ── Best detection selection ──────────────────────────────────
                # When two same-color circles are both visible (e.g. F1+F2 both
                # red, or F3+F4 both yellow), we must pick the one the drone is
                # hovering above — not an arbitrary one.
                #
                # OLD logic: pick the detection whose Z_drop is closest to
                # DROP_ALTITUDE_TARGET (3.25 m). Problem: when FC altitude is
                # authoritative, EVERY detection gets the SAME Z_drop (the baro
                # reading). First detection in YOLO output order won — random.
                #
                # NEW logic: pick the detection whose centre (cx, cy) is closest
                # to the frame centre (frame_w//2, frame_h//2). This correctly
                # selects whichever circle the drone is hovering above, regardless
                # of how many same-colour circles are visible in the frame.
                # Z-proximity is kept as a tiebreaker for vision-only mode.
                frame_cx = frame_w // 2
                frame_cy = frame_h // 2
                offset_to_center = (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2
                if best_match is None:
                    best_match = (Z_drop, cx, cy, radius, detected_color,
                                  fr, cnt_frame, z_src, tof_blocks_drop,
                                  offset_to_center, edge_clipped)
                else:
                    prev_offset = best_match[9]
                    # Prefer whichever is closer to frame center.
                    # Only fall back to Z-proximity as tiebreaker in vision mode.
                    if offset_to_center < prev_offset:
                        best_match = (Z_drop, cx, cy, radius, detected_color,
                                      fr, cnt_frame, z_src, tof_blocks_drop,
                                      offset_to_center, edge_clipped)

        if best_match is None:
            self._no_detect_frames += 1
            if self._no_detect_frames >= self._max_no_detect:
                self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
                self._pid_x.reset()
                self._pid_y.reset()
                self._coast_vx = 0.0
                self._coast_vy = 0.0
                self._coast_vz = 0.0
                self._recovery_frames = 0
                self.phase = Phase.RECOVERY
                self.last_command = FlightCommand(
                    yaw_lock=True, phase=self.phase.name, action="ENTERING_RECOVERY"
                )
            else:
                # ── Zero-Order Hold coasting (Threat 2 fix) ──────────────────
                # Hold the EXACT last-known velocity — no decay.
                #
                # Previous code decayed velocity by ×0.75 per frame, which
                # commanded the FC to brake (Δv in one frame = 0.25 m/s).
                # Braking tilts the nose back → camera tilts up → target leaves
                # the frame → RECOVERY triggered by a transient glare event.
                #
                # With Zero-Order Hold, the drone glides flat at constant speed
                # for the ~10-frame (0.33 s) occlusion window so YOLO can
                # reacquire without any FC-induced camera shake.
                # _coast_vx / _coast_vy are already set from the last good frame.
                #
                # Fix #4b (ZOH altitude brake):
                # _coast_vz holds the last good cmd_vz (0.0, 0.6, or -0.5).
                # Forcing vz=0.0 here commanded the FC to brake the Z-axis
                # the moment detection was lost — a sudden throttle change that
                # tilts the airframe, further destabilising the camera.
                # Holding _coast_vz lets the drone continue its altitude
                # trajectory smoothly through the occlusion window.
                self.last_command = FlightCommand(
                    vx=self._coast_vx, vy=self._coast_vy, vz=self._coast_vz,
                    yaw_lock=True, phase=self.phase.name,
                    action=f"ZOH_COAST_{self._no_detect_frames}/{self._max_no_detect}",
                )
            return frame

        self._no_detect_frames = 0
        Z_drop, raw_cx, raw_cy, raw_r, color, fr, cnt_frame, z_src, tof_blocks, _, _edge = best_match
        # tof_blocks_drop is set True by two unrelated paths inside the detection loop:
        #   _tof_low_blocks — ToF sensor reports altitude < DROP_ALTITUDE_MIN
        #   _edge_blocks    — circle is edge-clipped; Z_est is untrustworthy
        # Naming them separately makes the altitude-command logic below explicit.
        _tof_low_blocks: bool = tof_blocks and not _edge
        _edge_blocks:    bool = _edge
        cx, cy, _  = self._smooth(raw_cx, raw_cy, Z_drop)
        radius     = raw_r
        color_bgr  = _color_bgr(color)

        # Raw pixel offset from frame centre
        offset_x = cx - frame_w // 2
        offset_y = cy - frame_h // 2

        # ── Fix J1: compute feedforward BEFORE the PID fires ─────────────────
        # Previous order:
        #   1. PID fires on offset_x/y  (drives to current target position)
        #   2. Predictive offset computed  (drive to FUTURE target position)
        #   3. Drop trigger gated on pred_offset
        # Result: PID holds drone over current F2 position; trigger demands
        # drone to be AHEAD of F2 by v*t_fall pixels.  They can never BOTH be
        # satisfied → drone follows F2 forever and never drops (Intercept Deadlock).
        #
        # Fix: compute FF first, derive _pred_offset_x/y, THEN feed that into
        # the PID.  The PID now steers toward the intercept point.  The moment
        # the drone arrives, pred_offset → 0 and the trigger fires.
        _ff_vx_px: float = 0.0
        _ff_vy_px: float = 0.0
        # Default: use raw geometric offset (non-Finals or FF not available)
        _pid_error_x = float(offset_x)
        _pid_error_y = float(offset_y)
        _pred_offset_x = float(offset_x)
        _pred_offset_y = float(offset_y)

        if self.is_finals:
            # Guard: suppress FF when drone is actively maneuvering
            if pitch_rate is not None and roll_rate is not None:
                _drone_maneuvering = (pitch_rate > FF_ANGULAR_RATE_GATE_DPS or
                                      roll_rate  > FF_ANGULAR_RATE_GATE_DPS)
            else:
                # Proxy: use PID gain × raw offset as estimate of intended cmd speed
                _drone_maneuvering = (
                    abs(float(np.clip(-self._pid_y.kp * offset_y * PX_TO_MS_FIRE, -1, 1))) > FF_CMD_THRESHOLD_MS or
                    abs(float(np.clip( self._pid_x.kp * offset_x * PX_TO_MS_FIRE, -1, 1))) > FF_CMD_THRESHOLD_MS
                )

            if not _drone_maneuvering and len(self._pos_history) >= 4:
                _ff_vx_px, _ff_vy_px = self._feedforward_velocity()
                # Ball free-fall time: t = √(2Z/g)
                _t_fall = (2.0 * max(Z_drop, 0.5) / 9.81) ** 0.5
                # Predicted offset: in t_fall seconds the target moves ff*t_fall further
                # in the same direction it is already travelling.
                #   offset_future = offset_current + ff_velocity * t_fall
                # (positive ff_vx_px → target moving RIGHT → future offset is further right)
                _pred_offset_x = offset_x + _ff_vx_px * _t_fall
                _pred_offset_y = offset_y + _ff_vy_px * _t_fall

                # Feed the INTERCEPT error to the PID.  The PID now steers the drone
                # to where F2 will be, not where it is now.  The trigger fires the
                # instant the drone reaches the intercept point (pred_offset → 0).
                _pid_error_x = _pred_offset_x
                _pid_error_y = _pred_offset_y

        # ── Alignment and attitude — computed BEFORE the PID so that
        #    freeze_integral can use the CURRENT frame's attitude_level.
        #
        # Dependency order (all inputs below are available before the PID):
        #   _pred_offset_x/y  — from FF block above (or raw offset_x/y)
        #   Z_drop, _tof_low_blocks, _edge_blocks — from the detection loop
        #   pitch_deg, roll_deg — from synced_sd (top of update())
        #
        # Integral-windup fix (WAIT_LEVEL scenario):
        #   When the drone is aligned + altitude-safe but the tilt gate blocks
        #   the drop, the PID keeps running to hold position.  Even with
        #   near-zero offset error the integral accumulates over the 1.5 s tilt
        #   timeout.  When the wind stops and attitude finally clears, the wound-
        #   up integral fires a sudden correction — the drone overshoots F2 and
        #   misses the golden drop window.
        #
        #   Fix: pass freeze_integral=not attitude_level to both PID calls below.
        #   P and D still run normally (position correction continues), but the
        #   integral term stays frozen for the entire wait period.  The moment
        #   attitude_level becomes True again the integral is clean — no overshoot.

        # Alignment for drop trigger and tilt counter
        aligned = (abs(_pred_offset_x) < ALIGN_PIXEL_THRESH and
                   abs(_pred_offset_y) < ALIGN_PIXEL_THRESH)

        altitude_safe = (DROP_ALTITUDE_MIN <= Z_drop <= DROP_ALTITUDE_MAX
                         and not _tof_low_blocks)

        # ── Attitude-gated drop (BNO080) ──────────────────────────────────────
        attitude_level  = True
        attitude_reason = ""
        _tilt_threshold = DROP_MAX_PITCH_DEG   # baseline 8°
        # Fix #4: if we've been aligned+safe but tilt-blocked for >1.5 s,
        # the FC is almost certainly tilting to hold position against wind.
        # Relax the gate to _TILT_MAX_RELAXED (15°) — a ball dropped at 15° tilt
        # from 3.5 m lands ~93 cm off-centre, which misses F5 but is the only
        # option when wind prevents the drone from ever going flat.
        # The counter is reset immediately once a drop fires or the drone is no
        # longer in the "aligned + safe" condition.
        if altitude_safe and aligned:
            # Check with the current threshold first (may already be relaxed)
            _tilt_ok = True
            if pitch_deg is not None and abs(pitch_deg) > _tilt_threshold:
                _tilt_ok = False
            if roll_deg  is not None and abs(roll_deg)  > _tilt_threshold:
                _tilt_ok = False
            if not _tilt_ok:
                self._tilt_block_frames += 1
                if self._tilt_block_frames >= self._TILT_RELAX_FRAMES:
                    _tilt_threshold = self._TILT_MAX_RELAXED
            else:
                self._tilt_block_frames = 0
        else:
            self._tilt_block_frames = 0

        if pitch_deg is not None and abs(pitch_deg) > _tilt_threshold:
            attitude_level  = False
            attitude_reason = (f"pitch={pitch_deg:+.1f}°>{_tilt_threshold:.0f}°"
                               + (" [RELAXED]" if _tilt_threshold > DROP_MAX_PITCH_DEG else ""))
        if roll_deg is not None and abs(roll_deg) > _tilt_threshold:
            attitude_level  = False
            attitude_reason += (f" roll={roll_deg:+.1f}°>{_tilt_threshold:.0f}°"
                                + (" [RELAXED]" if _tilt_threshold > DROP_MAX_ROLL_DEG else ""))

        # PID fires on the correct error (intercept in Finals, raw otherwise).
        # timestamp=_last_frame_time  — pre-YOLO wall-clock time eliminates
        #   computation jitter from the D-term (Bug 3 fix).
        # freeze_integral=not attitude_level — prevents integral windup during
        #   WAIT_LEVEL: P and D still guide the drone over the target; the
        #   integral is held frozen until the tilt gate clears (fix above).
        vel_x = self._pid_x.update(float(_pid_error_x),
                                   timestamp=self._last_frame_time,
                                   freeze_integral=not attitude_level)
        vel_y = self._pid_y.update(float(_pid_error_y),
                                   timestamp=self._last_frame_time,
                                   freeze_integral=not attitude_level)

        # ── Fix #9: Ball-trajectory-corrected drop alignment ─────────────────
        # A released ball inherits the drone's current lateral velocity.  At
        # Z=3.25 m the free-fall time is t = √(2·3.25/9.81) ≈ 0.81 s.  A modest
        # correction velocity of 0.20 m/s drifts the ball 0.16 m sideways — that
        # already misses F5 (radius 0.20 m) if the PID is still converging.
        #
        # Body → image sign convention (nadir camera, NED body frame):
        #   cmd_vy = vel_x · PX_TO_MS  (body right +) → image X right  (+drift_x_px)
        #   cmd_vx = -vel_y · PX_TO_MS (body fwd  +) → image Y down   (+drift_y_px)
        #
        # Scale factor:
        #   drift_px = cmd_v_ms · t_fall · f_px / Z
        #            = vel_px · PX_TO_MS · t_fall · f_px / Z
        #            = vel_px · _traj_scale
        #
        # `aligned`         — drone body is within ALIGN_PIXEL_THRESH (computed
        #                     above, before PID, for tilt-block counter + ZOH coast).
        # `_aligned_for_drop` — ball's IMPACT POINT is within ALIGN_PIXEL_THRESH.
        #                     Needs vel_x/vel_y → computed here, after PID.
        _t_fall_traj = (2.0 * max(Z_drop, 0.5) / 9.81) ** 0.5
        _traj_scale  = (PX_TO_MS_FIRE * self.alt_est.f_px
                        * _t_fall_traj / max(Z_drop, 0.5))
        # Clamp vel_x/y to avoid absurd drift on first PID frame (large I spike)
        _vel_x_c = float(np.clip(vel_x, -200.0, 200.0))
        _vel_y_c = float(np.clip(vel_y, -200.0, 200.0))
        _drift_x_px   =  _vel_x_c * _traj_scale   # cmd_vy contribution
        _drift_y_px   = -_vel_y_c * _traj_scale   # cmd_vx contribution (note sign)
        _impact_off_x = _pred_offset_x - _drift_x_px
        _impact_off_y = _pred_offset_y - _drift_y_px

        _aligned_for_drop = (abs(_impact_off_x) < ALIGN_PIXEL_THRESH and
                             abs(_impact_off_y) < ALIGN_PIXEL_THRESH)

        _trigger_drop = (altitude_safe and not _edge_blocks and _aligned_for_drop and
                         attitude_level and self._drop_cooldown == 0)

        if _trigger_drop:
            self.phase = Phase.DROPPING
            self._execute_drop(target_label, target_pts, Z_drop, cx=cx, cy=cy,
                               frame_w=frame_w, frame_h=frame_h)
        else:
            self.phase = Phase.TARGETING

        # ── FlightCommand from PID ────────────────────────────────────────────
        cmd_vx = float(np.clip(-vel_y * PX_TO_MS_FIRE, -1.0, 1.0))
        cmd_vy = float(np.clip( vel_x * PX_TO_MS_FIRE, -1.0, 1.0))

        # Fix #2: altitude command decoupled from tof_blocks.
        #
        # `altitude_safe` intentionally checks only `_tof_low_blocks` (not `_edge_blocks`):
        #   (a) Z_drop is inside the [DROP_ALTITUDE_MIN, DROP_ALTITUDE_MAX] window
        #   (b) _tof_low_blocks is False (ToF sensor does not report genuine low altitude)
        # Edge-clip (_edge_blocks) is excluded here because the altitude COMMAND
        # (cmd_vz) must hold position when the circle is clipped — not ascend.
        # The drop TRIGGER adds `and not _edge_blocks` separately (see _trigger_drop).
        # For the VERTICAL COMMAND we only care about (a). Using the old `altitude_safe`
        # directly caused a silent bug:
        #
        #   Scenario — FC=3.3 m (safe), tof_altitude=2.7 m (real gate block, _edge=False):
        #     altitude_safe = (3.05≤3.3≤4.50) AND NOT tof_blocks = True AND False = False
        #     → tof_blocks AND _edge branch skipped (_edge=False)
        #     → altitude_safe branch skipped (False)
        #     → Z_drop > DROP_ALTITUDE_MAX: 3.3 > 4.50? No → skipped
        #     → else: cmd_vz = -0.5  ← ASCEND at 0.5 m/s even though drone is at 3.3 m!
        #
        # The drone would climb above DROP_ALTITUDE_MAX, then oscillate.
        #
        # Fix: base the altitude command solely on Z_drop (primary sensor value),
        # not on altitude_safe. This correctly handles all cases:
        #   • Z_drop in window (any tof_blocks state): hold altitude
        #   • Z_drop > MAX: descend
        #   • Z_drop < MIN (genuine low altitude, no FC): ascend ← still correct
        if _edge_blocks:
            cmd_vz = 0.0                              # edge clip: Z untrustworthy, hold
        elif DROP_ALTITUDE_MIN <= Z_drop <= DROP_ALTITUDE_MAX:
            cmd_vz = 0.0                              # primary altitude in window: hold
        elif Z_drop > DROP_ALTITUDE_MAX:
            cmd_vz = 0.6                              # too high: descend
        else:
            cmd_vz = -0.5                             # too low: ascend

        # ── Additive feedforward velocity term (Finals F2) ────────────────────
        # _ff_vx_px / _ff_vy_px were computed in the Fix J1 block above.
        # Adding scaled target velocity to cmd_vx/vy ensures the drone's speed
        # matches F2's lateral velocity, reducing steady-state tracking lag.
        # The PID intercept error already drives to the right POSITION; this term
        # adds the right SPEED so the drone doesn't have to "chase" during approach.
        if self.is_finals and (_ff_vx_px != 0.0 or _ff_vy_px != 0.0):
            cmd_vx  += float(np.clip(-_ff_vy_px * FF_SCALE, -0.4, 0.4))
            cmd_vy  += float(np.clip( _ff_vx_px * FF_SCALE, -0.4, 0.4))
            cmd_vx   = float(np.clip(cmd_vx, -1.0, 1.0))
            cmd_vy   = float(np.clip(cmd_vy, -1.0, 1.0))

        # Update coast state (used when target is briefly lost — Zero-Order Hold).
        #
        # Bug 3 fix (previous round): the old code zeroed _coast_vx/vy immediately
        # on the drop frame.  The fix carries the current PID velocity so the ZOH
        # window (0.33 s) keeps the drone gliding flat — no pitch-up, no recovery.
        #
        # Round 6 fix (Bug 2 — ZOH "Handbrake"):
        # _coast_vx/vy MUST be saved AFTER the feedforward block above.  In Finals,
        # when tracking the moving F2 target at 10 cm/s, the PID base cmd_vx is near
        # 0 (drone is centred on the intercept point), and the entire tracking
        # velocity comes from the FF term.  Saving coast BEFORE FF stores 0 m/s —
        # so on the first ZOH frame after ball release ArduPilot sees a 0 m/s
        # setpoint, brakes hard, pitches nose up, and loses lock.
        # Saving AFTER FF ensures the full tracking velocity is held through the
        # ZOH window, allowing natural deceleration without a camera-tilting brake.
        #
        # Fix #4b: also persist cmd_vz so ZOH doesn't issue a sudden altitude-brake
        # (see comment in the ZOH FlightCommand block above).
        self._coast_vx = cmd_vx
        self._coast_vy = cmd_vy
        self._coast_vz = cmd_vz

        if _tof_low_blocks or _edge_blocks:
            if _edge_blocks:
                action = "EDGE_CLIP_BLOCK"
            else:
                # _tof_low_blocks=True: synced_sd.tof_altitude_m is not None and
                # < DROP_ALTITUDE_MIN (the only set-path for _tof_low_blocks).
                action = f"TOF_BLOCK({synced_sd.tof_altitude_m:.2f}m<{DROP_ALTITUDE_MIN}m)"
        elif _trigger_drop:
            action = "DROP"
        elif not altitude_safe:
            # Show altitude action before attitude: drone is actively climbing/descending.
            # Previous order showed "WAIT_LEVEL" even when the drone was ascending to
            # reach drop altitude, misleading CSV logs and live HUD.
            action = "DESCEND" if Z_drop > DROP_ALTITUDE_MAX else "ASCEND"
        elif not attitude_level:
            action = f"WAIT_LEVEL({attitude_reason})"
        else:
            action = "ALIGN"

        self.last_command = FlightCommand(
            vx=cmd_vx, vy=cmd_vy, vz=cmd_vz,
            yaw_rate=0.0, yaw_lock=True,
            drop_ball=_trigger_drop,
            phase=self.phase.name, action=action,
        )

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.drawContours(frame, [cnt_frame], -1, color_bgr, 2)
        cv2.circle(frame, (cx, cy), 5, color_bgr, -1)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color_bgr, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color_bgr, 1)
        fc_x, fc_y = frame_w // 2, frame_h // 2
        cv2.line(frame, (fc_x-15, fc_y), (fc_x+15, fc_y), (255, 255, 255), 1)
        cv2.line(frame, (fc_x, fc_y-15), (fc_x, fc_y+15), (255, 255, 255), 1)

        tof_str  = (f" ToF={synced_sd.tof_altitude_m:.2f}m"
                    if synced_sd and synced_sd.tof_altitude_m is not None else "")
        tilt_str = (f" P={pitch_deg:+.1f}°R={roll_deg:+.1f}°"
                    if pitch_deg is not None else "")
        edge_str = " [EDGE]" if _edge_blocks else ""
        label    = (f"{target_label}[{color}] | Z={Z_drop:.2f}m[{z_src}]"
                    f"{tof_str}{tilt_str}{edge_str} | fr={fr:.2f} | {action}")
        label_y  = max(20, cy - radius - 12)
        cv2.putText(frame, label, (max(0, cx - 80), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

        _draw_altitude_bar(frame, Z_drop, frame_w, frame_h, color_bgr)

        if not attitude_level:
            cv2.putText(frame, f"WAIT LEVEL — {attitude_reason}",
                        (fc_x - 140, fc_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if _tof_low_blocks or _edge_blocks:
            if _edge_blocks:
                _tof_hud = "EDGE CLIP — altitude untrustworthy"
            else:
                # _tof_low_blocks=True → synced_sd.tof_altitude_m is guaranteed
                # not None (same set-path as the action string block above).
                _tof_hud = f"TOF BLOCK ({synced_sd.tof_altitude_m:.2f}m)"
            cv2.putText(frame, _tof_hud,
                        (fc_x - 120, fc_y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame


    def _execute_drop(self, label: str, points: int, Z_est: float = 0.0,
                      cx: int = -1, cy: int = -1,
                      frame_w: int = 0, frame_h: int = 0):
        """Register a successful ball drop and record centroid for world-space exclusion."""
        self.balls_dropped += 1
        self.score         += points
        # Time-based cooldown (2 seconds) instead of hardcoded 60 frames.
        # At 20 FPS: 40 frames.  At 30 FPS: 60 frames.
        self._drop_cooldown = int(DROP_COOLDOWN_S * self._fps_hint)
        # Reset the tilt-block counter so the NEXT target's attitude gate starts
        # fresh.  Without this, a relaxed threshold (15°) from the current drop
        # would bleed into the next targeting cycle — dropping immediately at 9°
        # tilt without the required 1.5-second wait (Fix #4 regression).
        self._tilt_block_frames = 0
        self._pid_x.reset()
        self._pid_y.reset()

        # Record centroid as world-space offset (metres from camera centre).
        # Stored in world space so the exclusion zone is altitude-invariant.
        # At drop time the target is centred within ALIGN_PIXEL_THRESH pixels,
        # so (cx - frame_cx) ≈ 0 — the stored offset is near (0, 0).
        if cx >= 0 and cy >= 0 and frame_w > 0 and frame_h > 0:
            Z_clamp  = max(Z_est, 0.5)
            f        = self.alt_est.f_px
            world_x  = (cx - frame_w // 2) * Z_clamp / f
            world_y  = (cy - frame_h // 2) * Z_clamp / f
            self._scored_positions.append((world_x, world_y))
            print(f"[MISSION] Scored world position: ({world_x:+.3f}m, {world_y:+.3f}m) "
                  f"— exclusion radius {SCORED_WORLD_RADIUS_M}m")

        print(f"[MISSION] DROP {label} at Z={Z_est:.2f}m "
              f"(+{points} pts) | {self.balls_dropped}/5 | score={self.score}")
        # Stub: replace with servo / MAVLink drop command
        # e.g. mavlink_conn.send_ball_drop()

        if self.balls_dropped >= 5:
            self._transition_to_helipad()
        else:
            self.current_target_idx += 1
            self.phase = Phase.SEARCHING
            # Reset EMA, PID, and pos history so the next target starts fresh.
            # Without this, the smoother would initialise from the previous
            # target's centroid, causing an initial position step on the HUD
            # and a derivative spike in the PID when the new target appears.
            self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
            self._pos_history.clear()


    def _world_vel_to_body(self,
                           world_vx: float,
                           world_vy: float,
                           synced_sd: "SensorData | None") -> tuple[float, float]:
        """
        Rotate a velocity vector from the WORLD frame into the BODY frame.

        Uses the inverse (= transpose) of the yaw rotation matrix:
            body_vx =  cos(ψ_rel)·world_vx + sin(ψ_rel)·world_vy
            body_vy = −sin(ψ_rel)·world_vx + cos(ψ_rel)·world_vy

        Falls back to returning the world-frame values unchanged when BNO080
        yaw data is unavailable — less accurate but no worse than the old code.

        WHY this is needed (Bug 2 — Blind RTH Direction):
        The RTH speed (-0.5 m/s) is defined as "-0.5 in the world X direction"
        (toward the Safe Zone from which the drone took off).  Passing this
        directly as `vx` to FlightCommand assumes the body X-axis (nose) always
        points in the world X direction.  During the yaw scan in FIND_HELIPAD
        the nose is continuously rotating, so -0.5 as body vx drives the drone
        backward relative to its CURRENT nose — a random direction that can push
        it deeper into the field.  Rotating the vector through the inverse yaw
        matrix guarantees the drone always flies toward the Safe Zone regardless
        of where the nose is pointing.
        """
        if synced_sd is not None and synced_sd.yaw_deg is not None:
            _yaw_rad = np.deg2rad(synced_sd.yaw_deg - (self._yaw_0 or 0.0))
            _cos_y   = float(np.cos(_yaw_rad))
            _sin_y   = float(np.sin(_yaw_rad))
            return (_cos_y * world_vx + _sin_y * world_vy,
                    -_sin_y * world_vx + _cos_y * world_vy)
        return world_vx, world_vy

    def _transition_to_helipad(self):
        print(f"[MISSION] All balls dropped! Score so far: {self.score}")
        print("[MISSION] Switching to FIND_HELIPAD mode...")
        self.phase = Phase.FIND_HELIPAD
        # Reset shared state so helipad phase starts clean.
        self._no_detect_frames = 0
        self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
        self._pid_hx.reset()
        self._pid_hy.reset()

    # ── Helipad phase logic ───────────────────────────────────────────────────

    def _helipad_phase(self, frame, result, frame_w, frame_h,
                       synced_sd: "SensorData | None" = None):
        names = result.names if hasattr(result, 'names') else {}
        helipad_color = (255, 128, 0)   # orange for H_landing

        # ── Clock-expiry force-land — MUST be the FIRST check in this function ──
        #
        # Previous location: inside the `if not found:` branch.
        # Fatal flaw: if the helipad IS visible at T-4s but d_ratio = 0.40 < 0.70,
        # the code branches into `found = True` → h_vz = 0.3 m/s (slow approach).
        # At 0.3 m/s the drone only descends 1.35 m in 4.5 s and is still ~2 m
        # in the air when the T=300 s buzzer sounds → 0 landing points.
        #
        # Fix: check the clock BEFORE the detection loop so this emergency descent
        # overrides everything — found or not, high or low, approaching or scanning.
        # vz = 1.0 m/s guarantees touchdown in ≤ 4.5 s from DROP_ALTITUDE_MAX.
        _remaining_s = MATCH_TIME_S - self.time_elapsed_s
        if _remaining_s < HELIPAD_FORCE_LAND_S:
            print(f"[HELIPAD] {_remaining_s:.1f}s left — FORCE LAND (clock expiry)")
            self.last_command = FlightCommand(
                vx=0.0, vy=0.0, vz=1.0,
                yaw_lock=True,
                phase=self.phase.name, action="FORCE_LAND_CLOCK",
            )
            cv2.putText(frame,
                        f"FORCE LAND — {_remaining_s:.0f}s LEFT",
                        (frame_w // 2 - 160, frame_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame

        found = False
        if result.masks is not None and result.boxes is not None:
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

                # Helipad approach velocity — PID on pixel offset from centre.
                # P-only control left a steady-state offset (~5-15 px) because
                # the FC's velocity deadband absorbs small commands.  The integral
                # term in _pid_hx/_pid_hy eliminates this.
                pad_offset_x = pad_cx - frame_w // 2
                pad_offset_y = pad_cy - frame_h // 2
                h_vel_x = self._pid_hx.update(float(pad_offset_x))
                h_vel_y = self._pid_hy.update(float(pad_offset_y))
                h_vx = float(np.clip(-h_vel_y * PX_TO_MS_HELIPAD, -0.5, 0.5))
                h_vy = float(np.clip( h_vel_x * PX_TO_MS_HELIPAD, -0.5, 0.5))
                # Descend at 0.3 m/s while approaching; increase to 0.4 m/s
                # once d_ratio confirms we are directly above the pad.
                h_vz = 0.3 if d_ratio < DROP_THRESHOLD else 0.4

                if d_ratio >= DROP_THRESHOLD:
                    if self.phase != Phase.LANDING:
                        print(f"[MISSION] Helipad in range (d={d_ratio:.2f}) — LANDING")
                        # Fix #1 (Geofence Drift Correction — Helipad Anchor):
                        # The optical-flow dead-reckoning position (_est_pos_x_m,
                        # _est_pos_y_m) can have accumulated several metres of drift
                        # over a 5-minute flight.  The helipad is always inside the
                        # Safe Zone (x ≤ SAFE_ZONE_DEPTH = 3.5 m, y ≈ 0).
                        #
                        # The moment d_ratio ≥ DROP_THRESHOLD we have a reliable
                        # visual confirmation: the drone IS over the helipad.  We
                        # know its world position with physical certainty (Safe Zone
                        # centre ≈ (1.75, 0)).  Resetting the odometry HERE makes the
                        # blind-RTH command in FIND_HELIPAD / BLIND_DESCENT immune to
                        # accumulated drift — the wrong _est_pos_x_m > 4.0 check can
                        # no longer trigger a 0.5 m/s backward command while the
                        # drone is already above the pad.
                        self._est_pos_x_m = SAFE_ZONE_DEPTH / 2.0   # ~1.75 m
                        self._est_pos_y_m = 0.0
                        self._geofence_violation_frames = 0
                        print(f"[GEOFENCE] Helipad anchor — odometry reset to "
                              f"({self._est_pos_x_m:.2f}, 0.00) m")
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

                # ── Bug fix (Helipad Limbo): Escape 2 — Scan exhaustion ──────
                # (Escape 1 — clock deadline — was moved to the TOP of this
                # function so it fires regardless of whether the pad is visible.)
                #
                # Old code clamped step_idx at len(_scan_yaw_steps)-1 = 3 (0°/s).
                # After the full sweep the drone hovered indefinitely — no yaw,
                # no descent, no exit — burning all remaining battery.
                #
                # Escape 2 (> _recovery_max * 3 ≈ 9 s of scan):
                #   Begin a very slow blind descent (0.15 m/s).  Keeps the drone
                #   drifting toward the ground so a late landing is still possible.

                _scan_exhausted = (self._no_detect_frames
                                   > self._recovery_max * 3)   # ≈ 9 s of scan

                if _scan_exhausted:
                    # Escape 2: full sweeps done with no pad found.
                    # Round 6 fix (Bug 3 — Helipad Mirage): if still out in the
                    # field (beyond Safe Zone + 0.5 m buffer), fly back toward the
                    # Safe Zone while descending slowly.  Blind descent in-place
                    # at X > 4 m guarantees landing far from the helipad (0 landing
                    # pts); adding RTH here is the only way to recover the bonus.
                    _blind_vz = 0.15   # very gentle; stops when MAVLink LAND_COMPLETE
                    # RTH velocity expressed in WORLD frame (-0.5 = toward Safe Zone).
                    # _world_vel_to_body() rotates it into the current BODY frame so
                    # the direction is correct regardless of the drone's current yaw
                    # (Bug 2 — Blind RTH Direction fix).
                    _rth_world_vx = -0.5 if self._est_pos_x_m > SAFE_ZONE_DEPTH + 0.5 else 0.0
                    _blind_bx, _blind_by = self._world_vel_to_body(_rth_world_vx, 0.0, synced_sd)
                    self.last_command = FlightCommand(
                        vx=_blind_bx, vy=_blind_by, vz=_blind_vz,
                        yaw_lock=True,
                        phase=self.phase.name,
                        action="BLIND_DESCENT" + ("_RTH" if _rth_world_vx else ""),
                    )
                    _rth_suffix_e2 = " + RTH" if _rth_world_vx else ""
                    cv2.putText(frame,
                                f"HELIPAD BLIND DESCENT (scan exhausted){_rth_suffix_e2}",
                                (10, frame_h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
                else:
                    # Normal yaw-scan sweep.
                    # Round 6 fix (Bug 3 — Helipad Mirage): if still out in the
                    # field, fly backward toward Safe Zone while scanning.  The
                    # downward camera cannot see the helipad from X > 4 m away —
                    # hovering in place wastes all remaining time and guarantees
                    # 0 landing points.  vx=-0.5 m/s RTH + yaw scan runs
                    # simultaneously so we home AND search at the same time.
                    scan_frame  = self._no_detect_frames - self._max_no_detect
                    step_idx    = min(scan_frame // self._scan_step_frames,
                                      len(self._scan_yaw_steps) - 1)
                    yaw_rate      = float(self._scan_yaw_steps[step_idx])
                    _scan_world_vx = -0.5 if self._est_pos_x_m > SAFE_ZONE_DEPTH + 0.5 else 0.0
                    _scan_bx, _scan_by = self._world_vel_to_body(_scan_world_vx, 0.0, synced_sd)
                    self.last_command = FlightCommand(
                        vx=_scan_bx, vy=_scan_by, vz=0.0,
                        yaw_rate=yaw_rate, yaw_lock=False,
                        phase=self.phase.name,
                        action=f"SCAN_STEP_{step_idx}" + ("_RTH" if _scan_world_vx else ""),
                    )
                    _rth_suffix_scan = " + RTH" if _scan_world_vx else ""
                    cv2.putText(frame,
                                f"HELIPAD SCAN — step {step_idx+1} "
                                f"yaw={yaw_rate:+.0f} deg/s{_rth_suffix_scan}",
                                (10, frame_h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
            else:
                # Helipad temporarily lost — hover or fly back if far from Safe Zone.
                # Round 6 fix (Bug 3): begin RTH immediately even on the first few
                # frames of lost detection — no reason to hover at X > 4 m.
                _early_world_vx = -0.5 if self._est_pos_x_m > SAFE_ZONE_DEPTH + 0.5 else 0.0
                _early_bx, _early_by = self._world_vel_to_body(_early_world_vx, 0.0, synced_sd)
                self.last_command = FlightCommand(
                    vx=_early_bx, vy=_early_by, vz=0.0,
                    yaw_lock=True, phase=self.phase.name,
                    action=(f"NO_DETECT_{self._no_detect_frames}/{self._max_no_detect}"
                            + ("_RTH" if _early_world_vx else "")),
                )
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
        self.last_command = FlightCommand(phase=self.phase.name, action="LANDED")
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
        self._low_alt_frames    = 0
        self._smooth_cx = self._smooth_cy = self._smooth_ratio = None
        self._pos_history.clear()
        self._scored_positions.clear()
        self._last_z_est        = None
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_hx.reset()
        self._pid_hy.reset()
        self._last_alt_authority = None
        self._last_alt_source    = "VIS"
        self._coast_vx  = 0.0
        self._coast_vy  = 0.0
        self._coast_vz  = 0.0
        self._timer_forced_land = False
        self._start_time = None
        self.time_elapsed_s = 0.0
        self._tof_buffer.clear()
        self._imu_buffer.clear()
        # Fix #4 — reset tilt-block counter
        self._tilt_block_frames = 0
        # Fix #7 — reset dead-reckoning position and geofence state
        self._est_pos_x_m              = 0.0
        self._est_pos_y_m              = 0.0
        self._geofence_violation_frames = 0
        self._prev_yaw_deg             = None   # Bug 1: Δψ rotation state
        self._yaw_0                    = None   # Bug 1: re-latch on next active frame
        self._last_frame_time          = None   # Bug 2: real-dt odometry
        self._tof_consecutive_spikes   = 0      # Bug 2 fix: clear spike streak
        self.last_command = FlightCommand(phase=self.phase.name, action="RETRY_RESET")
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
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 300, 158
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
    alt_src   = getattr(mission, '_last_alt_source', 'VIS')
    lines = [
        (f"Phase: {mission.phase.name}  {round_tag}  [{alt_src}]", pc),
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

    # Timer row — turns orange < 60 s, red < 20 s
    elapsed = getattr(mission, 'time_elapsed_s', 0.0)
    remain  = max(0.0, MATCH_TIME_S - elapsed)
    t_color = (0, 0, 220) if remain < 20 else (0, 165, 255) if remain < 60 else (120, 120, 120)
    lines.append((f"T: {int(elapsed)}s  left: {int(remain)}s", t_color))

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


# ---------------------------------------------------------------------------
# Runtime HSV calibration loader
#
# Usage:
#   Load tuned ranges from hsv_calibrator.py output before starting inference:
#     import mission
#     mission.load_hsv_calibration("hsv_calib.json")
#
# This overrides the hardcoded COLOR_HSV dict for the current process.
# Safe to call at any point — takes effect on the next color_of_mask() call.
# ---------------------------------------------------------------------------

def load_hsv_calibration(json_path: str) -> None:
    """
    Load HSV colour ranges from a JSON file produced by hsv_calibrator.py
    and update the module-level COLOR_HSV dict in place.

    The JSON format mirrors COLOR_HSV:
      {
        "red":    [{"lo": [0, 120, 80], "hi": [10, 255, 255]},
                   {"lo": [165, 120, 80], "hi": [180, 255, 255]}],
        "yellow": [{"lo": [18, 100, 100], "hi": [38, 255, 255]}],
        "blue":   [{"lo": [100, 130, 50], "hi": [145, 255, 255]}]
      }

    Args:
        json_path : Path to the JSON file output by hsv_calibrator.py.
    """
    global COLOR_HSV
    with open(json_path) as f:
        raw = json.load(f)
    new_hsv: dict = {}
    for color, ranges in raw.items():
        new_hsv[color] = [
            (tuple(r["lo"]), tuple(r["hi"])) for r in ranges
        ]
    # Validate structure
    for expected in ("red", "yellow", "blue"):
        if expected not in new_hsv:
            raise ValueError(
                f"hsv_calibration JSON missing key '{expected}' — "
                f"run hsv_calibrator.py to regenerate."
            )
    # Single rebind is atomic under CPython's GIL — avoids the TOCTOU window
    # that a .clear() + .update() pair would create if color_of_mask() ran
    # between the two operations on a second thread.
    globals()["COLOR_HSV"] = new_hsv
    print(f"[HSV] Loaded calibration from {json_path}: "
          + "  ".join(f"{k}:{len(v)} range(s)" for k, v in new_hsv.items()))