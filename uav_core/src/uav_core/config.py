"""Mission configuration: one validated, immutable schema.

Defaults are the legacy constants. A YAML file is loaded once, at the boundary (launch file
or CLI), and the resulting objects are passed explicitly to constructors; nothing reads
module globals. Invalid values fail at startup instead of being corrected mid-flight
(ADR-0002). docs/config-reference.md is generated from this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uav_core.types import Color, Round

HsvTriplet = tuple[int, int, int]
HsvRange = tuple[HsvTriplet, HsvTriplet]

RULE_MIN_DROP_ALTITUDE_M = 3.05  # rulebook §III-4: >= 3.0 m, plus a 5 cm margin
RULE_MAX_AUTO_ALTITUDE_M = 5.0  # rulebook §III-4: autonomous flight within 3.0-5.0 m

DEFAULT_HSV_RANGES: dict[Color, tuple[HsvRange, ...]] = {
    Color.RED: (((0, 120, 80), (10, 255, 255)), ((165, 120, 80), (180, 255, 255))),
    Color.YELLOW: (((18, 100, 100), (38, 255, 255)),),
    Color.BLUE: (((100, 130, 50), (145, 255, 255)),),
}


def _field(default: Any, unit: str, description: str, source: str = "legacy tuning", **kw: Any):
    return Field(
        default, description=description, json_schema_extra={"unit": unit, "source": source}, **kw
    )


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PidGains(_Section):
    kp: float = Field(ge=0.0)
    ki: float = Field(ge=0.0)
    kd: float = Field(ge=0.0)


class PerceptionConfig(_Section):
    """Filters applied to segmentation masks before the mission sees a detection.

    Every colour needs at least one HSV range, each with 0 <= lo <= hi <= (180, 255, 255)."""

    fill_ratio_min_circle: float = _field(
        0.40,
        "ratio",
        "Minimum mask fill ratio of a solid fire circle; rejects rings and arcs",
        gt=0.0,
        le=1.0,
    )
    h_landing_overlap_thresh: float = _field(
        0.45,
        "ratio",
        "Share of a circle's box inside an H_landing box above which the circle is suppressed",
        gt=0.0,
        le=1.0,
    )
    edge_clip_margin_px: int = _field(
        5,
        "px",
        "A contour this close to the frame border is edge-clipped; its radius is not "
        "trusted for altitude",
        ge=0,
    )
    color_match_threshold: float = _field(
        0.25, "ratio", "Minimum share of mask pixels inside a colour's HSV ranges", gt=0.0, le=1.0
    )
    hsv_ranges: dict[Color, tuple[HsvRange, ...]] = _field(
        DEFAULT_HSV_RANGES,
        "HSV (OpenCV)",
        "Inclusive (lo, hi) HSV ranges per colour; tune on site with tools/hsv_calibrator.py",
        source="field calibration",
    )

    @field_validator("hsv_ranges")
    @classmethod
    def _check_hsv(cls, value: dict[Color, tuple[HsvRange, ...]]):
        missing = set(Color) - set(value)
        if missing:
            raise ValueError(f"hsv_ranges is missing {sorted(missing)}")
        for color, ranges in value.items():
            for lo, hi in ranges:
                for (a, b), limit in zip(zip(lo, hi, strict=True), (180, 255, 255), strict=True):
                    if not 0 <= a <= b <= limit:
                        raise ValueError(f"bad HSV range for {color}: {lo}..{hi}")
        return value


class CameraConfig(_Section):
    """Downward camera intrinsics."""

    f_px: float = _field(
        500.0,
        "px",
        "Focal length; always override with the calibrated value",
        source="camera calibration",
        gt=0.0,
    )


class DropConfig(_Section):
    """Ball-release window and attitude gate.

    Cross-field checks: altitude_min_m >= 3.05 m (rulebook floor + margin);
    altitude_min_m < altitude_target_m < altitude_max_m <= 5.0 m;
    tilt_max_relaxed_deg >= max_pitch_deg."""

    altitude_min_m: float = _field(
        3.05,
        "m",
        "Lowest altitude at which a ball may be released",
        source="rulebook §III-4 (>= 3.0 m) + 5 cm margin",
    )
    altitude_target_m: float = _field(
        3.25,
        "m",
        "Preferred drop altitude; also the fallback vision altitude before any estimate exists",
    )
    altitude_max_m: float = _field(4.50, "m", "Highest drop altitude; above it the drone descends")
    align_pixel_thresh_px: int = _field(
        25, "px", "Max offset of the predicted impact point from the frame centre", gt=0
    )
    max_fire_diameter_ratio: float = _field(
        0.55,
        "ratio",
        "Largest circle diameter / frame width accepted as a fire target",
        gt=0.0,
        le=1.0,
    )
    scored_world_radius_m: float = _field(
        0.60, "m", "Exclusion radius around circles already hit", gt=0.0
    )
    max_pitch_deg: float = _field(8.0, "deg", "Max |pitch| and |roll| at release", gt=0.0)
    max_roll_deg: float = _field(
        8.0,
        "deg",
        "Roll limit shown in WAIT_LEVEL messages; the gate "
        "itself uses max_pitch_deg for both axes, as legacy did",
        gt=0.0,
    )
    tilt_max_relaxed_deg: float = _field(
        15.0, "deg", "Relaxed tilt limit after tilt_relax_after_s of tilt-blocked hovering"
    )
    tilt_relax_after_s: float = _field(1.5, "s", "Tilt-blocked time before relaxing", gt=0.0)
    cooldown_s: float = _field(2.0, "s", "Minimum time between two releases", ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.altitude_min_m < RULE_MIN_DROP_ALTITUDE_M:
            raise ValueError(
                f"drop.altitude_min_m={self.altitude_min_m} is below the rulebook "
                f"floor {RULE_MIN_DROP_ALTITUDE_M} m"
            )
        if not self.altitude_min_m < self.altitude_target_m < self.altitude_max_m:
            raise ValueError("need drop.altitude_min_m < altitude_target_m < altitude_max_m")
        if self.altitude_max_m > RULE_MAX_AUTO_ALTITUDE_M:
            raise ValueError(f"drop.altitude_max_m must be <= {RULE_MAX_AUTO_ALTITUDE_M} m")
        if self.tilt_max_relaxed_deg < self.max_pitch_deg:
            raise ValueError("drop.tilt_max_relaxed_deg must be >= max_pitch_deg")
        return self


class ControlConfig(_Section):
    """Velocity commands and PID gains."""

    takeoff_alt_m: float = _field(
        3.3, "m", "Altitude that ends TAKEOFF", source="rulebook §VI (>= 3.0 m) + margin", ge=3.0
    )
    takeoff_ascent_ms: float = _field(0.8, "m/s", "Climb rate requested during TAKEOFF", gt=0.0)
    descend_ms: float = _field(0.6, "m/s", "Descent rate when above the drop window", gt=0.0)
    ascend_ms: float = _field(0.5, "m/s", "Climb rate when below the drop window", gt=0.0)
    max_speed_ms: float = _field(1.0, "m/s", "Horizontal speed limit while targeting", gt=0.0)
    px_to_ms_fire: float = _field(0.005, "m/s per px", "Fire PID output scale", gt=0.0)
    px_to_ms_helipad: float = _field(0.004, "m/s per px", "Helipad PID output scale", gt=0.0)
    pid_qualifier: PidGains = _field(
        PidGains(kp=0.5, ki=0.01, kd=0.15), "-", "Fire-target PID gains, qualifier round"
    )
    pid_finals: PidGains = _field(
        PidGains(kp=0.6, ki=0.02, kd=0.20), "-", "Fire-target PID gains, finals (moving F2)"
    )
    pid_helipad: PidGains = _field(
        PidGains(kp=0.35, ki=0.03, kd=0.10), "-", "Helipad approach PID gains"
    )
    pid_output_limit: float = _field(100.0, "px", "PID output clamp", gt=0.0)
    smoothing_alpha_qualifier: float = _field(
        0.4, "-", "EMA weight on the OLD target position (0 = no smoothing)", ge=0.0, lt=1.0
    )
    smoothing_alpha_finals: float = _field(
        0.15, "-", "EMA weight on the old position in the finals (less lag on F2)", ge=0.0, lt=1.0
    )


class FeedforwardConfig(_Section):
    """Moving-target feedforward (finals F2), disabled while the drone manoeuvres."""

    scale: float = _field(
        0.003, "m/s per px/s", "Target pixel velocity -> added drone velocity (finals F2)", ge=0.0
    )
    max_px_s: float = _field(60.0, "px/s", "Clamp on the estimated target velocity", gt=0.0)
    max_add_ms: float = _field(0.4, "m/s", "Clamp on the added feedforward velocity", ge=0.0)
    cmd_threshold_ms: float = _field(
        0.35,
        "m/s",
        "Without IMU rates, estimated command speed above which the drone counts "
        "as manoeuvring and feedforward is off",
        gt=0.0,
    )
    angular_rate_gate_dps: float = _field(
        12.0, "deg/s", "IMU pitch/roll rate above which feedforward is off", gt=0.0
    )
    history_len: int = _field(8, "samples", "Target position history for the velocity fit", ge=4)


class TimingConfig(_Section):
    """Time windows, in seconds of match time rather than frames (ADR-0001)."""

    first_step_dt_s: float = _field(
        1.0 / 30.0, "s", "Time step assumed for the very first step (legacy: 1 / fps_hint)", gt=0.0
    )
    no_detect_timeout_s: float = _field(
        0.33, "s", "Target lost this long -> RECOVERY / helipad scan", gt=0.0
    )
    recovery_timeout_s: float = _field(3.0, "s", "Yaw scan length before skipping a target", gt=0.0)
    scan_step_s: float = _field(0.67, "s", "Duration of one yaw-scan step", gt=0.0)
    scan_yaw_steps_dps: tuple[float, ...] = _field(
        (30.0, -60.0, 30.0, 0.0),
        "deg/s",
        "Yaw rate of each scan step (+ = clockwise)",
        min_length=1,
    )
    helipad_scan_exhausted_s: float = _field(
        9.0, "s", "Helipad not seen for longer than this -> blind descent", gt=0.0
    )
    land_early_at_s: float = _field(
        285.0,
        "s",
        "Match time that forces FIND_HELIPAD",
        source="rulebook §XI (300 s) - 15 s margin",
        gt=0.0,
    )
    helipad_force_land_s: float = _field(
        5.0, "s", "Remaining match time that forces a 1 m/s descent", gt=0.0
    )
    geofence_warn_s: float = _field(
        1.0, "s", "Time outside the soft fence before returning", gt=0.0
    )


class EstimationConfig(_Section):
    """Rangefinder filtering and camera/IMU synchronisation."""

    tof_window: int = _field(5, "samples", "ToF median window", ge=3)
    tof_spike_threshold_m: float = _field(
        1.5, "m", "ToF deviation from the median rejected as an obstacle spike", gt=0.0
    )
    tof_spike_reset_count: int = _field(
        4, "samples", "Consecutive spikes treated as a real step change", ge=1
    )
    camera_latency_s: float = _field(
        0.070, "s", "Capture-to-decision latency used when a frame has no capture stamp", ge=0.0
    )
    imu_buffer_len: int = _field(60, "samples", "IMU history kept for frame synchronisation", ge=1)


class HelipadConfig(_Section):
    """Return to the Safe Zone, helipad approach and landing."""

    land_ratio: float = _field(
        0.70, "ratio", "Helipad box width / frame width that starts LANDING", gt=0.0, le=1.0
    )
    approach_ratio: float = _field(0.40, "ratio", "Box ratio reported as APPROACH", gt=0.0)
    align_ratio: float = _field(0.15, "ratio", "Box ratio reported as ALIGN", gt=0.0)
    approach_vz_ms: float = _field(0.3, "m/s", "Descent rate while approaching", ge=0.0)
    land_vz_ms: float = _field(0.4, "m/s", "Descent rate once LANDING", ge=0.0)
    max_speed_ms: float = _field(0.5, "m/s", "Horizontal speed limit over the helipad", gt=0.0)
    rth_speed_ms: float = _field(0.5, "m/s", "Return speed toward the Safe Zone", ge=0.0)
    rth_margin_m: float = _field(
        0.5, "m", "Return when farther than Safe Zone depth + this", ge=0.0
    )
    blind_descent_vz_ms: float = _field(
        0.15, "m/s", "Descent rate after the scan is exhausted", ge=0.0
    )
    force_land_vz_ms: float = _field(
        1.0, "m/s", "Descent rate when the match clock runs out", gt=0.0
    )


class GeofenceConfig(_Section):
    """Soft fence kept inside the field nets."""

    x_min_m: float = _field(-1.0, "m", "Soft fence behind the takeoff point")
    margin_m: float = _field(1.0, "m", "Soft fence distance inside the far and side nets", ge=0.0)
    return_gain: float = _field(0.3, "1/s", "Proportional return gain", gt=0.0)
    max_speed_ms: float = _field(0.5, "m/s", "Return speed cap", gt=0.0)


class MissionConfig(_Section):
    """Everything the mission can tune; YAML top-level keys are these sections."""

    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    drop: DropConfig = Field(default_factory=DropConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    feedforward: FeedforwardConfig = Field(default_factory=FeedforwardConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    helipad: HelipadConfig = Field(default_factory=HelipadConfig)
    geofence: GeofenceConfig = Field(default_factory=GeofenceConfig)


class FireTarget(_Section):
    color: Color
    label: str
    points: int
    diameter_m: float = Field(gt=0.0)
    moving: bool = False


QUALIFIER_TARGETS = (
    FireTarget(color=Color.RED, label="F1", points=20, diameter_m=1.0),
    FireTarget(color=Color.RED, label="F2", points=20, diameter_m=1.0),
    FireTarget(color=Color.YELLOW, label="F3", points=30, diameter_m=0.6),
    FireTarget(color=Color.YELLOW, label="F4", points=30, diameter_m=0.6),
    FireTarget(color=Color.BLUE, label="F5", points=50, diameter_m=0.4),
)
FINALS_TARGETS = (
    FireTarget(color=Color.RED, label="F1", points=10, diameter_m=1.0),
    FireTarget(color=Color.RED, label="F2", points=50, diameter_m=1.0, moving=True),
    FireTarget(color=Color.YELLOW, label="F3", points=30, diameter_m=0.6),
    FireTarget(color=Color.YELLOW, label="F4", points=30, diameter_m=0.6),
    FireTarget(color=Color.BLUE, label="F5", points=50, diameter_m=0.4),
)


class RulebookConfig(_Section):
    """Competition rules (rulebook 27/02/2026). Not loaded from YAML: change only if the
    rules change. The F1..F5 order is team strategy (§V lets the team choose)."""

    qualifier_targets: tuple[FireTarget, ...] = QUALIFIER_TARGETS
    finals_targets: tuple[FireTarget, ...] = FINALS_TARGETS
    takeoff_bonus_qualifier: int = 20
    takeoff_bonus_finals: int = 10
    landing_bonus_qualifier: int = 30
    landing_bonus_finals: int = 20
    match_time_s: float = 300.0
    alt_penalty_floor_m: float = 3.0
    alt_penalty_after_s: float = 5.0
    penalty_altitude_pts: int = -5
    field_length_m: float = 24.0
    field_width_m: float = 12.0
    safe_zone_depth_m: float = 3.5

    def targets(self, round_: Round) -> tuple[FireTarget, ...]:
        return self.finals_targets if round_ is Round.FINALS else self.qualifier_targets

    def takeoff_bonus(self, round_: Round) -> int:
        return self.takeoff_bonus_finals if round_ is Round.FINALS else self.takeoff_bonus_qualifier

    def landing_bonus(self, round_: Round) -> int:
        return self.landing_bonus_finals if round_ is Round.FINALS else self.landing_bonus_qualifier


def load_mission_config(path: str | Path) -> MissionConfig:
    """Load and validate a YAML file whose top-level keys are MissionConfig sections."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping of config sections, got {type(data).__name__}"
        )
    return MissionConfig.model_validate(data)


def with_hsv_ranges(cfg: MissionConfig, ranges: dict[Color, tuple[HsvRange, ...]]) -> MissionConfig:
    """Return a copy of cfg using field-calibrated HSV ranges (validated)."""
    perception = PerceptionConfig.model_validate(
        {**cfg.perception.model_dump(), "hsv_ranges": ranges}
    )
    return cfg.model_copy(update={"perception": perception})
