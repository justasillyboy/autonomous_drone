"""Immutable data contracts shared by perception, the mission and the ROS nodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TargetClass(StrEnum):
    CIRCLE = "circle"
    SEMICIRCLE = "semicircle"
    H_LANDING = "H_landing"


class Color(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"


class Phase(StrEnum):
    TAKEOFF = "TAKEOFF"
    SEARCHING = "SEARCHING"
    TARGETING = "TARGETING"
    DROPPING = "DROPPING"
    RECOVERY = "RECOVERY"
    FIND_HELIPAD = "FIND_HELIPAD"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"


class AltSource(StrEnum):
    FC = "FC"  # flight-controller altitude
    TOF = "TOF"  # filtered MTF-01 rangefinder
    VIS = "VIS"  # pinhole estimate from a circle's pixel diameter


class Round(StrEnum):
    QUALIFIER = "qualifier"
    FINALS = "finals"


@dataclass(frozen=True, slots=True)
class Detection:
    """One perceived object in frame pixels; integers are truncated as in the legacy code."""

    cls: TargetClass
    color: Color | None
    cx: int
    cy: int
    radius: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    fill_ratio: float
    conf: float
    edge_clipped: bool


@dataclass(frozen=True, slots=True)
class SensorSnapshot:
    """Sensor readings synchronised to one camera frame; None means not available."""

    stamp_s: float | None = None
    fc_alt_m: float | None = None
    tof_alt_m: float | None = None
    of_vx_ms: float | None = None  # optical-flow body velocity, forward
    of_vy_ms: float | None = None  # optical-flow body velocity, right
    pitch_deg: float | None = None
    roll_deg: float | None = None
    yaw_deg: float | None = None
    pitch_rate_dps: float | None = None
    roll_rate_dps: float | None = None


@dataclass(frozen=True, slots=True)
class FlightCommand:
    """Velocity setpoint in the body FRD frame: +vx forward, +vy right, +vz down,
    +yaw_rate clockwise seen from above. Convert with uav_core.frames before sending."""

    phase: Phase
    action: str
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate_dps: float = 0.0
    yaw_lock: bool = True
    drop_ball: bool = False


@dataclass(frozen=True, slots=True)
class MissionStatus:
    phase: Phase
    round: Round
    score: int
    penalties: int
    balls_dropped: int
    target_index: int
    alt_source: AltSource
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class StepOutput:
    command: FlightCommand
    status: MissionStatus
