"""Everything mission_node does except talking to ROS, so it is testable without rclpy.

Messages are read and written by attribute (duck typing): the node passes generated ROS
messages, the tests pass SimpleNamespace objects. Every sign or unit change between ROS
conventions (FLU/ENU, rad/s) and uav_core (FRD/NED, deg) goes through uav_core.frames
(ADR-0003). The legacy mission_control_node skipped this for the MAVROS body velocity and
the IMU attitude, so lateral motion and yaw entered the mission mirrored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from uav_core.config import MissionConfig, RulebookConfig
from uav_core.estimation.imu_sync import ImuSyncBuffer
from uav_core.frames import enu_quaternion_to_frd_euler_deg, flu_rates_to_frd_dps, flu_to_body_frd
from uav_core.mission import MissionStateMachine
from uav_core.types import Detection, FlightCommand, Phase, Round, SensorSnapshot, StepOutput

MAV_LANDED_STATE_ON_GROUND = 1
_LANDING_PHASES = (Phase.FIND_HELIPAD, Phase.LANDING)


class SensorState:
    """Latest FC altitude, rangefinder and body velocity, plus an IMU history for syncing
    the attitude to a frame's capture time. Inputs are in ROS conventions."""

    def __init__(self, imu_buffer_len: int):
        self._base = SensorSnapshot()
        self._imu = ImuSyncBuffer(imu_buffer_len)

    def on_rel_alt(self, alt_m: float) -> None:
        self._base = replace(self._base, fc_alt_m=float(alt_m))

    def on_range(self, range_m: float, min_m: float, max_m: float) -> None:
        """sensor_msgs/Range; out-of-range readings are sentinels, not measurements."""
        if math.isfinite(range_m) and min_m <= range_m <= max_m:
            self._base = replace(self._base, tof_alt_m=float(range_m))

    def on_velocity_body_flu(self, vx: float, vy: float) -> None:
        """/mavros/local_position/velocity_body (FLU) -> forward/right in FRD."""
        fwd, right, _, _ = flu_to_body_frd(vx, vy, 0.0, 0.0)
        self._base = replace(self._base, of_vx_ms=fwd, of_vy_ms=right)

    def on_imu(self, stamp_s: float, orientation: Any, angular_velocity: Any) -> None:
        """sensor_msgs/Imu; an all-zero quaternion means "no orientation" and is skipped."""
        q = orientation
        if q.x == q.y == q.z == q.w == 0.0:
            return
        roll, pitch, yaw = enu_quaternion_to_frd_euler_deg(q.x, q.y, q.z, q.w)
        w = angular_velocity
        roll_rate, pitch_rate, _ = flu_rates_to_frd_dps(w.x, w.y, w.z)
        self._imu.push(
            stamp_s,
            SensorSnapshot(
                pitch_deg=pitch,
                roll_deg=roll,
                yaw_deg=yaw,
                pitch_rate_dps=pitch_rate,
                roll_rate_dps=roll_rate,
            ),
        )

    def snapshot(self, stamp_s: float) -> SensorSnapshot:
        """Latest altitude/velocity with the IMU sample nearest to stamp_s."""
        imu = self._imu.nearest(stamp_s)
        out = replace(self._base, stamp_s=stamp_s)
        if imu is None:
            return out
        return replace(
            out,
            pitch_deg=imu.pitch_deg,
            roll_deg=imu.roll_deg,
            yaw_deg=imu.yaw_deg,
            pitch_rate_dps=imu.pitch_rate_dps,
            roll_rate_dps=imu.roll_rate_dps,
        )


@dataclass
class Watchdog:
    """Expires when nothing was fed for longer than timeout_s (never before the first feed)."""

    timeout_s: float
    last_s: float | None = None

    def feed(self, now_s: float) -> None:
        self.last_s = now_s

    def expired(self, now_s: float) -> bool:
        return self.last_s is not None and now_s - self.last_s > self.timeout_s


class MissionRunner:
    """The state machine plus the operator controls mission_node exposes."""

    def __init__(
        self,
        cfg: MissionConfig,
        round_: Round = Round.QUALIFIER,
        skip_takeoff: bool = False,
        rules: RulebookConfig | None = None,
    ):
        self.sm = MissionStateMachine(cfg, rules, round_)
        self.sensors = SensorState(cfg.estimation.imu_buffer_len)
        self.e_stopped = False
        self._skip_takeoff = skip_takeoff
        self._armed = False
        self._on_ground: bool | None = None
        if skip_takeoff:
            self.sm.skip_takeoff()

    def step(
        self, stamp_s: float, detections: list[Detection], frame_size: tuple[int, int]
    ) -> StepOutput | None:
        if self.e_stopped:
            return None
        return self.sm.step(stamp_s, detections, self.sensors.snapshot(stamp_s), frame_size)

    def hover(self, reason: str) -> FlightCommand:
        """Zero-velocity command; the mission state is left untouched so it can resume."""
        return FlightCommand(phase=self.sm.phase, action=reason)

    def on_armed(self, armed: bool, now_s: float) -> bool:
        """The match clock starts on the ARM rising edge (rulebook). Returns True on it."""
        rising = armed and not self._armed
        self._armed = armed
        if rising:
            self.sm.start_match(now_s)
        return rising

    def on_landed_state(self, landed_state: int) -> bool:
        """Touchdown (ON_GROUND rising edge) during the helipad phases awards the landing.
        Returns True when it did. Touchdowns in other phases are not a scored landing."""
        on_ground = landed_state == MAV_LANDED_STATE_ON_GROUND
        rising = on_ground and self._on_ground is False
        self._on_ground = on_ground
        if rising and self.sm.phase in _LANDING_PHASES:
            self.sm.signal_landed()
            return True
        return False

    def reset(self) -> None:
        """Retry (rulebook §VII): mission back to the start, e-stop released."""
        self.sm.reset()
        if self._skip_takeoff:
            self.sm.skip_takeoff()
        self.e_stopped = False
        self._armed = False

    def e_stop(self) -> None:
        self.e_stopped = True
