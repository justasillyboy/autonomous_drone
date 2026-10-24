"""The mission state machine: detections + sensors in, a FlightCommand out. No images."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

import numpy as np

from uav_core.config import MissionConfig, RulebookConfig
from uav_core.control.feedforward import MIN_ALTITUDE_M
from uav_core.estimation.altitude import resolve_altitude
from uav_core.frames import cos_sin_deg, wrap_deg
from uav_core.mission.context import (
    MissionContext,
    command,
    current_target,
    enter_helipad,
)
from uav_core.mission.geofence import outside_fence, return_velocity
from uav_core.mission.phases.helipad import helipad_step
from uav_core.mission.phases.recovery import recovery_step
from uav_core.mission.phases.search import on_target_missing, select_target
from uav_core.mission.phases.takeoff import takeoff_step
from uav_core.mission.phases.targeting import track_target
from uav_core.mission.timers import reached
from uav_core.types import (
    AltSource,
    Detection,
    FlightCommand,
    MissionStatus,
    Phase,
    Round,
    SensorSnapshot,
    StepOutput,
)

_log = logging.getLogger(__name__)
_ODOM_DT_RANGE_S = (1e-3, 0.5)  # survive a stall or a breakpoint without corrupting odometry
_MIN_YAW_STEP_DEG = 0.05  # smaller yaw changes are IMU noise
_INACTIVE = (Phase.TAKEOFF, Phase.COMPLETE)
_FIRE = (Phase.SEARCHING, Phase.TARGETING, Phase.DROPPING)


class MissionStateMachine:
    """Call step() once per perception frame, with timestamps in seconds (sim time works).

    Time-based windows (no-detection timeout, recovery scan, drop cooldown, tilt relaxation,
    altitude penalty, geofence) are timers on t_s, not frame counts (ADR-0001).
    """

    def __init__(
        self,
        cfg: MissionConfig,
        rules: RulebookConfig | None = None,
        round_: Round = Round.QUALIFIER,
    ):
        self._cfg = cfg
        self._rules = rules if rules is not None else RulebookConfig()
        self._round = round_
        self._ctx = MissionContext.create(cfg, self._rules, round_)

    # -- control ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Retry (rulebook §VII): everything back to TAKEOFF, scores to zero."""
        self._ctx = MissionContext.create(self._cfg, self._rules, self._round)
        self._ctx.last_command = command(self._ctx, "RETRY_RESET")

    def skip_takeoff(self) -> None:
        """Start in SEARCHING when the FC took off by itself (no takeoff bonus)."""
        if self._ctx.phase is Phase.TAKEOFF:
            self._ctx.phase = Phase.SEARCHING

    def start_match(self, t_s: float) -> None:
        """Start the match clock (mission_node calls this on the ARM rising edge). Without it
        the clock starts on the first step outside TAKEOFF, as in the legacy code."""
        if self._ctx.match_start_t is None:
            self._ctx.match_start_t = t_s

    def signal_landed(self) -> None:
        """The FC confirmed touchdown: award the landing bonus once and finish."""
        ctx = self._ctx
        if not ctx.landing_scored:
            ctx.score += self._rules.landing_bonus(self._round)
            ctx.landing_scored = True
        ctx.phase = Phase.COMPLETE
        ctx.last_command = command(ctx, "LANDED")

    # -- queries ---------------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._ctx.phase

    @property
    def last_command(self) -> FlightCommand:
        return self._ctx.last_command

    def status(self) -> MissionStatus:
        ctx = self._ctx
        return MissionStatus(
            phase=ctx.phase,
            round=self._round,
            score=ctx.score,
            penalties=ctx.penalties,
            balls_dropped=ctx.balls_dropped,
            target_index=ctx.target_idx,
            alt_source=ctx.alt_source,
            elapsed_s=ctx.elapsed_s,
        )

    # -- the step --------------------------------------------------------------------------

    def step(
        self,
        t_s: float,
        detections: Sequence[Detection],
        sensors: SensorSnapshot,
        frame_size: tuple[int, int],
    ) -> StepOutput:
        frame_w, frame_h = frame_size
        dt, odom_dt = self._advance_clock(t_s)
        self._update_match_clock(t_s)
        sensors = self._filter_tof(sensors)
        altitude_m, alt_source = resolve_altitude(sensors.fc_alt_m, sensors.tof_alt_m)
        self._ctx.alt_source = alt_source
        self._compensate_motion(sensors, odom_dt, frame_w, frame_h)
        self._track_altitude_penalty(altitude_m, dt)
        cmd = self._route(t_s, dt, detections, sensors, altitude_m, alt_source, frame_w, frame_h)
        if cmd is not None:
            self._ctx.last_command = cmd
        fence_cmd = self._geofence(sensors, dt)
        if fence_cmd is not None:
            self._ctx.last_command = fence_cmd
        return StepOutput(self._ctx.last_command, self.status())

    def _advance_clock(self, t_s: float) -> tuple[float, float]:
        """(timer dt, odometry dt). Timers never run backwards; odometry dt is clamped."""
        ctx = self._ctx
        raw = self._cfg.timing.first_step_dt_s if ctx.last_t is None else t_s - ctx.last_t
        ctx.last_t = t_s
        return max(raw, 0.0), float(np.clip(raw, *_ODOM_DT_RANGE_S))

    def _update_match_clock(self, t_s: float) -> None:
        ctx = self._ctx
        if ctx.match_start_t is None and ctx.phase not in _INACTIVE:
            ctx.match_start_t = t_s
        if ctx.match_start_t is None:
            return
        ctx.elapsed_s = t_s - ctx.match_start_t
        if (
            not ctx.timer_forced_land
            and reached(ctx.elapsed_s, self._cfg.timing.land_early_at_s)
            and ctx.phase not in (Phase.FIND_HELIPAD, Phase.LANDING, Phase.COMPLETE)
        ):
            _log.info("%.0f s elapsed: forcing FIND_HELIPAD to secure the landing", ctx.elapsed_s)
            ctx.timer_forced_land = True
            enter_helipad(ctx)

    def _filter_tof(self, sensors: SensorSnapshot) -> SensorSnapshot:
        """Median/spike filter, then the mounting offset: the rulebook's 3.0 m is measured
        at the drone's lowest point, while the rangefinder sits above it."""
        if sensors.tof_alt_m is None:
            return sensors
        filtered = self._ctx.tof.update(sensors.tof_alt_m)
        if filtered is not None:  # None: a rejected spike
            filtered -= self._cfg.estimation.tof_ground_offset_m
        if filtered == sensors.tof_alt_m:
            return sensors
        return dataclasses.replace(sensors, tof_alt_m=filtered)

    def _compensate_motion(self, s: SensorSnapshot, dt: float, frame_w: int, frame_h: int) -> None:
        """Keep scored zones, the target history and dead reckoning consistent with the
        drone's own translation (optical flow) and yaw since the previous step."""
        ctx = self._ctx
        rotation: tuple[float, float] | None = None
        if s.yaw_deg is not None and ctx.prev_yaw_deg is not None:
            d_psi = wrap_deg(s.yaw_deg - ctx.prev_yaw_deg)
            if abs(d_psi) > _MIN_YAW_STEP_DEG:
                rotation = cos_sin_deg(d_psi)
        has_flow = s.of_vx_ms is not None and s.of_vy_ms is not None
        if len(ctx.scored):
            if has_flow:
                # Image x right / y down: lateral +vy shifts objects left, forward +vx down.
                ctx.scored.translate(-(s.of_vy_ms * dt), +(s.of_vx_ms * dt))
            if rotation is not None:
                ctx.scored.rotate(*rotation)
        if s.yaw_deg is not None:
            ctx.prev_yaw_deg = s.yaw_deg
        if ctx.odom.yaw0_deg is None and ctx.phase not in _INACTIVE and s.yaw_deg is not None:
            ctx.odom.yaw0_deg = s.yaw_deg
        if len(ctx.motion) and has_flow:
            z_ref = max(ctx.smooth_z or self._cfg.drop.altitude_target_m, MIN_ALTITUDE_M)
            px_per_m = self._cfg.camera.f_px / z_ref
            ctx.motion.shift(-s.of_vy_ms * px_per_m * dt, +s.of_vx_ms * px_per_m * dt)
        if len(ctx.motion) and rotation is not None:
            ctx.motion.rotate_about(frame_w / 2.0, frame_h / 2.0, *rotation)
        if has_flow:
            ctx.odom.integrate(s.of_vx_ms, s.of_vy_ms, s.yaw_deg, dt)

    def _track_altitude_penalty(self, altitude_m: float | None, dt: float) -> None:
        """Rulebook §VII: below 3.0 m in autonomous flight costs 5 points per 5 s."""
        ctx, rules = self._ctx, self._rules
        if (
            ctx.phase not in _INACTIVE
            and altitude_m is not None
            and altitude_m < rules.alt_penalty_floor_m
        ):
            held = ctx.low_alt.tick(dt)
            while reached(held, (ctx.low_alt_penalties + 1) * rules.alt_penalty_after_s):
                ctx.low_alt_penalties += 1
                ctx.penalties += rules.penalty_altitude_pts
                _log.info(
                    "altitude %.2f m below %.1f m: penalty %d (total %d)",
                    altitude_m,
                    rules.alt_penalty_floor_m,
                    rules.penalty_altitude_pts,
                    ctx.penalties,
                )
        else:
            ctx.low_alt.reset()
            ctx.low_alt_penalties = 0

    def _route(
        self,
        t_s: float,
        dt: float,
        detections: Sequence[Detection],
        sensors: SensorSnapshot,
        altitude_m: float | None,
        alt_source: AltSource,
        frame_w: int,
        frame_h: int,
    ) -> FlightCommand | None:
        ctx = self._ctx
        if ctx.phase is Phase.TAKEOFF:
            return takeoff_step(ctx, altitude_m)
        if ctx.phase in _FIRE:
            target = current_target(ctx)
            if target is None:
                enter_helipad(ctx)
                return None  # legacy keeps the previous command for this step
            lock = select_target(
                ctx, target, detections, frame_w, frame_h, altitude_m, alt_source, sensors.tof_alt_m
            )
            if lock is None:
                return on_target_missing(ctx, dt)
            ctx.no_detect.reset()
            return track_target(ctx, target, lock, sensors, frame_w, frame_h, t_s, dt)
        if ctx.phase is Phase.RECOVERY:
            return recovery_step(ctx, detections, frame_w, frame_h, dt)
        if ctx.phase in (Phase.FIND_HELIPAD, Phase.LANDING):
            return helipad_step(ctx, detections, sensors, frame_w, frame_h, t_s, dt)
        return None  # COMPLETE

    def _geofence(self, sensors: SensorSnapshot, dt: float) -> FlightCommand | None:
        """Override any phase command after geofence_warn_s outside the soft fence."""
        ctx = self._ctx
        if ctx.phase in _INACTIVE or not outside_fence(ctx.odom, self._cfg.geofence, self._rules):
            ctx.geofence.reset()
            return None
        if not reached(ctx.geofence.tick(dt), self._cfg.timing.geofence_warn_s):
            return None
        vx, vy = return_velocity(ctx.odom, sensors.yaw_deg, self._cfg.geofence)
        return command(ctx, "GEOFENCE_RETURN", vx=vx, vy=vy, vz=0.0)
