"""TARGETING / DROPPING: fly over the locked circle and release when every gate passes."""

from __future__ import annotations

import numpy as np

from uav_core.config import FireTarget
from uav_core.control.drop_gate import attitude_check, impact_offset, tilt_within
from uav_core.control.feedforward import add_feedforward, fall_time_s, is_maneuvering
from uav_core.mission.context import MissionContext, command, smooth_target
from uav_core.mission.phases.drop import execute_drop
from uav_core.mission.phases.search import TargetLock
from uav_core.mission.timers import reached
from uav_core.types import FlightCommand, Phase, Round, SensorSnapshot

_MIN_FF_SAMPLES = 4


def _vertical_command(ctx: MissionContext, lock: TargetLock) -> float:
    d, c = ctx.cfg.drop, ctx.cfg.control
    if lock.edge_clipped or d.altitude_min_m <= lock.z_drop <= d.altitude_max_m:
        return 0.0  # hold: in the window, or Z untrustworthy
    return c.descend_ms if lock.z_drop > d.altitude_max_m else -c.ascend_ms


def track_target(
    ctx: MissionContext,
    target: FireTarget,
    lock: TargetLock,
    sensors: SensorSnapshot,
    frame_w: int,
    frame_h: int,
    t: float,
    dt: float,
) -> FlightCommand:
    cfg = ctx.cfg
    d, c, ff = cfg.drop, cfg.control, cfg.feedforward
    cx, cy = smooth_target(ctx, lock.cx, lock.cy, lock.z_drop, t)
    offset_x = cx - frame_w // 2
    offset_y = cy - frame_h // 2
    pitch, roll = sensors.pitch_deg, sensors.roll_deg

    # Finals: aim at where F2 will be when the ball lands, before the PID runs, so the PID
    # and the drop trigger chase the same point (otherwise they can never both be satisfied).
    ff_vx = ff_vy = 0.0
    pred_x, pred_y = float(offset_x), float(offset_y)
    if ctx.round is Round.FINALS:
        maneuvering = is_maneuvering(
            sensors.pitch_rate_dps,
            sensors.roll_rate_dps,
            offset_x,
            offset_y,
            ctx.pid_x.kp,
            c.px_to_ms_fire,
            ff.angular_rate_gate_dps,
            ff.cmd_threshold_ms,
        )
        if not maneuvering and len(ctx.motion) >= _MIN_FF_SAMPLES:
            ff_vx, ff_vy = ctx.motion.velocity(ff.max_px_s)
            t_fall = fall_time_s(lock.z_drop)
            pred_x = offset_x + ff_vx * t_fall
            pred_y = offset_y + ff_vy * t_fall

    aligned = abs(pred_x) < d.align_pixel_thresh_px and abs(pred_y) < d.align_pixel_thresh_px
    altitude_safe = d.altitude_min_m <= lock.z_drop <= d.altitude_max_m and not lock.tof_low_blocks

    # Wind: aligned and safe but tilted for tilt_relax_after_s -> accept up to the relaxed
    # limit rather than never dropping.
    limit = d.max_pitch_deg
    if altitude_safe and aligned and not tilt_within(pitch, roll, d.max_pitch_deg):
        if reached(ctx.tilt_block.tick(dt), d.tilt_relax_after_s):
            limit = d.tilt_max_relaxed_deg
    else:
        ctx.tilt_block.reset()
    level, reason = attitude_check(pitch, roll, limit, d.max_pitch_deg, d.max_roll_deg)

    # Integral frozen while waiting for level attitude: no wind-up overshoot afterwards.
    vel_x = ctx.pid_x.update(pred_x, timestamp=t, freeze_integral=not level)
    vel_y = ctx.pid_y.update(pred_y, timestamp=t, freeze_integral=not level)

    impact_x, impact_y = impact_offset(
        pred_x, pred_y, vel_x, vel_y, lock.z_drop, c.px_to_ms_fire, cfg.camera.f_px
    )
    aligned_for_drop = (
        abs(impact_x) < d.align_pixel_thresh_px and abs(impact_y) < d.align_pixel_thresh_px
    )
    cooled_down = ctx.last_drop_t is None or reached(t - ctx.last_drop_t, d.cooldown_s)
    trigger = altitude_safe and not lock.edge_clipped and aligned_for_drop and level and cooled_down
    if trigger:
        ctx.phase = Phase.DROPPING
        execute_drop(ctx, target, lock.z_drop, cx, cy, frame_w, frame_h, t)
    else:
        ctx.phase = Phase.TARGETING

    cmd_vx = float(np.clip(-vel_y * c.px_to_ms_fire, -c.max_speed_ms, c.max_speed_ms))
    cmd_vy = float(np.clip(vel_x * c.px_to_ms_fire, -c.max_speed_ms, c.max_speed_ms))
    cmd_vz = _vertical_command(ctx, lock)
    if ctx.round is Round.FINALS and (ff_vx != 0.0 or ff_vy != 0.0):
        cmd_vx, cmd_vy = add_feedforward(
            cmd_vx, cmd_vy, ff_vx, ff_vy, ff.scale, ff.max_add_ms, c.max_speed_ms
        )
    # Coast with the full command (after feedforward) if the target is lost next step.
    ctx.coast_vx, ctx.coast_vy, ctx.coast_vz = cmd_vx, cmd_vy, cmd_vz

    if lock.edge_clipped:
        action = "EDGE_CLIP_BLOCK"
    elif lock.tof_low_blocks:
        action = f"TOF_BLOCK({sensors.tof_alt_m:.2f}m<{d.altitude_min_m}m)"
    elif trigger:
        action = "DROP"
    elif not altitude_safe:
        action = "DESCEND" if lock.z_drop > d.altitude_max_m else "ASCEND"
    elif not level:
        action = f"WAIT_LEVEL({reason})"
    else:
        action = "ALIGN"
    return command(
        ctx, action, vx=cmd_vx, vy=cmd_vy, vz=cmd_vz, yaw_rate_dps=0.0, drop_ball=trigger
    )
