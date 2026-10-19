"""FIND_HELIPAD / LANDING: home on the H_landing pad, scan for it, or descend blind."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from uav_core.mission.context import MissionContext, clear_smoothing, command
from uav_core.mission.timers import below, exceeded, reached, scan_step_index
from uav_core.types import Detection, FlightCommand, Phase, SensorSnapshot, TargetClass

_log = logging.getLogger(__name__)


def helipad_action(
    d_ratio: float, land_ratio: float, approach_ratio: float, align_ratio: float
) -> str:
    if d_ratio >= land_ratio:
        return "LAND"
    if d_ratio >= approach_ratio:
        return "APPROACH"
    if d_ratio >= align_ratio:
        return "ALIGN"
    return "SEARCH"


def _approach(
    ctx: MissionContext, pad: Detection, frame_w: int, frame_h: int, t: float
) -> FlightCommand:
    hp, c = ctx.cfg.helipad, ctx.cfg.control
    x1, y1, x2, y2 = pad.bbox
    d_ratio = float(x2 - x1) / float(frame_w)
    action = helipad_action(d_ratio, hp.land_ratio, hp.approach_ratio, hp.align_ratio)
    vel_x = ctx.pid_hx.update(float((x1 + x2) // 2 - frame_w // 2), timestamp=t)
    vel_y = ctx.pid_hy.update(float((y1 + y2) // 2 - frame_h // 2), timestamp=t)
    vx = float(np.clip(-vel_y * c.px_to_ms_helipad, -hp.max_speed_ms, hp.max_speed_ms))
    vy = float(np.clip(vel_x * c.px_to_ms_helipad, -hp.max_speed_ms, hp.max_speed_ms))
    if d_ratio >= hp.land_ratio:
        if ctx.phase is not Phase.LANDING:
            # Visual fix over the pad: the pad is in the Safe Zone, so reset the drifting
            # dead reckoning to its centre.
            ctx.odom.anchor(ctx.rules.safe_zone_depth_m / 2.0, 0.0)
            ctx.geofence.reset()
            _log.info("helipad in range (d=%.2f): LANDING", d_ratio)
        ctx.phase = Phase.LANDING
        vz = hp.land_vz_ms
    else:
        ctx.phase = Phase.FIND_HELIPAD
        vz = hp.approach_vz_ms
    ctx.no_detect.reset()
    return command(ctx, action, vx=vx, vy=vy, vz=vz, yaw_rate_dps=0.0)


def helipad_step(
    ctx: MissionContext,
    detections: Sequence[Detection],
    sensors: SensorSnapshot,
    frame_w: int,
    frame_h: int,
    t: float,
    dt: float,
) -> FlightCommand:
    hp, timing = ctx.cfg.helipad, ctx.cfg.timing
    # Clock expiry overrides everything, pad visible or not.
    if below(ctx.rules.match_time_s - ctx.elapsed_s, timing.helipad_force_land_s):
        return command(ctx, "FORCE_LAND_CLOCK", vz=hp.force_land_vz_ms)
    pad = next((d for d in detections if d.cls is TargetClass.H_LANDING), None)
    if pad is not None:
        return _approach(ctx, pad, frame_w, frame_h, t)

    held = ctx.no_detect.tick(dt)
    # Far out in the field the downward camera cannot see the pad: head home meanwhile.
    far = ctx.odom.x_m > ctx.rules.safe_zone_depth_m + hp.rth_margin_m
    world_vx = -hp.rth_speed_ms if far else 0.0
    bx, by = ctx.odom.world_to_body(world_vx, 0.0, sensors.yaw_deg)
    suffix = "_RTH" if world_vx else ""
    if not reached(held, timing.no_detect_timeout_s):
        return command(ctx, "NO_DETECT" + suffix, vx=bx, vy=by, vz=0.0)
    clear_smoothing(ctx)
    ctx.phase = Phase.FIND_HELIPAD
    if exceeded(held, timing.helipad_scan_exhausted_s):
        return command(ctx, "BLIND_DESCENT" + suffix, vx=bx, vy=by, vz=hp.blind_descent_vz_ms)
    step = scan_step_index(
        held - timing.no_detect_timeout_s, timing.scan_step_s, len(timing.scan_yaw_steps_dps)
    )
    return command(
        ctx,
        f"SCAN_STEP_{step}" + suffix,
        vx=bx,
        vy=by,
        vz=0.0,
        yaw_rate_dps=float(timing.scan_yaw_steps_dps[step]),
        yaw_lock=False,
    )
