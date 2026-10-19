"""SEARCHING / TARGETING front half: pick the circle to fly to, or coast when it is lost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uav_core.config import FireTarget
from uav_core.mission.context import MissionContext, clear_smoothing, command
from uav_core.mission.timers import reached
from uav_core.perception.altitude_vision import altitude_from_diameter
from uav_core.perception.geometry import diameter_ratio
from uav_core.types import AltSource, Color, Detection, FlightCommand, Phase, TargetClass


@dataclass(frozen=True, slots=True)
class TargetLock:
    z_drop: float  # altitude used for the drop decision
    z_source: AltSource
    cx: int
    cy: int
    radius: int
    color: Color
    tof_low_blocks: bool  # ToF says below the drop window
    edge_clipped: bool  # radius untrustworthy: no drop, hold altitude


def select_target(
    ctx: MissionContext,
    target: FireTarget,
    detections: Sequence[Detection],
    frame_w: int,
    frame_h: int,
    altitude_m: float | None,
    alt_source: AltSource,
    tof_alt_m: float | None,
) -> TargetLock | None:
    """The target-coloured circle closest to the frame centre, or None.

    Legacy quirks kept on purpose (fixed later, with a golden update): the vision altitude
    memory `last_z_est` is updated for every unclipped circle before the colour check, using
    the current target's diameter; the scored-zone check uses the Z of the previous circle.
    """
    cfg = ctx.cfg
    f_px = cfg.camera.f_px
    frame_cx, frame_cy = frame_w // 2, frame_h // 2
    z_est = ctx.last_z_est if ctx.last_z_est is not None else cfg.drop.altitude_target_m
    best: TargetLock | None = None
    best_offset = 0
    for det in detections:
        if det.cls is not TargetClass.CIRCLE:
            continue
        if diameter_ratio(det.radius, frame_w) > cfg.drop.max_fire_diameter_ratio:
            continue
        if ctx.scored.contains(det.cx, det.cy, z_est, frame_cx, frame_cy, f_px):
            continue
        z_est = altitude_from_diameter(det.radius * 2.0, target.diameter_m, f_px)
        if det.edge_clipped:
            z_est = ctx.last_z_est if ctx.last_z_est is not None else cfg.drop.altitude_target_m
        else:
            ctx.last_z_est = z_est
        if altitude_m is not None:
            z_drop, z_src = altitude_m, alt_source
        else:
            z_drop, z_src = z_est, AltSource.VIS
        tof_low = tof_alt_m is not None and tof_alt_m < cfg.drop.altitude_min_m
        if det.color != target.color:
            continue
        offset = (det.cx - frame_cx) ** 2 + (det.cy - frame_cy) ** 2
        if best is None or offset < best_offset:
            best = TargetLock(
                z_drop,
                z_src,
                det.cx,
                det.cy,
                det.radius,
                det.color,
                tof_low and not det.edge_clipped,
                det.edge_clipped,
            )
            best_offset = offset
    return best


def on_target_missing(ctx: MissionContext, dt: float) -> FlightCommand:
    """Hold the last velocity (zero-order hold) for no_detect_timeout_s, then RECOVERY.
    Braking would pitch the camera up and lose a target hidden only by glare."""
    if reached(ctx.no_detect.tick(dt), ctx.cfg.timing.no_detect_timeout_s):
        clear_smoothing(ctx)
        ctx.pid_x.reset()
        ctx.pid_y.reset()
        ctx.coast_vx = ctx.coast_vy = ctx.coast_vz = 0.0
        ctx.recovery.reset()
        ctx.phase = Phase.RECOVERY
        return command(ctx, "ENTERING_RECOVERY")
    return command(ctx, "ZOH_COAST", vx=ctx.coast_vx, vy=ctx.coast_vy, vz=ctx.coast_vz)
