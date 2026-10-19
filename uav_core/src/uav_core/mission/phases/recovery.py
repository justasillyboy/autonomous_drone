"""RECOVERY: yaw scan to find a lost target; skip it when the scan runs out."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from uav_core.mission.context import (
    MissionContext,
    command,
    current_target,
    enter_helipad,
    reset_tracking,
)
from uav_core.mission.timers import reached, scan_step_index
from uav_core.types import Detection, FlightCommand, Phase, TargetClass

_log = logging.getLogger(__name__)


def _target_visible(
    ctx: MissionContext, detections: Sequence[Detection], frame_w: int, frame_h: int
) -> bool:
    """Any unscored circle of the target colour. Unlike targeting, no size limit applies
    (legacy behaviour)."""
    target = current_target(ctx)
    if target is None:
        return False
    z = ctx.last_z_est if ctx.last_z_est is not None else ctx.cfg.drop.altitude_target_m
    for det in detections:
        if det.cls is not TargetClass.CIRCLE:
            continue
        if ctx.scored.contains(det.cx, det.cy, z, frame_w // 2, frame_h // 2, ctx.cfg.camera.f_px):
            continue
        if det.color == target.color:
            return True
    return False


def recovery_step(
    ctx: MissionContext, detections: Sequence[Detection], frame_w: int, frame_h: int, dt: float
) -> FlightCommand:
    timing = ctx.cfg.timing
    held = ctx.recovery.tick(dt)
    if _target_visible(ctx, detections, frame_w, frame_h):
        ctx.phase = Phase.TARGETING
        ctx.recovery.reset()
        ctx.no_detect.reset()
        reset_tracking(ctx)
        return command(ctx, "TARGET_REACQUIRED")
    if reached(held, timing.recovery_timeout_s):
        _log.info("recovery scan exhausted: skipping target %d", ctx.target_idx)
        ctx.recovery.reset()
        ctx.no_detect.reset()
        reset_tracking(ctx)
        if ctx.target_idx >= len(ctx.targets) - 1:
            enter_helipad(ctx)
            return command(ctx, "SKIP_LAST_TARGET")
        ctx.target_idx += 1
        ctx.phase = Phase.SEARCHING
        return command(ctx, "SKIP_TARGET")
    step = scan_step_index(held, timing.scan_step_s, len(timing.scan_yaw_steps_dps))
    return command(
        ctx,
        f"SCAN_STEP_{step}",
        yaw_rate_dps=float(timing.scan_yaw_steps_dps[step]),
        yaw_lock=False,
    )
