"""Ball release bookkeeping."""

from __future__ import annotations

import logging

from uav_core.config import FireTarget
from uav_core.mission.context import MissionContext, clear_smoothing, enter_helipad
from uav_core.mission.scoring import pixel_to_world
from uav_core.types import Phase

_log = logging.getLogger(__name__)


def execute_drop(
    ctx: MissionContext,
    target: FireTarget,
    altitude_m: float,
    cx: int,
    cy: int,
    frame_w: int,
    frame_h: int,
    t: float,
) -> None:
    """Score the drop, remember where the circle is, and move to the next target."""
    ctx.balls_dropped += 1
    ctx.score += target.points
    ctx.last_drop_t = t
    ctx.tilt_block.reset()
    ctx.pid_x.reset()
    ctx.pid_y.reset()
    if cx >= 0 and cy >= 0 and frame_w > 0 and frame_h > 0:
        ctx.scored.add(
            *pixel_to_world(cx, cy, frame_w // 2, frame_h // 2, altitude_m, ctx.cfg.camera.f_px)
        )
    _log.info(
        "drop %s at Z=%.2f m (+%d) -> %d/%d, score %d",
        target.label,
        altitude_m,
        target.points,
        ctx.balls_dropped,
        len(ctx.targets),
        ctx.score,
    )
    if ctx.balls_dropped >= len(ctx.targets):
        enter_helipad(ctx)
    else:
        ctx.target_idx += 1
        ctx.phase = Phase.SEARCHING
        clear_smoothing(ctx)
        ctx.motion.clear()
