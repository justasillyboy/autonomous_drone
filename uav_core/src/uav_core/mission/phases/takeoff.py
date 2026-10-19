"""TAKEOFF: climb until the altitude source confirms takeoff_alt_m."""

from __future__ import annotations

from uav_core.mission.context import MissionContext, command
from uav_core.types import FlightCommand, Phase


def takeoff_step(ctx: MissionContext, altitude_m: float | None) -> FlightCommand:
    if altitude_m is not None and altitude_m >= ctx.cfg.control.takeoff_alt_m:
        ctx.phase = Phase.SEARCHING
        if not ctx.takeoff_scored:
            ctx.score += ctx.rules.takeoff_bonus(ctx.round)
            ctx.takeoff_scored = True
        return command(ctx, "REACHED_ALT", vz=0.0)
    # Without an altitude reading, keep asking to climb and trust the FC's auto-takeoff.
    return command(ctx, "ASCEND", vz=-ctx.cfg.control.takeoff_ascent_ms)
