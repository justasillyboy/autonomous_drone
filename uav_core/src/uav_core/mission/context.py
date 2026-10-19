"""The mission's mutable state, owned by one MissionStateMachine, plus shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from uav_core.config import FireTarget, MissionConfig, RulebookConfig
from uav_core.control.pid import PIDController
from uav_core.estimation.target_motion import TargetMotionHistory
from uav_core.estimation.tof_filter import TofMedianFilter
from uav_core.mission.geofence import DeadReckoning
from uav_core.mission.scoring import ScoredExclusion
from uav_core.mission.timers import HoldTimer
from uav_core.types import AltSource, FlightCommand, Phase, Round


@dataclass(slots=True)
class MissionContext:
    cfg: MissionConfig
    rules: RulebookConfig
    round: Round
    targets: tuple[FireTarget, ...]
    pid_x: PIDController
    pid_y: PIDController
    pid_hx: PIDController
    pid_hy: PIDController
    tof: TofMedianFilter
    motion: TargetMotionHistory
    scored: ScoredExclusion
    alpha: float
    phase: Phase = Phase.TAKEOFF
    target_idx: int = 0
    balls_dropped: int = 0
    score: int = 0
    penalties: int = 0
    takeoff_scored: bool = False
    landing_scored: bool = False
    # clock
    last_t: float | None = None
    match_start_t: float | None = None
    elapsed_s: float = 0.0
    timer_forced_land: bool = False
    last_drop_t: float | None = None
    # timers (seconds a condition has held)
    no_detect: HoldTimer = field(default_factory=HoldTimer)
    recovery: HoldTimer = field(default_factory=HoldTimer)
    tilt_block: HoldTimer = field(default_factory=HoldTimer)
    geofence: HoldTimer = field(default_factory=HoldTimer)
    low_alt: HoldTimer = field(default_factory=HoldTimer)
    low_alt_penalties: int = 0
    # target tracking memory
    last_z_est: float | None = None
    smooth_cx: float | None = None
    smooth_cy: float | None = None
    smooth_z: float | None = None
    coast_vx: float = 0.0
    coast_vy: float = 0.0
    coast_vz: float = 0.0
    prev_yaw_deg: float | None = None
    odom: DeadReckoning = field(default_factory=DeadReckoning)
    alt_source: AltSource = AltSource.VIS
    last_command: FlightCommand = field(
        default_factory=lambda: FlightCommand(phase=Phase.TAKEOFF, action="ASCEND")
    )

    @classmethod
    def create(cls, cfg: MissionConfig, rules: RulebookConfig, round_: Round) -> MissionContext:
        c = cfg.control
        fire_gains = c.pid_finals if round_ is Round.FINALS else c.pid_qualifier
        alpha = c.smoothing_alpha_finals if round_ is Round.FINALS else c.smoothing_alpha_qualifier
        e = cfg.estimation
        return cls(
            cfg=cfg,
            rules=rules,
            round=round_,
            targets=rules.targets(round_),
            pid_x=PIDController(fire_gains, c.pid_output_limit),
            pid_y=PIDController(fire_gains, c.pid_output_limit),
            pid_hx=PIDController(c.pid_helipad, c.pid_output_limit),
            pid_hy=PIDController(c.pid_helipad, c.pid_output_limit),
            tof=TofMedianFilter(e.tof_window, e.tof_spike_threshold_m, e.tof_spike_reset_count),
            motion=TargetMotionHistory(cfg.feedforward.history_len),
            scored=ScoredExclusion(cfg.drop.scored_world_radius_m),
            alpha=alpha,
        )


def current_target(ctx: MissionContext) -> FireTarget | None:
    return ctx.targets[ctx.target_idx] if ctx.target_idx < len(ctx.targets) else None


def command(ctx: MissionContext, action: str, **kwargs: float | bool) -> FlightCommand:
    """A FlightCommand stamped with the phase as it is *now* (after any transition)."""
    return FlightCommand(phase=ctx.phase, action=action, **kwargs)


def clear_smoothing(ctx: MissionContext) -> None:
    ctx.smooth_cx = ctx.smooth_cy = ctx.smooth_z = None


def reset_tracking(ctx: MissionContext) -> None:
    """Forget the current target: EMA, motion history and the fire PIDs."""
    clear_smoothing(ctx)
    ctx.motion.clear()
    ctx.pid_x.reset()
    ctx.pid_y.reset()


def enter_helipad(ctx: MissionContext) -> None:
    ctx.phase = Phase.FIND_HELIPAD
    ctx.no_detect.reset()
    clear_smoothing(ctx)
    ctx.pid_hx.reset()
    ctx.pid_hy.reset()


def smooth_target(ctx: MissionContext, cx: int, cy: int, z: float, t: float) -> tuple[int, int]:
    """EMA with alpha weighting the OLD value (legacy convention); records the history."""
    a = ctx.alpha
    if ctx.smooth_cx is None or ctx.smooth_cy is None or ctx.smooth_z is None:
        ctx.smooth_cx, ctx.smooth_cy, ctx.smooth_z = float(cx), float(cy), z
    else:
        ctx.smooth_cx = a * ctx.smooth_cx + (1 - a) * cx
        ctx.smooth_cy = a * ctx.smooth_cy + (1 - a) * cy
        ctx.smooth_z = a * ctx.smooth_z + (1 - a) * z
    ctx.motion.append(ctx.smooth_cx, ctx.smooth_cy, t)
    return int(ctx.smooth_cx), int(ctx.smooth_cy)
