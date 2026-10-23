"""Randomised scenarios for Monte Carlo runs: one seed -> one reproducible match. No ROS."""

from __future__ import annotations

import random
from dataclasses import dataclass

from uav_core.types import Round

from uav_sim.field import Layout, qualifier_layout, random_layout


@dataclass(frozen=True)
class Scenario:
    seed: int
    layout: Layout
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_intensity: float
    floor_rgb: tuple[float, float, float]
    wind_speed_ms: float
    wind_dir_deg: float
    sitl_params: dict[str, float]

    @property
    def name(self) -> str:
        return f"{self.layout.round.value}-{self.seed:05d}"


def make_scenario(
    seed: int, round_: Round = Round.QUALIFIER, fixed_layout: bool = True
) -> Scenario:
    """Qualifier targets sit where figure 1 puts them unless fixed_layout is False; the
    semi-final and final draw them at random. Lighting, floor tint, wind and sensor noise
    are always drawn from the seed."""
    rng = random.Random(seed)
    if round_ is Round.QUALIFIER and fixed_layout:
        layout = qualifier_layout()
    else:
        layout = random_layout(seed, round_)
    grey = rng.uniform(0.22, 0.38)  # dark court floor, slightly tinted
    tint = (grey + rng.uniform(-0.03, 0.03), grey, grey + rng.uniform(-0.03, 0.05))
    wind = round(rng.uniform(0.0, 3.0), 2)
    wind_dir = round(rng.uniform(0.0, 360.0), 1)
    return Scenario(
        seed=seed,
        layout=layout,
        sun_azimuth_deg=rng.uniform(0.0, 360.0),
        sun_elevation_deg=rng.uniform(35.0, 85.0),
        sun_intensity=rng.uniform(0.6, 1.2),
        floor_rgb=tuple(round(c, 3) for c in tint),
        wind_speed_ms=wind,
        wind_dir_deg=wind_dir,
        sitl_params={
            "SIM_WIND_SPD": wind,
            "SIM_WIND_DIR": wind_dir,
            "SIM_WIND_TURB": round(rng.uniform(0.0, 0.5), 2),
            "SIM_BARO_RND": round(rng.uniform(0.05, 0.3), 3),
            "SIM_GYR_RND": round(rng.uniform(0.0, 0.05), 3),
            "SIM_ACC_RND": round(rng.uniform(0.0, 0.05), 3),
        },
    )


# The drop servo on AUX channel 9 is driven only by MAV_CMD_DO_SET_SERVO (function 0).
BASE_SITL_PARAMS = {"SERVO9_FUNCTION": 0}


def sitl_param_file(s: Scenario) -> str:
    """ArduPilot .parm text with this scenario's wind and noise, plus the base settings."""
    params = {**BASE_SITL_PARAMS, **s.sitl_params}
    return "".join(f"{k} {v}\n" for k, v in sorted(params.items()))
