"""Finals feedforward for the moving F2 target: aim where it will be when the ball lands."""

from __future__ import annotations

import numpy as np

GRAVITY_MS2 = 9.81
MIN_ALTITUDE_M = 0.5  # floor used whenever an altitude divides or roots


def fall_time_s(altitude_m: float) -> float:
    return (2.0 * max(altitude_m, MIN_ALTITUDE_M) / GRAVITY_MS2) ** 0.5


def is_maneuvering(
    pitch_rate_dps: float | None,
    roll_rate_dps: float | None,
    offset_x: float,
    offset_y: float,
    kp: float,
    px_to_ms: float,
    rate_gate_dps: float,
    cmd_threshold_ms: float,
) -> bool:
    """While the drone is pitching or rolling, image motion is camera tilt, not the target.
    Uses IMU rates when both exist, else the speed the PID would roughly command."""
    if pitch_rate_dps is not None and roll_rate_dps is not None:
        return abs(pitch_rate_dps) > rate_gate_dps or abs(roll_rate_dps) > rate_gate_dps
    return (
        abs(float(np.clip(-kp * offset_y * px_to_ms, -1, 1))) > cmd_threshold_ms
        or abs(float(np.clip(kp * offset_x * px_to_ms, -1, 1))) > cmd_threshold_ms
    )


def add_feedforward(
    cmd_vx: float,
    cmd_vy: float,
    ff_vx_px: float,
    ff_vy_px: float,
    scale: float,
    max_add_ms: float,
    max_speed_ms: float,
) -> tuple[float, float]:
    """Add the target's velocity (image px/s -> body m/s) to the PID command."""
    cmd_vx += float(np.clip(-ff_vy_px * scale, -max_add_ms, max_add_ms))
    cmd_vy += float(np.clip(ff_vx_px * scale, -max_add_ms, max_add_ms))
    return (
        float(np.clip(cmd_vx, -max_speed_ms, max_speed_ms)),
        float(np.clip(cmd_vy, -max_speed_ms, max_speed_ms)),
    )
