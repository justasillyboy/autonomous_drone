"""Release conditions: attitude gate and the ball's predicted impact point."""

from __future__ import annotations

import numpy as np

from uav_core.control.feedforward import MIN_ALTITUDE_M, fall_time_s

_MAX_VEL_PX = 200.0  # clamp on PID output before predicting drift (first-frame spikes)


def tilt_within(pitch_deg: float | None, roll_deg: float | None, limit_deg: float) -> bool:
    return not (
        (pitch_deg is not None and abs(pitch_deg) > limit_deg)
        or (roll_deg is not None and abs(roll_deg) > limit_deg)
    )


def attitude_check(
    pitch_deg: float | None,
    roll_deg: float | None,
    limit_deg: float,
    base_pitch_deg: float,
    base_roll_deg: float,
) -> tuple[bool, str]:
    """(level, reason). The same limit applies to both axes; the base limits only decide
    whether the message says [RELAXED]."""
    level, reason = True, ""
    if pitch_deg is not None and abs(pitch_deg) > limit_deg:
        level = False
        reason = f"pitch={pitch_deg:+.1f}°>{limit_deg:.0f}°" + (
            " [RELAXED]" if limit_deg > base_pitch_deg else ""
        )
    if roll_deg is not None and abs(roll_deg) > limit_deg:
        level = False
        reason += f" roll={roll_deg:+.1f}°>{limit_deg:.0f}°" + (
            " [RELAXED]" if limit_deg > base_roll_deg else ""
        )
    return level, reason


def impact_offset(
    pred_x: float,
    pred_y: float,
    vel_x: float,
    vel_y: float,
    altitude_m: float,
    px_to_ms: float,
    f_px: float,
) -> tuple[float, float]:
    """Pixel offset of where the ball lands: it inherits the commanded velocity
    (cmd_vy = vel_x * px_to_ms moves the image right, cmd_vx = -vel_y * px_to_ms down)."""
    scale = px_to_ms * f_px * fall_time_s(altitude_m) / max(altitude_m, MIN_ALTITUDE_M)
    drift_x = float(np.clip(vel_x, -_MAX_VEL_PX, _MAX_VEL_PX)) * scale
    drift_y = -float(np.clip(vel_y, -_MAX_VEL_PX, _MAX_VEL_PX)) * scale
    return pred_x - drift_x, pred_y - drift_y
