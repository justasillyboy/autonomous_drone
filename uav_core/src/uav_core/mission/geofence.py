"""Optical-flow dead reckoning and the soft geofence."""

from __future__ import annotations

from uav_core.config import GeofenceConfig, RulebookConfig
from uav_core.frames import cos_sin_deg, rotate, rotate_inverse


class DeadReckoning:
    """Integrated position from the takeoff point. World x is the nose direction when the
    drone first became active (yaw_0), not magnetic north."""

    def __init__(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0
        self.yaw0_deg: float | None = None

    def _cos_sin(self, yaw_deg: float) -> tuple[float, float]:
        return cos_sin_deg(yaw_deg - (self.yaw0_deg or 0.0))

    def integrate(self, of_vx: float, of_vy: float, yaw_deg: float | None, dt: float) -> None:
        if yaw_deg is not None:
            wvx, wvy = rotate(of_vx, of_vy, *self._cos_sin(yaw_deg))
        else:
            wvx, wvy = of_vx, of_vy
        self.x_m += wvx * dt
        self.y_m += wvy * dt

    def anchor(self, x_m: float, y_m: float) -> None:
        self.x_m, self.y_m = x_m, y_m

    def world_to_body(self, wx: float, wy: float, yaw_deg: float | None) -> tuple[float, float]:
        """Rotate a world-frame vector into the body frame (unchanged without yaw)."""
        if yaw_deg is None:
            return wx, wy
        return rotate_inverse(wx, wy, *self._cos_sin(yaw_deg))


def outside_fence(odom: DeadReckoning, cfg: GeofenceConfig, rules: RulebookConfig) -> bool:
    x_max = rules.field_length_m - cfg.margin_m
    y_max = rules.field_width_m / 2.0 - cfg.margin_m
    return odom.x_m < cfg.x_min_m or odom.x_m > x_max or abs(odom.y_m) > y_max


def return_velocity(
    odom: DeadReckoning, yaw_deg: float | None, cfg: GeofenceConfig
) -> tuple[float, float]:
    """Body-frame velocity back toward the takeoff point; the magnitude is capped without
    changing the direction (clipping each axis would bend it to 45 degrees)."""
    bx, by = odom.world_to_body(-odom.x_m * cfg.return_gain, -odom.y_m * cfg.return_gain, yaw_deg)
    mag = (bx**2 + by**2) ** 0.5 + 1e-9
    scale = min(1.0, cfg.max_speed_ms / mag)
    return float(bx * scale), float(by * scale)
