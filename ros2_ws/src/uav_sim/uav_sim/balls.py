"""Ball release and ball-at-rest detection for the simulation. No ROS imports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

BALL_RADIUS_M = 0.0335  # tennis ball
BALL_MASS_KG = 0.057
REST_SPEED_MS = 0.02
REST_HOLD_S = 0.5
REST_MAX_Z_M = 0.25  # on the floor or on a 0.1 m target plinth


def ball_sdf(name: str) -> str:
    """SDF of one tennis ball (solid sphere inertia)."""
    i = 2.0 / 5.0 * BALL_MASS_KG * BALL_RADIUS_M**2
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <link name="link">
      <inertial>
        <mass>{BALL_MASS_KG}</mass>
        <inertia><ixx>{i:.3e}</ixx><iyy>{i:.3e}</iyy><izz>{i:.3e}</izz></inertia>
      </inertial>
      <collision name="collision">
        <geometry><sphere><radius>{BALL_RADIUS_M}</radius></sphere></geometry>
        <surface><bounce><restitution_coefficient>0.6</restitution_coefficient></bounce></surface>
      </collision>
      <visual name="visual">
        <geometry><sphere><radius>{BALL_RADIUS_M}</radius></sphere></geometry>
        <material><ambient>0.8 1 0.2 1</ambient><diffuse>0.8 1 0.2 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""


@dataclass
class DropTrigger:
    """A release is the servo PWM crossing `threshold` upward on the drop channel, which is
    exactly what action_node commands with MAV_CMD_DO_SET_SERVO."""

    threshold: int = 1500
    _high: bool = False

    def update(self, pwm: int) -> bool:
        high = pwm >= self.threshold
        rising = high and not self._high
        self._high = high
        return rising


@dataclass
class _Track:
    x: float
    y: float
    z: float
    t: float
    still_since: float | None = None
    reported: bool = False


@dataclass
class BallTracker:
    """Reports each ball once, when it has stayed still and low for REST_HOLD_S."""

    _tracks: dict[str, _Track] = field(default_factory=dict)

    def update(
        self, name: str, t_s: float, x: float, y: float, z: float
    ) -> tuple[float, float] | None:
        tr = self._tracks.get(name)
        if tr is None:
            self._tracks[name] = _Track(x, y, z, t_s)
            return None
        dt = t_s - tr.t
        if dt <= 0.0:
            return None
        speed = math.dist((x, y, z), (tr.x, tr.y, tr.z)) / dt
        tr.x, tr.y, tr.z, tr.t = x, y, z, t_s
        if tr.reported:
            return None
        if speed > REST_SPEED_MS or z > REST_MAX_Z_M:
            tr.still_since = None
            return None
        if tr.still_since is None:
            tr.still_since = t_s
        if t_s - tr.still_since >= REST_HOLD_S:
            tr.reported = True
            return x, y
        return None
