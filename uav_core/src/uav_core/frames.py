"""Frame conventions and conversions, kept in one place.

- Body FRD (what FlightCommand uses): +x forward, +y right, +z down; yaw rate positive
  clockwise seen from above, in deg/s.
- ROS body FLU (REP 103, what MAVROS expects): +x forward, +y left, +z up; yaw rate positive
  counter-clockwise, in rad/s.
- Mission world frame: x along the drone's nose when it became active (yaw_0), y to the right
  of that. Yaw angles are degrees, positive clockwise.
- ROS attitude (sensor_msgs/Imu): body FLU relative to ENU, rates in rad/s about FLU axes.
  In FRD, pitch and yaw change sign and the heading is measured from north (NED).

The 2-D rotations use numpy's cos/sin so the port reproduces the legacy floats exactly.
"""

from __future__ import annotations

import math

import numpy as np


def body_frd_to_flu(
    vx: float, vy: float, vz: float, yaw_rate_dps: float
) -> tuple[float, float, float, float]:
    """Return (vx, vy, vz, yaw_rate_rad_s) in ROS body FLU."""
    return vx, -vy, -vz, -math.radians(yaw_rate_dps)


def flu_to_body_frd(
    vx: float, vy: float, vz: float, yaw_rate_rad_s: float
) -> tuple[float, float, float, float]:
    """Inverse of body_frd_to_flu; yaw rate comes back in deg/s."""
    return vx, -vy, -vz, -math.degrees(yaw_rate_rad_s)


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def cos_sin_deg(angle_deg: float) -> tuple[float, float]:
    rad = np.deg2rad(angle_deg)
    return float(np.cos(rad)), float(np.sin(rad))


def rotate(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    """Rotate (x, y) by +a: body -> world when a is the relative yaw."""
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def rotate_inverse(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    """Rotate (x, y) by -a: world -> body when a is the relative yaw."""
    return cos_a * x + sin_a * y, -sin_a * x + cos_a * y


def flu_rates_to_frd_dps(
    p_rad_s: float, q_rad_s: float, r_rad_s: float
) -> tuple[float, float, float]:
    """Body rates about FLU axes (rad/s) -> (roll, pitch, yaw) rates about FRD axes (deg/s)."""
    return math.degrees(p_rad_s), -math.degrees(q_rad_s), -math.degrees(r_rad_s)


def enu_quaternion_to_frd_euler_deg(
    x: float, y: float, z: float, w: float
) -> tuple[float, float, float]:
    """ROS orientation (FLU body in ENU) -> (roll, pitch, heading) in degrees, FRD in NED.

    Roll keeps its sign (right wing down is positive in both); pitch flips (nose up is
    positive in FRD); the heading is 90 deg - ENU yaw, i.e. clockwise from north, wrapped.
    """
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), -math.degrees(pitch), wrap_deg(90.0 - math.degrees(yaw))
