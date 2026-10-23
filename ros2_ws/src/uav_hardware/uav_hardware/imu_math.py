"""Quaternion helpers for bno080_node. No ROS imports."""

from __future__ import annotations

import math

Quat = tuple[float, float, float, float]  # (x, y, z, w)


def normalise(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        raise ValueError("zero quaternion")
    return tuple(c / n for c in q)  # type: ignore[return-value]


def multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product a * b (apply b first, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def from_rpy_deg(roll: float, pitch: float, yaw: float) -> Quat:
    r, p, y = (math.radians(a) / 2 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = (
        math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y),
    )  # fmt: skip
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotate_vector(q: Quat, v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = q
    p = multiply(multiply(q, (v[0], v[1], v[2], 0.0)), (-x, -y, -z, w))
    return p[0], p[1], p[2]


def sensor_to_body(q_world_sensor: Quat, q_body_sensor: Quat) -> Quat:
    """Orientation of the body given the sensor's, when the sensor is mounted rotated by
    q_body_sensor relative to the body (base_link FLU)."""
    x, y, z, w = q_body_sensor
    return normalise(multiply(q_world_sensor, (-x, -y, -z, w)))
