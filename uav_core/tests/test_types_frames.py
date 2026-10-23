import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from uav_core.frames import (
    body_frd_to_flu,
    cos_sin_deg,
    enu_quaternion_to_frd_euler_deg,
    flu_rates_to_frd_dps,
    flu_to_body_frd,
    rotate,
    rotate_inverse,
    wrap_deg,
)
from uav_core.types import FlightCommand, Phase, SensorSnapshot

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)
angle = st.floats(min_value=-720.0, max_value=720.0, allow_nan=False)


def test_contracts_are_immutable():
    cmd = FlightCommand(phase=Phase.TAKEOFF, action="ASCEND", vz=-0.8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.vz = 0.0  # type: ignore[misc]
    assert SensorSnapshot().fc_alt_m is None


def test_frd_to_flu_flips_lateral_vertical_and_yaw():
    vx, vy, vz, yaw = body_frd_to_flu(1.0, 2.0, 0.5, 90.0)
    assert (vx, vy, vz) == (1.0, -2.0, -0.5)
    assert yaw == pytest.approx(-math.pi / 2)


@given(finite, finite, finite, finite)
def test_frd_flu_round_trip(vx, vy, vz, yaw):
    back = flu_to_body_frd(*body_frd_to_flu(vx, vy, vz, yaw))
    assert back == pytest.approx((vx, vy, vz, yaw), abs=1e-9)


@given(angle)
def test_wrap_deg_range(a):
    w = wrap_deg(a)
    assert -180.0 <= w < 180.0
    assert math.isclose(math.cos(math.radians(w)), math.cos(math.radians(a)), abs_tol=1e-9)


@given(finite, finite, angle)
def test_rotate_then_inverse_is_identity(x, y, a):
    c, s = cos_sin_deg(a)
    assert rotate_inverse(*rotate(x, y, c, s), c, s) == pytest.approx((x, y), abs=1e-6)


def test_rotate_90_clockwise_turns_forward_into_right():
    c, s = cos_sin_deg(90.0)
    assert rotate(1.0, 0.0, c, s) == pytest.approx((0.0, 1.0), abs=1e-12)


def _quat_zyx(roll_deg, pitch_deg, yaw_deg):
    """Quaternion (x, y, z, w) of an intrinsic Z-Y-X rotation, as ROS publishes it."""
    r, p, y = (math.radians(a) / 2 for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


@pytest.mark.parametrize(
    ("enu_rpy", "frd_rpy"),
    [
        ((0, 0, 90), (0, 0, 0)),  # facing north: NED heading 0
        ((0, 0, 0), (0, 0, 90)),  # facing east: heading 90, clockwise positive
        ((0, -10, 90), (0, 10, 0)),  # nose up is negative pitch in FLU, positive in FRD
        ((5, 0, 90), (5, 0, 0)),  # right wing down is positive roll in both
    ],
)
def test_ros_attitude_becomes_frd(enu_rpy, frd_rpy):
    got = enu_quaternion_to_frd_euler_deg(*_quat_zyx(*enu_rpy))
    assert got == pytest.approx(frd_rpy, abs=1e-9)


@given(angle)
def test_frd_heading_turns_clockwise_when_ros_yaw_turns_counter_clockwise(a):
    heading = enu_quaternion_to_frd_euler_deg(*_quat_zyx(0, 0, a))[2]
    assert -180.0 <= heading < 180.0
    assert wrap_deg(heading - (90.0 - a)) == pytest.approx(0.0, abs=1e-6)


def test_ros_body_rates_become_frd():
    roll, pitch, yaw = flu_rates_to_frd_dps(math.radians(10), math.radians(20), math.radians(30))
    assert (roll, pitch, yaw) == pytest.approx((10.0, -20.0, -30.0))
