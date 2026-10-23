"""action_node logic without ROS: frame conversion, GUIDED gate, servo pulse."""

import math
from types import SimpleNamespace as NS

import pytest
from uav_core.types import FlightCommand, Phase
from uav_hardware.imu_math import from_rpy_deg, multiply, rotate_vector, sensor_to_body
from uav_hardware.setpoint import (
    FRAME_BODY_NED,
    MAV_CMD_DO_SET_SERVO,
    VELOCITY_AND_YAW_RATE,
    ActionGate,
    fill_position_target,
    servo_command,
    to_setpoint,
)


def test_descend_right_clockwise_becomes_flu_for_mavros():
    sp = to_setpoint(
        FlightCommand(Phase.TARGETING, "ALIGN", vx=0.3, vy=0.2, vz=0.6, yaw_rate_dps=30)
    )
    assert sp.coordinate_frame == FRAME_BODY_NED
    assert (sp.vx, sp.vy, sp.vz) == (0.3, -0.2, -0.6)  # FLU: right is -y, descend is -z
    assert sp.yaw_rate == pytest.approx(-math.radians(30))  # clockwise is negative in ROS


def test_type_mask_uses_only_velocity_and_yaw_rate():
    ignored = {1, 2, 4, 64, 128, 256, 1024}
    assert sum(ignored) == VELOCITY_AND_YAW_RATE
    assert not VELOCITY_AND_YAW_RATE & (8 | 16 | 32 | 2048)  # vx vy vz yaw_rate used


def test_fill_position_target():
    msg = NS(velocity=NS(x=0, y=0, z=0))
    fill_position_target(msg, to_setpoint(FlightCommand(Phase.SEARCHING, "X", vx=1.0)))
    assert msg.velocity.x == 1.0 and msg.type_mask == VELOCITY_AND_YAW_RATE


DROP = FlightCommand(Phase.DROPPING, "DROP", drop_ball=True)


def test_no_setpoint_and_no_drop_outside_guided():
    gate = ActionGate()
    assert gate.on_command(DROP, 0.0) == (None, None)  # mode unknown
    gate.mode = "LOITER"
    assert gate.on_command(DROP, 0.0) == (None, None)
    assert gate.on_tick(5.0) is None


def test_drop_is_a_pulse_back_to_idle():
    gate = ActionGate(mode="GUIDED", pulse_s=0.5)
    sp, pwm = gate.on_command(DROP, 10.0)
    assert sp is not None and pwm == 1900
    assert gate.on_command(DROP, 10.1)[1] is None  # no second drop mid-pulse
    assert gate.on_tick(10.3) is None
    assert gate.on_tick(10.5) == 1100
    assert gate.on_command(DROP, 12.0)[1] == 1900  # the next ball can go


def test_guided_gate_can_be_disabled_for_bench_tests():
    assert ActionGate(require_guided=False).on_command(DROP, 0.0)[1] == 1900


def test_servo_command_fields():
    assert servo_command(9, 1900) == {
        "broadcast": False,
        "command": MAV_CMD_DO_SET_SERVO,
        "param1": 9.0,
        "param2": 1900.0,
    }


def test_mounting_rotation_is_removed():
    mount = from_rpy_deg(0, 0, 90)  # sensor turned 90 deg left on the frame
    body = from_rpy_deg(0, 10, 30)
    sensor = multiply(body, mount)
    assert sensor_to_body(sensor, mount) == pytest.approx(body)
    assert rotate_vector(mount, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
