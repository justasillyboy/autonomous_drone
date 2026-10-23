"""mission_node logic without ROS: sensor conventions, watchdog, operator controls."""

import math
from types import SimpleNamespace as NS

import pytest
from uav_core.config import MissionConfig
from uav_core.types import Color, Detection, Phase, TargetClass
from uav_mission.bridge import MAV_LANDED_STATE_ON_GROUND, MissionRunner, SensorState, Watchdog


def _quat_yaw(yaw_deg):
    h = math.radians(yaw_deg) / 2
    return NS(x=0.0, y=0.0, z=math.sin(h), w=math.cos(h))


def _quat_pitch(pitch_deg):
    h = math.radians(pitch_deg) / 2
    return NS(x=0.0, y=math.sin(h), z=0.0, w=math.cos(h))


ZERO = NS(x=0.0, y=0.0, z=0.0)


def test_body_velocity_from_mavros_flu_becomes_frd():
    s = SensorState(10)
    s.on_velocity_body_flu(0.5, 0.2)  # forward, LEFT
    snap = s.snapshot(1.0)
    assert (snap.of_vx_ms, snap.of_vy_ms) == (0.5, -0.2)


def test_imu_yaw_and_pitch_become_frd_and_sync_to_the_frame_stamp():
    s = SensorState(10)
    s.on_imu(1.00, _quat_yaw(80.0), ZERO)  # ROS yaw 80 = heading 10 clockwise from north
    s.on_imu(1.10, _quat_pitch(-5.0), NS(x=0.0, y=math.radians(10), z=0.0))  # nose up
    early = s.snapshot(1.02)
    assert early.yaw_deg == pytest.approx(10.0)
    late = s.snapshot(1.09)
    assert late.pitch_deg == pytest.approx(5.0)
    assert late.pitch_rate_dps == pytest.approx(-10.0)


def test_all_zero_quaternion_is_ignored_and_ranges_are_validated():
    s = SensorState(10)
    s.on_imu(1.0, NS(x=0.0, y=0.0, z=0.0, w=0.0), ZERO)
    s.on_range(float("inf"), 0.1, 8.0)
    s.on_range(9.5, 0.1, 8.0)
    snap = s.snapshot(1.0)
    assert snap.yaw_deg is None and snap.tof_alt_m is None
    s.on_range(3.3, 0.1, 8.0)
    s.on_rel_alt(3.4)
    snap = s.snapshot(2.0)
    assert (snap.tof_alt_m, snap.fc_alt_m, snap.stamp_s) == (3.3, 3.4, 2.0)


def test_watchdog_needs_a_first_feed():
    w = Watchdog(0.5)
    assert not w.expired(100.0)
    w.feed(100.0)
    assert not w.expired(100.5) and w.expired(100.6)


def _centred_red():
    return Detection(
        TargetClass.CIRCLE, Color.RED, 320, 240, 60, (260, 180, 380, 300), 0.95, 0.9, False
    )


def test_runner_drops_and_e_stop_blocks_until_reset():
    r = MissionRunner(MissionConfig(), skip_takeoff=True)
    r.sensors.on_rel_alt(3.4)
    out = r.step(0.0, [_centred_red()], (640, 480))
    assert out.command.drop_ball and out.status.balls_dropped == 1
    r.e_stop()
    assert r.step(0.1, [_centred_red()], (640, 480)) is None
    hover = r.hover("E_STOP")
    assert (hover.vx, hover.vy, hover.vz, hover.drop_ball) == (0.0, 0.0, 0.0, False)
    r.reset()
    assert not r.e_stopped and r.sm.phase is Phase.SEARCHING and r.sm.status().score == 0


def test_arm_rising_edge_starts_the_clock_once():
    r = MissionRunner(MissionConfig())
    assert r.on_armed(True, 10.0) is True
    assert r.on_armed(True, 11.0) is False
    r.sensors.on_rel_alt(1.0)
    out = r.step(12.0, [], (640, 480))
    assert out.status.elapsed_s == pytest.approx(2.0)


def test_touchdown_scores_only_during_the_helipad_phases():
    r = MissionRunner(MissionConfig(), skip_takeoff=True)
    assert r.on_landed_state(MAV_LANDED_STATE_ON_GROUND) is False  # on the ground at start
    r.on_landed_state(2)  # airborne
    assert r.on_landed_state(MAV_LANDED_STATE_ON_GROUND) is False  # SEARCHING: not a landing
    r.sm._ctx.phase = Phase.LANDING
    r.on_landed_state(2)
    assert r.on_landed_state(MAV_LANDED_STATE_ON_GROUND) is True
    assert r.sm.phase is Phase.COMPLETE and r.sm.status().score == 30
