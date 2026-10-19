import pytest
from uav_core.config import PidGains
from uav_core.control.drop_gate import attitude_check, impact_offset, tilt_within
from uav_core.control.feedforward import (
    add_feedforward,
    fall_time_s,
    is_maneuvering,
)
from uav_core.control.pid import PIDController


def pid(kp=0.0, ki=0.0, kd=0.0, limit=100.0) -> PIDController:
    return PIDController(PidGains(kp=kp, ki=ki, kd=kd), output_limit=limit, first_dt_s=0.1)


def test_pid_proportional_term():
    assert pid(kp=0.5).update(10.0, timestamp=0.0) == 5.0


def test_pid_integrates_error_over_time():
    c = pid(ki=1.0)
    assert c.update(2.0, timestamp=0.0) == pytest.approx(0.2)  # first dt = 0.1 s
    assert c.update(2.0, timestamp=0.5) == pytest.approx(1.2)  # + 2.0 * 0.5 s


def test_pid_derivative_uses_the_real_time_step():
    c = pid(kd=1.0)
    c.update(0.0, timestamp=0.0)
    assert c.update(5.0, timestamp=0.5) == pytest.approx(10.0)


def test_pid_frozen_integral_holds_while_p_keeps_running():
    c = pid(kp=1.0, ki=1.0)
    c.update(2.0, timestamp=0.0)  # integral 0.2
    out = c.update(4.0, timestamp=1.0, freeze_integral=True)
    assert out == pytest.approx(4.0 + 0.2)


def test_pid_anti_windup_and_output_clamp():
    c = pid(ki=1.0, limit=10.0)
    for k in range(100):
        out = c.update(50.0, timestamp=float(k))
    assert out == 10.0
    # The integral is capped at limit / ki, so one opposite error brings it straight back.
    assert c.update(-50.0, timestamp=100.0) == pytest.approx(-10.0)


def test_pid_reset_restores_first_step_behaviour():
    c = pid(kp=1.0, ki=1.0, kd=1.0)
    c.update(3.0, timestamp=0.0)
    c.update(7.0, timestamp=0.2)
    c.reset()
    fresh = pid(kp=1.0, ki=1.0, kd=1.0)
    assert c.update(1.0, timestamp=5.0) == fresh.update(1.0, timestamp=5.0)


def test_pid_non_increasing_timestamp_does_not_divide_by_zero():
    c = pid(kd=1.0)
    c.update(0.0, timestamp=1.0)
    assert c.update(1.0, timestamp=1.0) == 100.0  # dt floored to 1e-4 s, then clamped


def test_fall_time_from_three_metres():
    assert fall_time_s(3.0) == pytest.approx((2 * 3.0 / 9.81) ** 0.5)
    assert fall_time_s(0.0) == fall_time_s(0.5)  # floored altitude


def test_maneuvering_from_imu_rates():
    kwargs = dict(kp=0.6, px_to_ms=0.005, rate_gate_dps=12.0, cmd_threshold_ms=0.35)
    assert is_maneuvering(13.0, 0.0, 0, 0, **kwargs)
    assert is_maneuvering(0.0, -13.0, 0, 0, **kwargs)
    assert not is_maneuvering(5.0, 5.0, 500, 500, **kwargs)  # rates win over the offset


def test_maneuvering_estimated_from_the_offset_without_rates():
    kwargs = dict(kp=0.6, px_to_ms=0.005, rate_gate_dps=12.0, cmd_threshold_ms=0.35)
    assert not is_maneuvering(None, None, 100, 0, **kwargs)  # 0.6*100*0.005 = 0.30 m/s
    assert is_maneuvering(None, None, 0, 120, **kwargs)  # 0.36 m/s
    assert is_maneuvering(5.0, None, 120, 0, **kwargs)  # one rate alone is not enough


def test_feedforward_maps_image_velocity_to_body_axes():
    # Target moving right in the image (+x) -> fly right (+vy); down (+y) -> back (-vx).
    vx, vy = add_feedforward(0.0, 0.0, 50.0, 20.0, 0.003, 0.4, 1.0)
    assert vx == pytest.approx(-0.06)
    assert vy == pytest.approx(0.15)


def test_feedforward_addition_and_total_speed_are_clamped():
    vx, vy = add_feedforward(0.9, -0.9, -1000.0, 1000.0, 0.003, 0.4, 1.0)
    assert (vx, vy) == (pytest.approx(0.5), -1.0)


def test_tilt_within_ignores_missing_axes():
    assert tilt_within(None, None, 8.0)
    assert tilt_within(7.9, -7.9, 8.0)
    assert not tilt_within(None, 8.5, 8.0)


def test_attitude_check_reports_both_axes():
    level, reason = attitude_check(9.0, -10.0, 8.0, 8.0, 8.0)
    assert not level
    assert reason == "pitch=+9.0°>8° roll=-10.0°>8°"


def test_attitude_check_marks_the_relaxed_limit():
    level, reason = attitude_check(16.0, None, 15.0, 8.0, 8.0)
    assert not level and reason.endswith("[RELAXED]")
    assert attitude_check(12.0, 12.0, 15.0, 8.0, 8.0) == (True, "")


def test_impact_offset_without_velocity_is_the_aim_point():
    assert impact_offset(4.0, -3.0, 0.0, 0.0, 3.25, 0.005, 500.0) == (4.0, -3.0)


def test_impact_offset_moves_against_the_commanded_drift():
    scale = 0.005 * 500.0 * fall_time_s(3.25) / 3.25
    x, y = impact_offset(0.0, 0.0, 10.0, 10.0, 3.25, 0.005, 500.0)
    assert x == pytest.approx(-10.0 * scale)
    assert y == pytest.approx(10.0 * scale)


def test_impact_offset_clamps_velocity_spikes():
    assert impact_offset(0.0, 0.0, 1e6, 0.0, 3.25, 0.005, 500.0) == impact_offset(
        0.0, 0.0, 200.0, 0.0, 3.25, 0.005, 500.0
    )
