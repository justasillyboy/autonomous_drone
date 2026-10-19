import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from uav_core.estimation.altitude import resolve_altitude
from uav_core.estimation.imu_sync import ImuSyncBuffer
from uav_core.estimation.target_motion import TargetMotionHistory
from uav_core.estimation.tof_filter import TofMedianFilter
from uav_core.frames import cos_sin_deg
from uav_core.types import AltSource, SensorSnapshot


def make_tof() -> TofMedianFilter:
    return TofMedianFilter(window=5, spike_threshold_m=1.5, spike_reset_count=4)


def test_tof_passes_none_through():
    assert make_tof().update(None) is None


def test_tof_returns_raw_until_the_median_has_three_samples():
    tof = make_tof()
    assert tof.update(3.0) == 3.0
    assert tof.update(3.4) == 3.4
    assert tof.update(3.2) == 3.2  # median of (3.0, 3.4, 3.2)


def test_tof_median_smooths_noise():
    tof = make_tof()
    for raw in (3.0, 3.1, 3.2, 3.3):
        tof.update(raw)
    assert tof.update(2.9) == pytest.approx(3.1)


def test_tof_rejects_a_single_obstacle_spike():
    tof = make_tof()
    for raw in (3.5, 3.5, 3.5):
        tof.update(raw)
    assert tof.update(0.9) is None  # a TZ gate under the drone
    assert tof.update(3.5) == 3.5


def test_tof_accepts_a_real_step_change_after_reset_count_spikes():
    tof = make_tof()
    for raw in (3.5, 3.5, 3.5):
        tof.update(raw)
    assert [tof.update(1.5) for _ in range(3)] == [None, None, None]
    assert tof.update(1.5) == 1.5  # 4th consecutive spike: buffer re-seeded


def test_tof_spike_count_restarts_after_an_accepted_reading():
    tof = make_tof()
    for raw in (3.5, 3.5, 3.5):
        tof.update(raw)
    for _ in range(3):
        tof.update(1.0)
    tof.update(3.5)  # accepted: the spike streak ends
    assert tof.update(1.0) is None


def test_tof_reset_forgets_history():
    tof = make_tof()
    for raw in (3.5, 3.5, 3.5):
        tof.update(raw)
    tof.reset()
    assert tof.update(1.0) == 1.0


@pytest.mark.parametrize(
    ("fc", "tof", "expected"),
    [
        (3.2, 3.0, (3.2, AltSource.FC)),
        (None, 3.0, (3.0, AltSource.TOF)),
        (None, None, (None, AltSource.VIS)),
        (0.0, 3.0, (0.0, AltSource.FC)),  # 0.0 is a reading, not a missing value
    ],
)
def test_altitude_priority_ladder(fc, tof, expected):
    assert resolve_altitude(fc, tof) == expected


def test_imu_sync_empty_buffer_has_no_match():
    assert ImuSyncBuffer(maxlen=4).nearest(1.0) is None


def test_imu_sync_returns_the_sample_closest_to_the_frame_stamp():
    buf = ImuSyncBuffer(maxlen=10)
    for k in range(10):
        buf.push(k * 0.01, SensorSnapshot(stamp_s=k * 0.01, pitch_deg=float(k)))
    snap = buf.nearest(0.052)
    assert snap is not None and snap.pitch_deg == 5.0


def test_imu_sync_keeps_only_maxlen_samples():
    buf = ImuSyncBuffer(maxlen=3)
    for k in range(5):
        buf.push(float(k), SensorSnapshot(pitch_deg=float(k)))
    assert len(buf) == 3
    snap = buf.nearest(0.0)
    assert snap is not None and snap.pitch_deg == 2.0  # samples 0 and 1 were evicted
    buf.clear()
    assert len(buf) == 0


def test_motion_velocity_needs_four_samples():
    hist = TargetMotionHistory(maxlen=8)
    for k in range(3):
        hist.append(10.0 * k, 0.0, 0.1 * k)
    assert hist.velocity(max_px_s=60.0) == (0.0, 0.0)


def test_motion_velocity_of_a_linear_track():
    hist = TargetMotionHistory(maxlen=8)
    for k in range(6):
        hist.append(100.0 + 2.0 * k, 50.0 - 1.0 * k, 0.1 * k)
    vx, vy = hist.velocity(max_px_s=60.0)
    assert vx == pytest.approx(20.0)
    assert vy == pytest.approx(-10.0)


def test_motion_velocity_is_clamped():
    hist = TargetMotionHistory(maxlen=8)
    for k in range(5):
        hist.append(100.0 * k, -100.0 * k, 0.1 * k)
    assert hist.velocity(max_px_s=60.0) == (60.0, -60.0)


def test_motion_velocity_without_elapsed_time_is_zero():
    hist = TargetMotionHistory(maxlen=8)
    for k in range(5):
        hist.append(float(k), 0.0, 1.0)
    assert hist.velocity(max_px_s=60.0) == (0.0, 0.0)


def test_motion_history_is_bounded_and_clearable():
    hist = TargetMotionHistory(maxlen=4)
    for k in range(10):
        hist.append(float(k), 0.0, float(k))
    assert len(hist) == 4
    hist.shift(1.0, 1.0)
    assert len(hist) == 4  # shift keeps the bound
    hist.clear()
    assert len(hist) == 0


def test_motion_shift_removes_drone_translation():
    hist = TargetMotionHistory(maxlen=8)
    for k in range(4):
        hist.append(10.0, 20.0, 0.1 * k)
    hist.shift(5.0, -5.0)
    hist.append(15.0, 15.0, 0.4)  # same ground point after the drone moved
    assert hist.velocity(max_px_s=60.0) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_motion_rotate_about_centre_turns_points_against_the_yaw():
    hist = TargetMotionHistory(maxlen=8)
    hist.append(420.0, 240.0, 0.0)  # 100 px right of the centre
    hist.rotate_about(320.0, 240.0, *cos_sin_deg(90.0))  # camera yawed 90 deg clockwise
    (x, y, _), *_ = hist._hist
    assert (x, y) == pytest.approx((320.0, 140.0))  # now 100 px up the image


@given(st.floats(min_value=-180, max_value=180, allow_nan=False))
def test_motion_rotation_preserves_distance_to_centre(angle):
    hist = TargetMotionHistory(maxlen=8)
    hist.append(400.0, 300.0, 0.0)
    hist.rotate_about(320.0, 240.0, *cos_sin_deg(angle))
    (x, y, _), *_ = hist._hist
    assert math.hypot(x - 320.0, y - 240.0) == pytest.approx(100.0)
