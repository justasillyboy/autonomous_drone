import pytest
from uav_core.config import GeofenceConfig, MissionConfig, RulebookConfig
from uav_core.mission import MissionStateMachine
from uav_core.mission.geofence import DeadReckoning, outside_fence, return_velocity
from uav_core.mission.phases.helipad import helipad_action
from uav_core.mission.scoring import ScoredExclusion, pixel_to_world
from uav_core.mission.timers import HoldTimer, below, exceeded, reached, scan_step_index
from uav_core.types import Color, Detection, Phase, SensorSnapshot, TargetClass

FRAME = (640, 480)
LEVEL = dict(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0)


def red_circle(cx: int = 320, cy: int = 240, radius: int = 77, **kw) -> Detection:
    return Detection(
        cls=TargetClass.CIRCLE,
        color=Color.RED,
        cx=cx,
        cy=cy,
        radius=radius,
        bbox=(cx - radius, cy - radius, cx + radius, cy + radius),
        fill_ratio=0.9,
        conf=0.9,
        edge_clipped=kw.get("edge_clipped", False),
    )


def airborne(t: float = 0.0) -> MissionStateMachine:
    """A qualifier mission that has just finished TAKEOFF at time t."""
    sm = MissionStateMachine(MissionConfig())
    sm.step(t, [], SensorSnapshot(fc_alt_m=3.3, **LEVEL), FRAME)
    assert sm.phase is Phase.SEARCHING
    return sm


# -- timers ---------------------------------------------------------------------------------


def test_timer_comparisons_absorb_float_accumulation():
    held = sum([0.01] * 33)  # 0.32999999999999985
    assert reached(held, 0.33)
    assert not exceeded(0.33 + 1e-9, 0.33)
    assert not below(0.33 - 1e-9, 0.33)
    assert below(0.32, 0.33)


def test_hold_timer_accumulates_and_resets():
    timer = HoldTimer()
    timer.tick(0.1)
    assert timer.tick(0.25) == pytest.approx(0.35)
    timer.reset()
    assert timer.held_s == 0.0


@pytest.mark.parametrize(
    ("elapsed", "index"), [(0.0, 0), (0.669, 0), (0.67, 1), (2.01, 3), (99, 3)]
)
def test_scan_step_index_holds_the_last_step(elapsed, index):
    assert scan_step_index(elapsed, 0.67, 4) == index


# -- scored exclusion, geofence, helipad thresholds -----------------------------------------


def test_pixel_to_world_scales_with_altitude():
    assert pixel_to_world(420, 240, 320, 240, 5.0, 500.0) == (1.0, 0.0)
    assert pixel_to_world(320, 340, 320, 240, 0.1, 500.0) == (0.0, 0.1)  # altitude floored


def test_scored_exclusion_follows_drone_motion():
    zone = ScoredExclusion(radius_m=0.6)
    assert not zone.contains(320, 240, 3.0, 320, 240, 500.0)
    zone.add(0.0, 0.0)
    assert zone.contains(320, 240, 3.0, 320, 240, 500.0)
    zone.translate(1.0, 0.0)  # the circle now lies 1 m to the right
    assert not zone.contains(320, 240, 3.0, 320, 240, 500.0)
    assert zone.contains(320 + 167, 240, 3.0, 320, 240, 500.0)
    zone.rotate(0.0, 1.0)  # +90 deg: right -> down the image
    assert zone.contains(320, 240 + 167, 3.0, 320, 240, 500.0)
    zone.clear()
    assert len(zone) == 0


def test_dead_reckoning_rotates_flow_by_relative_yaw():
    odom = DeadReckoning()
    odom.yaw0_deg = 30.0
    odom.integrate(1.0, 0.0, yaw_deg=120.0, dt=2.0)  # nose turned 90 deg clockwise
    assert (odom.x_m, odom.y_m) == pytest.approx((0.0, 2.0), abs=1e-12)
    odom.integrate(1.0, 0.0, yaw_deg=None, dt=1.0)  # no yaw: flow taken as world axes
    assert (odom.x_m, odom.y_m) == pytest.approx((1.0, 2.0), abs=1e-12)
    odom.anchor(1.75, 0.0)
    assert (odom.x_m, odom.y_m) == (1.75, 0.0)


@pytest.mark.parametrize(
    ("x", "y", "outside"),
    [
        (0.0, 0.0, False),
        (-1.1, 0.0, True),
        (22.9, 0.0, False),
        (23.1, 0.0, True),
        (0.0, -5.1, True),
    ],
)
def test_soft_fence_limits(x, y, outside):
    odom = DeadReckoning()
    odom.anchor(x, y)
    assert outside_fence(odom, GeofenceConfig(), RulebookConfig()) is outside


def test_return_velocity_points_home_with_capped_speed():
    odom = DeadReckoning()
    odom.anchor(20.0, 20.0)
    vx, vy = return_velocity(odom, yaw_deg=None, cfg=GeofenceConfig())
    assert vx == pytest.approx(vy) and vx < 0  # straight back, not bent to one axis
    assert (vx**2 + vy**2) ** 0.5 == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("ratio", "action"), [(0.1, "SEARCH"), (0.15, "ALIGN"), (0.4, "APPROACH"), (0.7, "LAND")]
)
def test_helipad_action_thresholds(ratio, action):
    assert helipad_action(ratio, 0.70, 0.40, 0.15) == action


# -- the state machine through its public API -----------------------------------------------


def test_takeoff_awards_the_bonus_once_altitude_is_reached():
    sm = MissionStateMachine(MissionConfig())
    out = sm.step(0.0, [], SensorSnapshot(fc_alt_m=1.0), FRAME)
    assert out.command.action == "ASCEND" and out.command.vz < 0  # FRD: up is negative
    out = sm.step(0.1, [], SensorSnapshot(fc_alt_m=3.3), FRAME)
    assert out.command.action == "REACHED_ALT"
    assert out.status.phase is Phase.SEARCHING and out.status.score == 20


def test_skip_takeoff_starts_searching_without_the_bonus():
    sm = MissionStateMachine(MissionConfig())
    sm.skip_takeoff()
    assert sm.phase is Phase.SEARCHING
    assert sm.step(0.0, [], SensorSnapshot(fc_alt_m=3.3), FRAME).status.score == 0


def test_centred_level_target_is_dropped_and_the_mission_moves_on():
    sm = airborne()
    out = sm.step(0.1, [red_circle()], SensorSnapshot(fc_alt_m=3.25, **LEVEL), FRAME)
    assert out.command.drop_ball and out.command.action == "DROP"
    assert out.status.score == 40 and out.status.balls_dropped == 1
    assert out.status.target_index == 1 and out.status.phase is Phase.SEARCHING


@pytest.mark.parametrize(
    ("sensors", "det", "action"),
    [
        (dict(fc_alt_m=3.25, pitch_deg=12.0, roll_deg=0.0), red_circle(), "WAIT_LEVEL"),
        (dict(fc_alt_m=4.8), red_circle(), "DESCEND"),
        (dict(fc_alt_m=3.25), red_circle(edge_clipped=True), "EDGE_CLIP_BLOCK"),
        (dict(fc_alt_m=3.25, tof_alt_m=2.9), red_circle(), "TOF_BLOCK"),
    ],
)
def test_drop_gates_block_the_release(sensors, det, action):
    sm = airborne()
    out = sm.step(0.1, [det], SensorSnapshot(**sensors), FRAME)
    assert out.command.action.startswith(action)
    assert not out.command.drop_ball and out.status.phase is Phase.TARGETING


@pytest.mark.parametrize("rate_hz", [10, 30, 100])
def test_target_loss_timeout_is_in_seconds_at_any_frame_rate(rate_hz):
    sm = airborne()
    sm.step(0.0, [red_circle(cx=400)], SensorSnapshot(fc_alt_m=3.25, **LEVEL), FRAME)
    t = 0.0
    while sm.phase is not Phase.RECOVERY:
        t += 1.0 / rate_hz
        out = sm.step(t, [], SensorSnapshot(fc_alt_m=3.25, **LEVEL), FRAME)
        assert t < 1.0
    assert out.command.action == "ENTERING_RECOVERY"
    assert t == pytest.approx(0.33, abs=1.0 / rate_hz)


def test_low_altitude_costs_five_points_per_five_seconds():
    sm = airborne()
    penalties = {}
    for k in range(1, 102):
        t = k * 0.1
        penalties[round(t, 1)] = sm.step(
            t, [], SensorSnapshot(fc_alt_m=2.5), FRAME
        ).status.penalties
    assert penalties[4.9] == 0 and penalties[5.0] == -5
    assert penalties[9.9] == -5 and penalties[10.0] == -10


def test_geofence_overrides_the_phase_after_the_warning_time():
    sm = airborne()
    out = None
    for k in range(1, 26):  # fly backwards at 1 m/s for 2.5 s
        sensors = SensorSnapshot(fc_alt_m=3.3, of_vx_ms=-1.0, of_vy_ms=0.0, **LEVEL)
        out = sm.step(k * 0.1, [], sensors, FRAME)
    assert out.command.action == "GEOFENCE_RETURN"
    assert out.command.vx == pytest.approx(0.5) and out.command.vy == pytest.approx(0.0)


def test_match_clock_forces_the_landing():
    sm = MissionStateMachine(MissionConfig())
    sm.start_match(0.0)
    sm.step(1.0, [], SensorSnapshot(fc_alt_m=3.3), FRAME)
    out = sm.step(286.0, [], SensorSnapshot(fc_alt_m=3.3), FRAME)
    assert out.status.phase is Phase.FIND_HELIPAD
    out = sm.step(296.0, [], SensorSnapshot(fc_alt_m=3.3), FRAME)
    assert out.command.action == "FORCE_LAND_CLOCK" and out.command.vz == 1.0
    assert out.status.elapsed_s == 296.0


def test_landing_bonus_is_awarded_once_and_the_mission_stops():
    sm = airborne()
    sm.signal_landed()
    sm.signal_landed()
    out = sm.step(1.0, [red_circle()], SensorSnapshot(fc_alt_m=0.1), FRAME)
    assert out.status.score == 20 + 30
    assert out.status.phase is Phase.COMPLETE and out.command.action == "LANDED"


def test_reset_starts_a_retry_from_scratch():
    sm = airborne()
    sm.reset()
    assert sm.phase is Phase.TAKEOFF and sm.last_command.action == "RETRY_RESET"
    assert sm.status().score == 0
