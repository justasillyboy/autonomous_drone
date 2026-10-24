"""Field geometry, referee rules, scenarios and the world file, without ROS or Gazebo."""

import json
import math
import xml.etree.ElementTree as ET

import pytest
from uav_core.perception.camera import Calibration
from uav_core.types import Phase, Round
from uav_sim import field
from uav_sim.balls import BallTracker, DropTrigger, ball_sdf
from uav_sim.direction_check import DirectionCheck, enu_yaw_deg, motion_along
from uav_sim.field import (
    Gate,
    f2_position,
    gate_clearance,
    helipad_centre,
    qualifier_layout,
    random_layout,
)
from uav_sim.referee import Referee
from uav_sim.scenario import make_scenario, sitl_param_file
from uav_sim.world import build_world, helipad_texture, main, write_scenario

# -- field ----------------------------------------------------------------------------------


def test_qualifier_layout_matches_figure_1():
    lay = qualifier_layout()
    hx, hy = helipad_centre()
    assert (hx, hy) == (22.25, 6.0)
    assert (lay.target("F1").x, lay.target("F1").y) == (13.5, 6.0)  # 8 m left of the pad edge
    assert (lay.target("F2").x, lay.target("F5").x) == (11.0, 8.5)
    assert (lay.target("F3").y, lay.target("F4").y) == (3.5, 8.5)
    assert [t.points for t in lay.targets] == [20, 20, 30, 30, 50]
    assert {(g.label, g.x, g.y) for g in lay.gates} == {
        ("TZ1", 16.5, 9.5), ("TZ2", 6.5, 9.5), ("TZ3", 6.5, 2.5), ("TZ4", 16.5, 2.5),
    }  # fmt: skip


@pytest.mark.parametrize("seed", range(20))
def test_random_layouts_keep_their_distances(seed):
    lay = random_layout(seed, Round.FINALS)
    ts = lay.targets
    for i, a in enumerate(ts):
        assert field.in_field(a.x, a.y)
        assert field.VICTIM_ZONE_DEPTH_M < a.x < field.FIELD_LENGTH_M - field.SAFE_ZONE_DEPTH_M
        for b in ts[i + 1 :]:
            assert math.dist((a.x, a.y), (b.x, b.y)) >= a.radius_m + b.radius_m + 1.0
    assert lay.target("F2").moving and lay.target("F2").points == 50


def test_random_layout_is_reproducible():
    assert random_layout(3, Round.QUALIFIER) == random_layout(3, Round.QUALIFIER)
    assert random_layout(3, Round.QUALIFIER) != random_layout(4, Round.QUALIFIER)


def test_f2_moves_back_and_forth_at_10_cm_per_s():
    assert f2_position(10.0, 6.0, 0.0) == (10.0, 6.0)
    assert f2_position(10.0, 6.0, 25.0) == pytest.approx((12.5, 6.0))
    assert f2_position(10.0, 6.0, 75.0) == pytest.approx((7.5, 6.0))
    assert f2_position(10.0, 6.0, 100.0) == pytest.approx((10.0, 6.0))


def test_gate_clearance():
    g = Gate("TZ1", 10.0, 5.0)
    assert gate_clearance(g, 10.0, 5.0, 1.7) == pytest.approx(0.8)  # ring centre: free
    assert gate_clearance(g, 10.0, 5.0 + 0.85, 1.7) == pytest.approx(-0.05)  # in the tube
    assert gate_clearance(g, 10.0, 5.0, 3.5) == pytest.approx(3.5 - 2.6)  # above the ring top
    assert gate_clearance(g, 10.3, 5.0, 0.5) == pytest.approx(0.26)  # beside the post
    assert gate_clearance(g, 10.0, 5.0, 0.84) < 0.0  # inside the post
    assert gate_clearance(g, 10.0, 5.0, 0.95) == pytest.approx(0.05)  # above the ring's bottom


def test_helipad_footprint():
    lay = qualifier_layout()
    assert lay.on_helipad(22.25, 6.0, margin_m=0.3)
    assert not lay.on_helipad(22.25 + 0.5, 6.0, margin_m=0.3)  # drone overhangs the edge
    assert lay.on_helipad(22.25 + 1.5, 6.0, margin_m=-1.0)  # inside the landing approach


# -- referee --------------------------------------------------------------------------------


def _fly(ref, t0, t1, xyz, dt=0.1):
    t = t0
    while t <= t1 + 1e-9:
        ref.drone(t, *xyz)
        t += dt
    return t


def test_a_perfect_qualifier_scores_200():
    lay = qualifier_layout()
    ref = Referee(lay)
    ref.start(0.0)
    _fly(ref, 0.0, 1.0, (22.25, 6.0, 0.05))  # on the pad: no takeoff yet
    _fly(ref, 1.1, 5.0, (22.25, 6.0, 3.3))
    for i, t in enumerate(lay.targets):
        ref.ball_at_rest(10.0 + i, t.x + 0.1, t.y - 0.1)
    _fly(ref, 20.0, 25.0, (22.25, 6.0, 1.0))  # descending over the pad: no altitude penalty
    ref.drone(25.1, 22.3, 6.05, 0.1)
    res = ref.result(25.1)
    assert res["score"] == 200 and res["penalties"] == 0
    assert res["finished"] and res["landed_on_helipad"] and res["failures"] == []


def test_misses_repeats_and_penalties():
    lay = qualifier_layout()
    ref = Referee(lay)
    ref.start(0.0)
    _fly(ref, 0.0, 2.0, (20.0, 6.0, 3.5))  # takeoff: +20
    ref.ball_at_rest(3.0, 13.5, 6.0)  # F1: +20
    ref.ball_at_rest(4.0, 13.5, 6.2)  # F1 again: 0
    ref.ball_at_rest(5.0, 0.5, 0.5)  # nowhere: 0
    _fly(ref, 6.0, 17.0, (15.0, 6.0, 2.5))  # 11 s low: two full 5 s blocks, -10
    _fly(ref, 17.1, 23.0, (-1.0, 6.0, 3.5))  # 5.9 s outside: -5
    ref.gate_contact(24.0, "TZ1")
    ref.gate_contact(24.5, "TZ1")  # same episode
    ref.gate_contact(26.0, "TZ1")  # new episode
    res = ref.result(26.0)
    assert res["score"] == 20 + 20 - 10 - 5 - 10
    assert res["targets_hit"] == ["F1"] and res["balls_dropped"] == 3
    assert set(res["failures"]) >= {"targets_missed", "missed_drop", "altitude_penalty",
                                    "out_of_bounds", "gate_contact", "timeout"}  # fmt: skip


def test_landing_outside_ends_the_match_without_bonus():
    ref = Referee(qualifier_layout())
    ref.start(0.0)
    _fly(ref, 0.0, 1.0, (20.0, 6.0, 3.2))
    ref.drone(2.0, 15.0, 6.0, 0.05)
    ref.ball_at_rest(3.0, 13.5, 6.0)  # after the finish: ignored
    res = ref.result(3.0)
    assert res["finished"] and not res["landed_on_helipad"]
    assert res["score"] == 20 and "landed_outside_helipad" in res["failures"]


def test_a_gap_in_the_pose_stream_is_not_time_held():
    ref = Referee(qualifier_layout())
    ref.start(0.0)
    ref.drone(0.0, 20.0, 6.0, 3.5)
    ref.drone(1.0, 20.0, 6.0, 2.0)
    ref.drone(9.0, 20.0, 6.0, 2.0)  # 8 s without poses
    assert ref.result(9.0)["penalties"] == 0


def test_nothing_counts_before_start_or_after_300_s():
    ref = Referee(qualifier_layout())
    ref.drone(0.0, 20.0, 6.0, 3.5)
    ref.start(1.0)
    ref.drone(302.0, 20.0, 6.0, 3.5)
    res = ref.result(302.0)
    assert res["score"] == 0 and "no_takeoff" in res["failures"] and "timeout" in res["failures"]


def test_finals_moving_target_is_judged_where_it_is():
    lay = random_layout(1, Round.FINALS)
    ref = Referee(lay)
    ref.start(0.0)
    f2 = lay.target("F2")
    moved = lay.with_target_at("F2", *f2_position(f2.x, f2.y, 20.0))
    ref.ball_at_rest(20.0, f2.x + 2.0, f2.y, targets=moved)
    assert ref.result(20.0)["targets_hit"] == ["F2"] and ref.score == 50


# -- scenario and world ---------------------------------------------------------------------


def test_scenarios_are_reproducible_and_carry_sitl_noise():
    a, b = make_scenario(5), make_scenario(5)
    assert a == b and a.name == "qualifier-00005"
    assert a.layout == qualifier_layout()
    assert make_scenario(5, fixed_layout=False).layout != qualifier_layout()
    text = sitl_param_file(a)
    assert f"SIM_WIND_SPD {a.wind_speed_ms}\n" in text and "SIM_BARO_RND" in text


def test_world_has_every_object_and_spawns_the_drone_on_the_pad():
    s = make_scenario(2, Round.FINALS)
    root = build_world(s).getroot()
    world = root.find("world")
    names = {m.get("name") for m in world.findall("model")}
    assert {"ground", "helipad", "target_F1", "target_F5", "gate_TZ1", "gate_TZ4"} <= names
    gate = world.find("model[@name='gate_TZ1']/link")
    assert len(gate.findall("visual")) == 25 and len(gate.findall("collision")) == 25
    drone = world.find("include")
    x, y, _z, *_, yaw = map(float, drone.find("pose").text.split())
    assert (x, y) == (s.layout.helipad_x, s.layout.helipad_y) and yaw == pytest.approx(
        math.pi, abs=1e-3
    )
    assert world.find("plugin[@name='gz::sim::systems::Sensors']/render_engine").text == "ogre2"


def test_red_target_uses_the_rulebook_colour():
    root = build_world(make_scenario(0)).getroot()
    disc = root.find(".//model[@name='target_F1']/link/visual[@name='disc_visual']")
    assert disc.find("material/diffuse").text == "0.988 0.165 0.106 1"  # FC2A1B


def test_write_scenario_and_cli(tmp_path, capsys):
    assert main(["--seed", "3", "--round", "finals", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith("competition_field.sdf")
    ET.parse(tmp_path / "competition_field.sdf")  # well-formed
    meta = json.loads((tmp_path / "scenario.json").read_text())
    assert meta["layout"]["round"] == "finals" and len(meta["layout"]["targets"]) == 5
    parm = (tmp_path / "sitl.parm").read_text()
    assert parm.startswith("SERVO9_FUNCTION 0\n") and "SIM_ACC_RND" in parm
    model = ET.parse(tmp_path / "models" / "iris_downcam" / "model.sdf").getroot()
    cam = model.find(".//sensor[@name='downcam']/camera")
    assert cam.find("image/width").text == "640"
    ET.parse(tmp_path / "models" / "iris_downcam" / "model.config")
    calib = Calibration.load(tmp_path / "sim_calibration.npz")
    assert calib.frame_size == (640, 480)
    assert calib.camera_matrix[0, 0] == pytest.approx(320 / math.tan(0.9))
    assert write_scenario(make_scenario(1), tmp_path / "q").exists()


def test_helipad_texture_has_a_white_h():
    img = helipad_texture(256)
    assert img.shape == (256, 256, 3) and img[128, 128].min() > 200  # the H crossbar
    assert img[100, 128].max() < 150  # grey between the bars, above the crossbar


# -- balls ----------------------------------------------------------------------------------


def test_drop_trigger_fires_on_the_rising_edge_only():
    trig = DropTrigger(1500)
    assert [trig.update(p) for p in (1100, 1900, 1900, 1100, 1900)] == [
        False, True, False, False, True,
    ]  # fmt: skip


def test_ball_reported_once_after_resting():
    tr = BallTracker()
    assert tr.update("ball_0", 0.0, 13.5, 6.0, 3.0) is None
    assert tr.update("ball_0", 0.5, 13.5, 6.0, 1.8) is None  # falling
    t, rest = 1.0, None
    while rest is None and t < 3.0:
        rest = tr.update("ball_0", t, 13.6, 6.1, 0.134)  # on the plinth
        t += 0.1
    assert rest == (13.6, 6.1) and t < 1.75  # still from 1.1 s, reported 0.5 s later
    assert tr.update("ball_0", 3.0, 13.6, 6.1, 0.134) is None  # reported already


def test_ball_sdf_is_valid_xml_with_the_given_name():
    root = ET.fromstring(ball_sdf("ball_3").split("?>", 1)[1])
    assert root.find("model").get("name") == "ball_3"
    assert float(root.find(".//sphere/radius").text) == pytest.approx(0.0335)


# -- direction check (ADR-0003) ------------------------------------------------------------


def _fly_check(flip_vz=False, flip_vy=False, flip_yaw=False):
    """A point-mass drone facing -x (ENU yaw 180) that executes FRD commands, optionally with
    the legacy sign errors."""
    x, y, z, yaw = 22.25, 6.0, 0.2, 180.0
    check, t, dt = DirectionCheck(step_s=3.0), 0.0, 0.1
    while not check.done and t < 200:
        c = check.update(t, (x, y, z, yaw))
        psi = math.radians(yaw)
        right = (math.sin(psi), -math.cos(psi))  # body right in ENU
        fwd = (math.cos(psi), math.sin(psi))
        vy = -c.vy if flip_vy else c.vy
        x += (fwd[0] * c.vx + right[0] * vy) * dt
        y += (fwd[1] * c.vx + right[1] * vy) * dt
        if c.phase is Phase.TAKEOFF:
            z += 0.8 * dt  # the climb is ArduPilot's own NAV_TAKEOFF (action_node), sign-free
        else:
            z -= (-c.vz if flip_vz else c.vz) * dt
        yaw -= (-c.yaw_rate_dps if flip_yaw else c.yaw_rate_dps) * dt
        t += dt
    return check


def test_correct_signs_pass_every_axis():
    check = _fly_check()
    assert check.passed, check.results
    assert check.results["right"]["moved"] == pytest.approx(0.9, abs=0.05)


@pytest.mark.parametrize(
    ("flip", "axis"), [("flip_vz", "descend"), ("flip_vy", "right"), ("flip_yaw", "clockwise")]
)
def test_each_legacy_sign_error_is_caught(flip, axis):
    check = _fly_check(**{flip: True})
    assert check.done and not check.passed
    assert not check.results[axis]["ok"]
    assert all(r["ok"] for name, r in check.results.items() if name != axis)


def test_motion_helpers():
    assert enu_yaw_deg(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)) == pytest.approx(90)
    # facing north (ENU yaw 90): the drone's right is east (+x)
    assert motion_along("right", (0, 0, 3, 90.0), (1, 0, 3, 90.0)) == pytest.approx(1.0)
    assert motion_along("clockwise", (0, 0, 3, 10.0), (0, 0, 3, -10.0)) == pytest.approx(20.0)


# -- Monte Carlo report ---------------------------------------------------------------------


def _result(score, missed=(), landed=True, failures=(), self_score=None):
    return {"score": score, "targets_missed": list(missed), "landed_on_helipad": landed,
            "failures": list(failures), "self_reported_score": self_score}  # fmt: skip


def test_report_success_rate_failures_and_self_report_gap(tmp_path):
    from uav_sim import report

    runs = [
        _result(200, self_score=200),
        _result(170, missed=["F5"], failures=["targets_missed"], self_score=220),
        _result(150, landed=False, failures=["landed_outside_helipad"]),
        _result(200),
    ]
    for k, r in enumerate(runs):
        (tmp_path / f"qualifier-{k:05d}.json").write_text(json.dumps(r))
    (tmp_path / "direction_check.json").write_text('{"passed": true}')  # ignored
    s = report.summarise(report.load_results(tmp_path))
    assert s["runs"] == 4 and s["success_rate"] == 0.5
    lo, hi = s["success_ci95"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert s["failures"] == {"targets_missed": 1, "landed_outside_helipad": 1}
    assert s["self_report_gap_mean"] == 25.0 and s["self_report_overclaims"] == 1
    text = report.markdown(s)
    assert "**50%**" in text and "| 200-219 | 2 |" in text
    out = tmp_path / "report.md"
    assert report.main([str(tmp_path), "--out", str(out)]) == 0 and out.exists()


def test_wilson_interval_edges():
    from uav_sim.report import wilson

    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(50, 50)
    assert hi == 1.0 and lo > 0.9
