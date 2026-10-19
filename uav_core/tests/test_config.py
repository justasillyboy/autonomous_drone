import pytest
from pydantic import ValidationError
from uav_core.config import (
    DropConfig,
    MissionConfig,
    RulebookConfig,
    load_mission_config,
    with_hsv_ranges,
)
from uav_core.types import Color, Round


def test_defaults_are_the_legacy_constants():
    cfg = MissionConfig()
    assert cfg.drop.altitude_min_m == 3.05
    assert cfg.drop.align_pixel_thresh_px == 25
    assert cfg.perception.fill_ratio_min_circle == 0.40
    assert cfg.timing.land_early_at_s == 285.0
    assert cfg.control.pid_finals.kp == 0.6


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        MissionConfig().drop.altitude_min_m = 3.5  # type: ignore[misc]


def test_drop_altitude_below_rule_fails_fast():
    with pytest.raises(ValidationError, match="below the rulebook floor"):
        DropConfig(altitude_min_m=2.9)


def test_drop_window_must_be_ordered():
    with pytest.raises(ValidationError, match="altitude_min_m < altitude_target_m"):
        DropConfig(altitude_target_m=5.0)


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "mission.yaml"
    path.write_text("drop:\n  altitude_minimum_m: 3.2\n")
    with pytest.raises(ValidationError, match="altitude_minimum_m"):
        load_mission_config(path)


def test_yaml_overrides_one_value(tmp_path):
    path = tmp_path / "mission.yaml"
    path.write_text("drop:\n  align_pixel_thresh_px: 20\ncamera:\n  f_px: 487.5\n")
    cfg = load_mission_config(path)
    assert cfg.drop.align_pixel_thresh_px == 20
    assert cfg.camera.f_px == 487.5
    assert cfg.drop.altitude_min_m == 3.05


def test_empty_yaml_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_mission_config(path) == MissionConfig()


def test_yaml_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_mission_config(path)


def test_hsv_ranges_must_cover_every_colour():
    with pytest.raises(ValidationError, match="missing"):
        with_hsv_ranges(MissionConfig(), {Color.RED: (((0, 0, 0), (10, 255, 255)),)})


def test_hsv_values_are_range_checked():
    bad = {c: (((0, 0, 0), (200, 255, 255)),) for c in Color}
    with pytest.raises(ValidationError, match="bad HSV range"):
        with_hsv_ranges(MissionConfig(), bad)


def test_rulebook_tables_per_round():
    rules = RulebookConfig()
    assert [t.points for t in rules.targets(Round.QUALIFIER)] == [20, 20, 30, 30, 50]
    assert [t.points for t in rules.targets(Round.FINALS)] == [10, 50, 30, 30, 50]
    assert rules.takeoff_bonus(Round.FINALS) == 10
    assert rules.landing_bonus(Round.QUALIFIER) == 30
