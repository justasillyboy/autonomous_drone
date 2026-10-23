import numpy as np
import pytest
from uav_core.config import MissionConfig, RulebookConfig
from uav_core.types import (
    AltSource,
    Color,
    Detection,
    FlightCommand,
    MissionStatus,
    Phase,
    Round,
    StepOutput,
    TargetClass,
)
from uav_core.viz import draw_hud
from uav_core.viz.hud import COLOR_BGR, HELIPAD_BGR, _status_lines

CFG = MissionConfig()
RULES = RulebookConfig()


def _out(phase=Phase.TARGETING, round_=Round.QUALIFIER, drop=False, target=0, penalties=0):
    return StepOutput(
        command=FlightCommand(phase=phase, action="ALIGN", vx=0.1, vz=-0.2, drop_ball=drop),
        status=MissionStatus(
            phase=phase,
            round=round_,
            score=40,
            penalties=penalties,
            balls_dropped=2,
            target_index=target,
            alt_source=AltSource.TOF,
            elapsed_s=250.0,
        ),
    )


def _circle(color=Color.RED, edge=False):
    return Detection(TargetClass.CIRCLE, color, 320, 240, 50, (270, 190, 370, 290), 0.9, 0.8, edge)


def test_returns_a_new_image_and_leaves_the_input_alone():
    frame = np.zeros((480, 640, 3), np.uint8)
    img = draw_hud(frame, _out(), [_circle()], CFG, altitude_m=3.4)
    assert not frame.any()
    assert img.shape == frame.shape and img.any()


def test_circle_drawn_in_its_colour_and_helipad_as_box():
    frame = np.zeros((480, 640, 3), np.uint8)
    pad = Detection(TargetClass.H_LANDING, None, 100, 400, 40, (60, 360, 140, 440), 0, 0.9, False)
    img = draw_hud(frame, _out(), [_circle(Color.BLUE), pad], CFG)
    assert tuple(img[240, 370]) == COLOR_BGR[Color.BLUE]  # right edge of the circle
    assert tuple(img[400, 60]) == HELIPAD_BGR  # left edge of the helipad box


@pytest.mark.parametrize(
    ("phase", "target", "expected"),
    [
        (Phase.TARGETING, 2, "Next: F3 yellow D=0.6 m (+30)"),
        (Phase.LANDING, 5, "Target: H_landing pad"),
        (Phase.COMPLETE, 5, "MISSION COMPLETE"),
        (Phase.SEARCHING, 5, None),  # every target done, no "Next" line
    ],
)
def test_target_line_per_phase(phase, target, expected):
    texts = [t for t, _ in _status_lines(_out(phase, target=target), CFG, RULES)]
    if expected is None:
        assert not any(t.startswith(("Next", "Target", "MISSION")) for t in texts)
    else:
        assert expected in texts


def test_finals_marks_the_moving_target_and_the_clock_turns_orange():
    lines = _status_lines(_out(round_=Round.FINALS, target=1), CFG, RULES)
    texts = dict(lines)
    assert "Next: F2 red moving D=1.0 m (+50)" in texts
    assert texts["T: 250 s  left: 50 s"] == (0, 165, 255)
    assert lines[0][0].endswith("[FINALS]  [TOF]")


def test_drop_and_penalties_are_highlighted():
    frame = np.zeros((480, 640, 3), np.uint8)
    img = draw_hud(frame, _out(drop=True, penalties=-5), [], CFG, altitude_m=9.0)
    assert (img[:, :, 2] == 255).any()  # the red DROP command line
    texts = dict(_status_lines(_out(penalties=-5), CFG, RULES))
    assert texts["Penalties: -5 pts  Net: 35 pts"] == (200, 80, 80)


def test_unknown_colour_and_edge_clip_still_draw():
    frame = np.zeros((480, 640, 3), np.uint8)
    img = draw_hud(frame, _out(), [_circle(color=None, edge=True)], CFG)
    assert img.any()
