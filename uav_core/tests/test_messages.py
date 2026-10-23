from types import SimpleNamespace as NS

from uav_core.messages import (
    command_from_msg,
    detection_from_msg,
    fill_command_msg,
    fill_detection_msg,
    fill_status_msg,
    stamp_to_s,
)
from uav_core.types import (
    AltSource,
    Color,
    Detection,
    FlightCommand,
    MissionStatus,
    Phase,
    Round,
    TargetClass,
)


def test_detection_round_trip_including_unknown_colour():
    for color in (Color.BLUE, None):
        d = Detection(TargetClass.CIRCLE, color, 1, 2, 3, (0, 1, 4, 5), 0.5, 0.75, True)
        assert detection_from_msg(fill_detection_msg(NS(), d)) == d
    assert fill_detection_msg(NS(), d).color == ""


def test_command_round_trip():
    c = FlightCommand(Phase.DROPPING, "DROP", 0.1, -0.2, 0.3, 15.0, False, True)
    assert command_from_msg(fill_command_msg(NS(), c)) == c


def test_status_and_stamp():
    s = MissionStatus(Phase.LANDING, Round.FINALS, 120, -5, 5, 5, AltSource.TOF, 280.5)
    m = fill_status_msg(NS(), s, e_stopped=True)
    assert (m.phase, m.round, m.alt_source, m.e_stopped) == ("LANDING", "finals", "TOF", True)
    assert stamp_to_s(NS(sec=3, nanosec=500_000_000)) == 3.5
