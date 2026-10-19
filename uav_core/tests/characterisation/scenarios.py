"""The scripted scenarios behind the characterisation golden.

Every scenario runs at a fixed 100 Hz, where each legacy frame-count threshold is an exact
number of seconds (0.33 s = 33 frames, 0.67 s = 67, 1.5 s = 150, 2 s = 200, 3 s = 300,
5 s = 500), so a frame-counting legacy and a seconds-based port must agree step for step.
Objects never overlap in the image, because the legacy code draws on the frame between
the colour checks of different detections.
"""

from __future__ import annotations

from characterisation.scene import Disc, Pad, Path, Scenario, Signal, hold, ramp

SKIP = (0, "skip_takeoff")

QUAL_FULL_MISSION = Scenario(
    name="qual_full_mission",
    n_steps=2800,
    path=Path(
        (
            (0.0, 0.0, 0.0, 0.3),
            (2.0, 0.0, 0.0, 3.4),
            (4.5, 5.0, 0.5, 3.4),
            (6.0, 5.0, 0.5, 3.4),
            (7.6, 5.0, 2.5, 3.4),
            (9.0, 5.0, 2.5, 3.4),
            (10.6, 7.0, 2.5, 3.4),
            (12.0, 7.0, 2.5, 3.4),
            (13.6, 7.0, 0.5, 3.4),
            (15.0, 7.0, 0.5, 3.4),
            (16.2, 8.5, 0.5, 3.4),
            (18.0, 8.5, 0.5, 3.4),
            (22.0, 1.75, 0.0, 3.4),
            (23.0, 1.75, 0.0, 3.4),
            (27.0, 1.75, 0.0, 1.0),
        )
    ),
    objects=(
        Disc("red", 5.0, 0.5, 1.0),
        Disc("red", 5.0, 2.5, 1.0),
        Disc("yellow", 7.0, 2.5, 0.6),
        Disc("yellow", 7.0, 0.5, 0.6),
        Disc("blue", 8.5, 0.5, 0.4),
        Pad(1.75, 0.0, 1.2),  # in the Safe Zone, away from the fire circles
    ),
    events=((2750, "signal_landed"),),
)

QUAL_VISION_ONLY = Scenario(
    name="qual_vision_only",
    n_steps=900,
    path=Path(
        (
            (0.0, 0.0, 0.0, 4.6),
            (1.0, 0.0, 0.0, 4.6),
            (3.0, 0.0, 0.0, 2.9),
            (3.5, 0.0, 0.0, 3.3),
            (4.5, 0.3, 0.2, 3.3),
            (5.5, 0.3, 0.2, 3.3),
            (8.5, 0.3, 2.6, 3.3),
        )
    ),
    objects=(
        Disc("yellow", -0.6, -0.9, 0.6),  # wrong colour, listed before the target
        Disc("green", 0.9, -1.0, 0.6),  # matches no colour
        Disc("yellow", 0.0, 2.0, 0.6),
        Disc("red", 0.3, 0.2, 1.0),
        Disc("red", 0.3, 2.6, 1.0),  # enters from the right edge (edge-clipped)
    ),
    fc_alt_from_path=False,
    events=(SKIP,),
)

QUAL_TOF_FILTER = Scenario(
    name="qual_tof_filter",
    n_steps=1200,
    path=Path(((0.0, 0.0, 0.0, 3.4), (3.0, 0.0, 0.0, 3.4), (5.0, 0.6, 0.0, 3.4))),
    objects=(Disc("red", 0.6, 0.0, 1.0),),
    fc_alt_from_path=False,
    signals=(
        (
            "tof",
            Signal(
                (
                    hold(3.4, 0.0, 0.99),
                    hold(0.9, 1.0, 1.02),  # 3-sample spike: rejected
                    hold(3.4, 1.03, 3.99),
                    hold(1.5, 4.0, 10.49),  # real step: accepted after 4
                    hold(3.4, 10.5, 12.0),
                )
            ),
        ),
    ),
    events=(SKIP,),
)

QUAL_ATTITUDE_GATE = Scenario(
    name="qual_attitude_gate",
    n_steps=300,
    path=Path(((0.0, 0.0, 0.0, 3.4),)),
    objects=(Disc("red", 0.0, 0.0, 1.0),),
    signals=(
        ("pitch", Signal((hold(10.0, 0.0, 1.19), hold(16.0, 1.2, 1.99), hold(12.0, 2.0, 3.0)))),
        ("roll", Signal((hold(3.0, 0.0, 3.0),))),
    ),
    events=(SKIP,),
)

FINALS_MOVING_F2 = Scenario(
    name="finals_moving_f2",
    n_steps=800,
    finals=True,
    path=Path(((0.0, 0.0, 0.0, 3.4), (1.5, 0.0, 0.0, 3.4), (3.0, 0.0, 2.2, 3.4))),
    objects=(
        Disc("red", 0.0, 0.0, 1.0),
        Disc("red", 0.0, 1.6, 1.0, vy=0.1),  # F2 moves at 0.1 m/s, as in the finals
    ),
    signals=(
        ("pitch_rate", Signal((hold(2.0, 0.0, 0.99), hold(20.0, 1.0, 1.29), hold(2.0, 1.3, 8.0)))),
        ("roll_rate", Signal((hold(-3.0, 0.0, 1.49),))),  # then None: proxy guard
        ("yaw", Signal((hold(90.0, 0.0, 1.79), ramp(1.8, 2.4, 90.0, 96.0), hold(96.0, 2.4, 8.0)))),
    ),
    events=(SKIP,),
)

QUAL_GEOFENCE = Scenario(
    name="qual_geofence",
    n_steps=600,
    path=Path(((0.0, 0.0, 0.0, 3.4), (3.5, 0.0, 7.0, 3.4))),
    signals=(("yaw", Signal((hold(30.0, 0.0, 3.99), hold(75.0, 4.0, 6.0)))),),
    events=(SKIP,),
)

QUAL_SKIP_ALL_TO_HELIPAD = Scenario(
    name="qual_skip_all_to_helipad",
    n_steps=3700,
    path=Path(
        (
            (0.0, 0.0, 0.0, 3.4),
            (1.0, 0.0, 0.0, 3.4),
            (9.0, 8.0, 0.0, 3.4),
            (26.0, 8.0, 0.0, 3.4),
            (32.0, 1.75, 0.0, 3.4),
            (33.0, 1.75, 0.0, 3.4),
            (36.0, 1.75, 0.0, 1.0),
        )
    ),
    objects=(Pad(1.75, 0.0, 1.2),),
    events=(SKIP, (3650, "signal_landed")),
)

QUAL_MATCH_TIMER = Scenario(
    name="qual_match_timer",
    n_steps=1250,
    path=Path(((0.0, 0.0, 0.0, 5.0),)),
    objects=(Disc("red", 0.0, 0.0, 1.0),),
    events=(SKIP,),
    # Jump from t = 0.99 s to t = 284.50 s while no timer is running, then cross the
    # forced-landing time (285 s) and the clock-expiry descent (295 s).
    jumps=((100, 28350),),
)

QUAL_RETRY_RESET = Scenario(
    name="qual_retry_reset",
    n_steps=400,
    path=Path(((0.0, 0.0, 0.0, 3.4),)),
    objects=(Disc("red", 0.0, 0.0, 1.0),),
    events=(SKIP, (200, "reset")),
)

QUAL_PERCEPTION_FILTERS = Scenario(
    name="qual_perception_filters",
    n_steps=400,
    path=Path(((0.0, 0.0, 0.0, 3.4), (1.0, 0.0, 0.0, 3.4), (3.0, 0.5, 0.5, 3.4))),
    objects=(
        Disc("red", -0.6, 0.9, 1.0, ring=True),  # hollow: fill ratio too low
        Pad(0.8, -1.0, 0.9),
        Disc("red", 0.8, -1.0, 0.3),  # inside the H_landing box: suppressed
        Disc("red", -0.9, -1.3, 0.8, kind="semicircle"),  # wrong class: ignored
        Disc("red", 0.0, 0.0, 2.6, t_off=0.995),  # too large for a fire circle
        Disc("red", 0.5, 0.5, 1.0, t_on=1.0),
    ),
    events=(SKIP,),
)

SCENARIOS = (
    QUAL_FULL_MISSION,
    QUAL_VISION_ONLY,
    QUAL_TOF_FILTER,
    QUAL_ATTITUDE_GATE,
    FINALS_MOVING_F2,
    QUAL_GEOFENCE,
    QUAL_SKIP_ALL_TO_HELIPAD,
    QUAL_MATCH_TIMER,
    QUAL_RETRY_RESET,
    QUAL_PERCEPTION_FILTERS,
)
