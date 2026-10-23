"""The ADR-0003 release gate: do FlightCommand signs move the drone the way they say?

After a normal takeoff, three steps each command one axis for `step_s` and compare the
ground-truth motion with the expected direction: +vz must descend, +vy must move to the
drone's right, +yaw_rate must turn clockwise seen from above. No ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from uav_core.frames import wrap_deg
from uav_core.types import FlightCommand, Phase

CLIMB_TO_M = 3.0
SETTLE_S = 2.0


@dataclass(frozen=True)
class Step:
    name: str
    command: FlightCommand
    min_motion: float  # metres or degrees in the expected direction


STEPS = (
    Step("descend", FlightCommand(Phase.SEARCHING, "CHECK_DESCEND", vz=0.3), 0.3),
    Step("right", FlightCommand(Phase.SEARCHING, "CHECK_RIGHT", vy=0.3), 0.3),
    Step("clockwise", FlightCommand(Phase.SEARCHING, "CHECK_CW", yaw_rate_dps=20.0), 15.0),
)


def enu_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def motion_along(step: str, start: tuple, end: tuple) -> float:
    """Signed motion in the step's expected direction, from (x, y, z, enu_yaw_deg) poses."""
    x0, y0, z0, yaw0 = start
    x1, y1, z1, yaw1 = end
    if step == "descend":
        return z0 - z1
    if step == "right":  # the drone's right in ENU is its heading rotated -90 deg
        psi = math.radians(yaw0)
        return (x1 - x0) * math.sin(psi) - (y1 - y0) * math.cos(psi)
    return -wrap_deg(yaw1 - yaw0)  # clockwise from above = ENU yaw decreasing


@dataclass
class DirectionCheck:
    step_s: float = 3.0
    phase: str = "climb"
    index: int = 0
    t_mark: float | None = None
    start_pose: tuple | None = None
    results: dict[str, dict] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def passed(self) -> bool:
        return self.done and all(r["ok"] for r in self.results.values())

    def update(self, t_s: float, pose: tuple) -> FlightCommand:
        """pose = (x, y, z, enu_yaw_deg); returns the command to publish now."""
        hover = FlightCommand(Phase.SEARCHING, "CHECK_HOVER")
        if self.phase == "climb":
            if pose[2] < CLIMB_TO_M:
                return FlightCommand(Phase.TAKEOFF, "CHECK_TAKEOFF", vz=-0.8)
            self.phase, self.t_mark = "settle", t_s
            return hover
        if self.phase == "settle":
            if t_s - self.t_mark < SETTLE_S:
                return hover
            self.phase, self.t_mark, self.start_pose = "step", t_s, pose
        if self.phase == "step":
            step = STEPS[self.index]
            if t_s - self.t_mark < self.step_s:
                return step.command
            moved = motion_along(step.name, self.start_pose, pose)
            self.results[step.name] = {"moved": round(moved, 3), "ok": moved >= step.min_motion}
            self.index += 1
            self.phase = "settle" if self.index < len(STEPS) else "done"
            self.t_mark = t_s
        return hover
