"""Independent referee: scores a simulated match from ground truth only. No ROS imports.

It never reads the mission's own score. Inputs are Gazebo poses (drone, balls, targets),
gate contacts and the match start. Rule interpretations, where the rulebook is terse:

- Takeoff bonus: the drone first reaches 3.0 m (§V Nhiệm vụ 1).
- A ball scores when it comes to rest inside a target's circle; each target scores once, a
  second ball in a done target scores nothing (§VII-2).
- Altitude penalty: below 3.0 m for more than 5 s costs 5 points per full 5 s, counted from
  the first time the drone reaches 3.0 m until it enters the landing approach (the space
  above the helipad, widened by `approach_margin_m`). Same counting as uav_core.
- Out of bounds: 5 points per full 5 s outside the field rectangle.
- Gate contact: 5 points per contact episode (contacts closer than 1 s apart merge).
- Landing: touchdown with the whole drone inside the helipad (centre inside the square
  shrunk by half the drone's footprint) scores the landing bonus. Any touchdown after
  takeoff ends the match; so does the 300 s limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from uav_core.config import RulebookConfig

from uav_sim.field import Layout, in_field

CONTACT_MERGE_S = 1.0
MAX_DT_S = 0.5  # a gap in the pose stream is not counted as time held
TOUCHDOWN_Z_M = 0.15


@dataclass
class _Block:
    """Counts full `period_s` blocks of a condition held continuously."""

    period_s: float
    held_s: float = 0.0
    counted: int = 0

    def update(self, active: bool, dt: float) -> int:
        """New full blocks since the last call."""
        if not active:
            self.held_s, self.counted = 0.0, 0
            return 0
        self.held_s += dt
        full = int(self.held_s / self.period_s + 1e-9)
        new, self.counted = full - self.counted, full
        return new


@dataclass
class Event:
    t_s: float
    kind: str
    points: int
    detail: str = ""


@dataclass
class Referee:
    layout: Layout
    rules: RulebookConfig = field(default_factory=RulebookConfig)
    drone_half_size_m: float = 0.3
    approach_margin_m: float = 1.0
    events: list[Event] = field(default_factory=list)
    scored: set[str] = field(default_factory=set)
    start_s: float | None = None
    finished: bool = False
    reached_takeoff: bool = False
    landed_on_helipad: bool = False
    balls: int = 0
    _last_t: float | None = None
    _last_contact_s: float = -math.inf
    _low: _Block = field(default_factory=lambda: _Block(5.0))
    _out: _Block = field(default_factory=lambda: _Block(5.0))

    def __post_init__(self) -> None:
        self._low = _Block(self.rules.alt_penalty_after_s)

    # -- inputs ----------------------------------------------------------------------------

    def start(self, t_s: float) -> None:
        if self.start_s is None:
            self.start_s = t_s
            self._last_t = t_s

    def drone(self, t_s: float, x: float, y: float, z: float) -> None:
        """Ground-truth drone position, called at a steady rate."""
        if not self._live(t_s):
            return
        dt = min(max(0.0, t_s - self._last_t), MAX_DT_S) if self._last_t is not None else 0.0
        self._last_t = t_s
        if not self.reached_takeoff and z >= self.rules.alt_penalty_floor_m:
            self.reached_takeoff = True
            self._add(t_s, "takeoff", self.rules.takeoff_bonus(self.layout.round))
        in_approach = self.layout.on_helipad(x, y, -self.approach_margin_m)
        low = self.reached_takeoff and not in_approach and z < self.rules.alt_penalty_floor_m
        for _ in range(self._low.update(low, dt)):
            self._add(t_s, "altitude_penalty", self.rules.penalty_altitude_pts, f"z={z:.2f}")
        for _ in range(self._out.update(not in_field(x, y), dt)):
            self._add(t_s, "boundary_penalty", -5, f"({x:.1f}, {y:.1f})")
        if self.reached_takeoff and z < TOUCHDOWN_Z_M:
            self._touchdown(t_s, x, y)

    def ball_at_rest(self, t_s: float, x: float, y: float, targets: Layout | None = None) -> None:
        """A dropped ball stopped at (x, y). `targets` carries moved targets (finals F2)."""
        if not self._live(t_s):
            return
        self.balls += 1
        layout = targets or self.layout
        hit = next((t for t in layout.targets if math.dist((x, y), (t.x, t.y)) <= t.radius_m), None)
        if hit is None:
            self._add(t_s, "ball_missed", 0, f"({x:.2f}, {y:.2f})")
        elif hit.label in self.scored:
            self._add(t_s, "ball_repeat", 0, hit.label)
        else:
            self.scored.add(hit.label)
            self._add(t_s, "ball_hit", hit.points, hit.label)

    def gate_contact(self, t_s: float, gate: str) -> None:
        if not self._live(t_s):
            return
        if t_s - self._last_contact_s >= CONTACT_MERGE_S:
            self._add(t_s, "gate_penalty", -5, gate)
        self._last_contact_s = t_s

    # -- results ---------------------------------------------------------------------------

    @property
    def score(self) -> int:
        return sum(e.points for e in self.events)

    def elapsed(self, t_s: float) -> float:
        return 0.0 if self.start_s is None else t_s - self.start_s

    def result(self, t_s: float) -> dict:
        targets = [t.label for t in self.layout.targets]
        return {
            "round": self.layout.round.value,
            "score": self.score,
            "elapsed_s": round(self.elapsed(t_s), 2),
            "finished": self.finished,
            "takeoff": self.reached_takeoff,
            "targets_hit": sorted(self.scored),
            "targets_missed": [t for t in targets if t not in self.scored],
            "balls_dropped": self.balls,
            "landed_on_helipad": self.landed_on_helipad,
            "penalties": sum(e.points for e in self.events if e.points < 0),
            "failures": self.failures(t_s),
            "events": [e.__dict__ for e in self.events],
        }

    def failures(self, t_s: float) -> list[str]:
        """Failure taxonomy for the Monte Carlo report."""
        kinds = [e.kind for e in self.events]
        out = []
        if not self.reached_takeoff:
            out.append("no_takeoff")
        if len(self.scored) < len(self.layout.targets):
            out.append("targets_missed")
        if "ball_missed" in kinds:
            out.append("missed_drop")
        if "altitude_penalty" in kinds:
            out.append("altitude_penalty")
        if "boundary_penalty" in kinds:
            out.append("out_of_bounds")
        if "gate_penalty" in kinds:
            out.append("gate_contact")
        if not self.finished:
            out.append("timeout")
        elif not self.landed_on_helipad:
            out.append("landed_outside_helipad")
        return out

    # -- internals -------------------------------------------------------------------------

    def _live(self, t_s: float) -> bool:
        """Scoring stops after a touchdown or the match time (a timeout is not a finish)."""
        if self.start_s is None or self.finished:
            return False
        return self.elapsed(t_s) <= self.rules.match_time_s

    def _touchdown(self, t_s: float, x: float, y: float) -> None:
        self.finished = True
        if self.layout.on_helipad(x, y, self.drone_half_size_m):
            self.landed_on_helipad = True
            self._add(t_s, "landing", self.rules.landing_bonus(self.layout.round))
        else:
            self._add(t_s, "landed_outside", 0, f"({x:.2f}, {y:.2f})")

    def _add(self, t_s: float, kind: str, points: int, detail: str = "") -> None:
        self.events.append(Event(round(self.elapsed(t_s), 3), kind, points, detail))
