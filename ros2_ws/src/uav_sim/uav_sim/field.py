"""Competition field geometry (rulebook 27/02/2026, §IV and figure 1). No ROS imports.

Field frame: origin at the bottom-left corner of figure 1, x along the 24 m length toward the
Safe Zone, y along the 12 m width, z up (the Gazebo world frame). The drone starts on the
helipad in the Safe Zone facing -x, toward the fires.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from uav_core.config import RulebookConfig
from uav_core.types import Color, Round

FIELD_LENGTH_M = 24.0
FIELD_WIDTH_M = 12.0
SAFE_ZONE_DEPTH_M = 3.5
VICTIM_ZONE_DEPTH_M = 4.5  # left end, round 1 only
HELIPAD_SIZE_M = 1.5
HELIPAD_EDGE_GAP_M = 1.0  # helipad to the right field edge
TARGET_PLINTH_M = 0.1
TZ_INNER_DIAMETER_M = 1.6
TZ_CENTRE_HEIGHT_M = 1.7
TZ_TOTAL_HEIGHT_M = 2.6
F2_TRACK_HALF_LENGTH_M = 2.5  # finals: F2 moves back and forth along the orange line
F2_SPEED_MS = 0.1

COLOR_HEX = {Color.RED: "#FC2A1B", Color.YELLOW: "#FAFF37", Color.BLUE: "#3C00FB"}


@dataclass(frozen=True)
class Target:
    label: str
    color: Color
    points: int
    diameter_m: float
    x: float
    y: float
    moving: bool = False

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0


@dataclass(frozen=True)
class Gate:
    label: str
    x: float
    y: float


@dataclass(frozen=True)
class Layout:
    round: Round
    targets: tuple[Target, ...]
    gates: tuple[Gate, ...]
    helipad_x: float
    helipad_y: float
    helipad_size_m: float = HELIPAD_SIZE_M

    def target(self, label: str) -> Target:
        return next(t for t in self.targets if t.label == label)

    def with_target_at(self, label: str, x: float, y: float) -> Layout:
        targets = tuple(replace(t, x=x, y=y) if t.label == label else t for t in self.targets)
        return replace(self, targets=targets)

    def on_helipad(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        """Inside the helipad square shrunk by margin_m (half the drone's footprint)."""
        half = self.helipad_size_m / 2.0 - margin_m
        return abs(x - self.helipad_x) <= half and abs(y - self.helipad_y) <= half


def helipad_centre() -> tuple[float, float]:
    return FIELD_LENGTH_M - HELIPAD_EDGE_GAP_M - HELIPAD_SIZE_M / 2.0, FIELD_WIDTH_M / 2.0


def _targets(rules: RulebookConfig, round_: Round, xy: dict[str, tuple[float, float]]):
    return tuple(
        Target(t.label, t.color, t.points, t.diameter_m, *xy[t.label], moving=t.moving)
        for t in rules.targets(round_)
    )


def _gates() -> tuple[Gate, ...]:
    x1 = FIELD_LENGTH_M - 7.5  # TZ1 is 7.5 m from the right edge, TZ2 10 m further left
    top, bottom = FIELD_WIDTH_M - 2.5, 2.5
    return (
        Gate("TZ1", x1, top),
        Gate("TZ2", x1 - 10.0, top),
        Gate("TZ3", x1 - 10.0, bottom),
        Gate("TZ4", x1, bottom),
    )


def qualifier_layout(rules: RulebookConfig | None = None) -> Layout:
    """Figure 1: F2 at the centre of a 2.5 m circle with F1 (right), F5 (left), F4 (up),
    F3 (down) on it; F1 is 8 m left of the helipad's left edge, on the centre line."""
    rules = rules or RulebookConfig()
    hx, hy = helipad_centre()
    f1x = hx - HELIPAD_SIZE_M / 2.0 - 8.0
    f2x = f1x - 2.5
    xy = {
        "F1": (f1x, hy),
        "F2": (f2x, hy),
        "F3": (f2x, hy - 2.5),
        "F4": (f2x, hy + 2.5),
        "F5": (f2x - 2.5, hy),
    }
    return Layout(Round.QUALIFIER, _targets(rules, Round.QUALIFIER, xy), _gates(), hx, hy)


def random_layout(seed: int, round_: Round, rules: RulebookConfig | None = None) -> Layout:
    """Semi-final / final style: targets drawn at random in the fire area, at least 1 m apart
    edge to edge and 1 m from the gates and field edges. The organisers draw from a fixed
    matrix of positions that is not published, so this samples the whole area instead."""
    rules = rules or RulebookConfig()
    rng = random.Random(seed)
    hx, hy = helipad_centre()
    gates = _gates()
    x_lo, x_hi = VICTIM_ZONE_DEPTH_M + 1.0, FIELD_LENGTH_M - SAFE_ZONE_DEPTH_M - 1.0
    placed: dict[str, tuple[float, float]] = {}
    radii = {t.label: t.diameter_m / 2.0 for t in rules.targets(round_)}
    for label, r in radii.items():
        margin = (
            r + 1.0 + (F2_TRACK_HALF_LENGTH_M if label == "F2" and round_ is Round.FINALS else 0)
        )
        for _ in range(10_000):
            x = rng.uniform(x_lo + margin, x_hi - margin)
            y = rng.uniform(1.0 + r, FIELD_WIDTH_M - 1.0 - r)
            clear_targets = all(
                math.dist((x, y), p) >= r + radii[other] + 1.0 for other, p in placed.items()
            )
            clear_gates = all(math.dist((x, y), (g.x, g.y)) >= r + 1.0 + 0.8 for g in gates)
            if clear_targets and clear_gates:
                placed[label] = (x, y)
                break
        else:
            raise RuntimeError(f"could not place {label} (seed {seed})")
    return Layout(round_, _targets(rules, round_, placed), gates, hx, hy)


def f2_position(base_x: float, base_y: float, t_s: float) -> tuple[float, float]:
    """Finals F2: back and forth along x at 0.1 m/s over +/-2.5 m around its drawn position
    (triangle wave starting toward +x)."""
    period = 4.0 * F2_TRACK_HALF_LENGTH_M / F2_SPEED_MS
    phase = (t_s % period) / period
    tri = 4.0 * phase if phase < 0.25 else 2.0 - 4.0 * phase if phase < 0.75 else 4.0 * phase - 4.0
    return base_x + tri * F2_TRACK_HALF_LENGTH_M, base_y


def in_field(x: float, y: float) -> bool:
    return 0.0 <= x <= FIELD_LENGTH_M and 0.0 <= y <= FIELD_WIDTH_M


TZ_TUBE_RADIUS_M = 0.05
TZ_RING_RADIUS_M = TZ_INNER_DIAMETER_M / 2.0 + TZ_TUBE_RADIUS_M  # tube centre line
TZ_POST_RADIUS_M = 0.04


def gate_clearance(gate: Gate, x: float, y: float, z: float) -> float:
    """Distance from a point to the gate's surface (ring in the y-z plane, normal along x,
    on a vertical post below it). Negative inside the material."""
    dx, dy, dz = x - gate.x, y - gate.y, z - TZ_CENTRE_HEIGHT_M
    ring = math.hypot(math.hypot(dy, dz) - TZ_RING_RADIUS_M, dx) - TZ_TUBE_RADIUS_M
    post_top = TZ_CENTRE_HEIGHT_M - TZ_RING_RADIUS_M
    radial = math.hypot(dx, dy) - TZ_POST_RADIUS_M
    beyond_ends = max(0.0, z - post_top, -z)
    post = math.hypot(max(radial, 0.0), beyond_ends) if beyond_ends else radial
    return min(ring, post)
