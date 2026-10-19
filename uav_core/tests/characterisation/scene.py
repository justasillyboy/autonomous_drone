"""Deterministic scenes for the characterisation tests.

A scripted drone flies over a flat field with a nadir camera (pinhole, f = 500 px, 640 x 480,
no distortion). Each step renders the BGR frame and the low-resolution instance masks that a
YOLO-seg model with mask_ratio 4 would return. The scene is open loop: the drone follows its
scripted path whatever the mission commands.

Ground frame: x forward (the drone's nose at takeoff), y to the right, metres. Image: the top
of the frame is the nose, so u = 320 + (y_obj - y_drone) * f / z and
v = 240 - (x_obj - x_drone) * f / z. Optical flow is the drone's body velocity, which matches
the legacy sign convention (+vx moves ground objects down the image, +vy moves them left).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

FRAME_W = 640
FRAME_H = 480
MASK_RATIO = 4
F_PX = 500.0
FLOOR_BGR = (90, 90, 90)
# Competition paints #FC2A1B, #FAFF37, #3C00FB in BGR, plus a decoy that matches no HSV range.
PAINT_BGR = {
    "red": (27, 42, 252),
    "yellow": (55, 255, 250),
    "blue": (251, 0, 60),
    "green": (40, 200, 40),
}
CLASS_IDS = {"circle": 0, "semicircle": 1, "H_landing": 2}
SENSOR_FIELDS = (
    "fc_alt",
    "tof",
    "of_vx",
    "of_vy",
    "pitch",
    "roll",
    "yaw",
    "pitch_rate",
    "roll_rate",
)
EVENTS = ("skip_takeoff", "reset", "signal_landed")


@dataclass(frozen=True)
class Signal:
    """Piecewise-linear segments (t0, t1, v0, v1); closed intervals, first match wins."""

    segments: tuple[tuple[float, float, float, float], ...]

    def at(self, t: float) -> float | None:
        for t0, t1, v0, v1 in self.segments:
            if t0 <= t <= t1:
                return v0 if t1 == t0 else v0 + (t - t0) / (t1 - t0) * (v1 - v0)
        return None


def hold(value: float, t0: float = 0.0, t1: float = 1.0e6) -> tuple[float, float, float, float]:
    return (t0, t1, value, value)


def ramp(t0: float, t1: float, v0: float, v1: float) -> tuple[float, float, float, float]:
    return (t0, t1, v0, v1)


@dataclass(frozen=True)
class Path:
    """Drone position keyframes (t, x, y, z); holds the first/last key outside their range."""

    keys: tuple[tuple[float, float, float, float], ...]

    def at(self, t: float) -> tuple[float, float, float]:
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1:]
        for (t0, *a), (t1, *b) in itertools.pairwise(keys):
            if t0 <= t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return tuple(p + w * (q - p) for p, q in zip(a, b, strict=True))
        return keys[-1][1:]

    def velocity(self, t: float) -> tuple[float, float]:
        """Ground velocity (vx, vy) of the segment containing t; zero outside the keys."""
        for (t0, x0, y0, _), (t1, x1, y1, _) in zip(self.keys, self.keys[1:], strict=False):
            if t0 <= t < t1:
                return ((x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0))
        return (0.0, 0.0)


@dataclass(frozen=True)
class Disc:
    """A painted circle on the ground; kind is 'circle' or 'semicircle', ring=True is hollow."""

    paint: str
    x: float
    y: float
    diameter_m: float
    t_on: float = 0.0
    t_off: float = 1.0e6
    kind: str = "circle"
    ring: bool = False
    vx: float = 0.0
    vy: float = 0.0


@dataclass(frozen=True)
class Pad:
    """The H_landing helipad: a white square outline with an H, side length size_m."""

    x: float
    y: float
    size_m: float
    t_on: float = 0.0
    t_off: float = 1.0e6


@dataclass(frozen=True)
class Instance:
    cls_name: str
    mask: np.ndarray  # float32 (FRAME_H / 4, FRAME_W / 4), 0 or 1
    xyxy: tuple[float, float, float, float]
    conf: float = 0.9


@dataclass(frozen=True)
class Scenario:
    name: str
    n_steps: int
    path: Path
    objects: tuple[Disc | Pad, ...] = ()
    finals: bool = False
    rate_hz: int = 100
    fc_alt_from_path: bool = True
    flow_from_path: bool = True
    signals: tuple[tuple[str, Signal], ...] = ()
    events: tuple[tuple[int, str], ...] = ()
    jumps: tuple[tuple[int, int], ...] = ()  # (step index, extra ticks from that step on)


@dataclass(frozen=True)
class Step:
    index: int
    t: float
    frame: np.ndarray
    instances: tuple[Instance, ...]
    sensors: dict[str, float | None]
    events: tuple[str, ...]


def _to_pixels(dx: float, dy: float, z: float) -> tuple[float, float]:
    return (FRAME_W / 2 + dy * F_PX / z, FRAME_H / 2 - dx * F_PX / z)


def _draw_disc(frame: np.ndarray, mask: np.ndarray, disc: Disc, u: float, v: float, r: float):
    color = PAINT_BGR[disc.paint]
    centre = (round(u), round(v))
    small = (round(u / MASK_RATIO), round(v / MASK_RATIO))
    r_small = max(1, round(r / MASK_RATIO))
    if disc.kind == "semicircle":
        cv2.ellipse(frame, centre, (round(r), round(r)), 0, 0, 180, color, -1)
        cv2.ellipse(mask, small, (r_small, r_small), 0, 0, 180, 1.0, -1)
    elif disc.ring:
        thickness = max(2, round(0.06 * r))
        cv2.circle(frame, centre, round(r), color, thickness)
        cv2.circle(mask, small, r_small, 1.0, max(1, round(thickness / MASK_RATIO)))
    else:
        cv2.circle(frame, centre, round(r), color, -1)
        cv2.circle(mask, small, r_small, 1.0, -1)


def _draw_pad(frame: np.ndarray, mask: np.ndarray, u: float, v: float, half: float):
    p1, p2 = (round(u - half), round(v - half)), (round(u + half), round(v + half))
    thickness = max(2, round(0.08 * half))
    cv2.rectangle(frame, p1, p2, (255, 255, 255), thickness)
    for x in (u - 0.4 * half, u + 0.4 * half):
        cv2.line(
            frame,
            (round(x), round(v - 0.6 * half)),
            (round(x), round(v + 0.6 * half)),
            (255, 255, 255),
            thickness,
        )
    cv2.line(
        frame,
        (round(u - 0.4 * half), round(v)),
        (round(u + 0.4 * half), round(v)),
        (255, 255, 255),
        thickness,
    )
    # The model segments the whole pad (outline and glyph), so the mask has both.
    small = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    lo = (max(0, round((u - half) / MASK_RATIO)), max(0, round((v - half) / MASK_RATIO)))
    hi = (round((u + half) / MASK_RATIO) + 1, round((v + half) / MASK_RATIO) + 1)
    region = small[lo[1] : hi[1], lo[0] : hi[0]] > 200
    mask[lo[1] : hi[1], lo[0] : hi[0]][region] = 1.0


def render(objects: Sequence[Disc | Pad], drone: tuple[float, float, float], t: float):
    """Return (frame_bgr, instances) for the drone at (x, y, z) at time t."""
    x_d, y_d, z = drone
    frame = np.full((FRAME_H, FRAME_W, 3), FLOOR_BGR, np.uint8)
    instances: list[Instance] = []
    for obj in objects:
        if not obj.t_on <= t <= obj.t_off:
            continue
        mask = np.zeros((FRAME_H // MASK_RATIO, FRAME_W // MASK_RATIO), np.float32)
        if isinstance(obj, Pad):
            u, v = _to_pixels(obj.x - x_d, obj.y - y_d, z)
            half = obj.size_m * F_PX / z / 2
            _draw_pad(frame, mask, u, v, half)
            cls_name, extent = "H_landing", half
        else:
            ox = obj.x + obj.vx * (t - obj.t_on)
            oy = obj.y + obj.vy * (t - obj.t_on)
            u, v = _to_pixels(ox - x_d, oy - y_d, z)
            extent = obj.diameter_m * F_PX / z / 2
            _draw_disc(frame, mask, obj, u, v, extent)
            cls_name = obj.kind
        if mask.sum() == 0:
            continue
        xyxy = (
            max(0.0, u - extent),
            max(0.0, v - extent),
            min(float(FRAME_W), u + extent),
            min(float(FRAME_H), v + extent),
        )
        instances.append(Instance(cls_name, mask, xyxy))
    return frame, tuple(instances)


def iter_steps(scenario: Scenario) -> Iterator[Step]:
    signals = dict(scenario.signals)
    extra = 0
    for i in range(scenario.n_steps):
        extra += sum(ticks for idx, ticks in scenario.jumps if idx == i)
        t = (i + extra) / scenario.rate_hz
        drone = scenario.path.at(t)
        frame, instances = render(scenario.objects, drone, t)
        sensors: dict[str, float | None] = dict.fromkeys(SENSOR_FIELDS)
        if scenario.fc_alt_from_path:
            sensors["fc_alt"] = drone[2]
        if scenario.flow_from_path:
            sensors["of_vx"], sensors["of_vy"] = scenario.path.velocity(t)
        for name, signal in signals.items():
            sensors[name] = signal.at(t)
        events = tuple(ev for idx, ev in scenario.events if idx == i)
        yield Step(i, t, frame, instances, sensors, events)
