"""Exclusion zones around fire circles that were already hit."""

from __future__ import annotations

from uav_core.control.feedforward import MIN_ALTITUDE_M
from uav_core.frames import rotate


def pixel_to_world(
    px: float, py: float, frame_cx: int, frame_cy: int, altitude_m: float, f_px: float
) -> tuple[float, float]:
    """Camera-relative ground offset in metres (x right, y down the image)."""
    z = max(altitude_m, MIN_ALTITUDE_M)
    return (px - frame_cx) * z / f_px, (py - frame_cy) * z / f_px


class ScoredExclusion:
    """Positions of scored circles in the camera-heading frame, kept up to date with optical
    flow and yaw so the drone does not re-lock on F1 while looking for F2. Metres, not
    pixels, so the radius does not change with altitude."""

    def __init__(self, radius_m: float):
        self._radius_m = radius_m
        self._positions: list[tuple[float, float]] = []

    def __len__(self) -> int:
        return len(self._positions)

    def clear(self) -> None:
        self._positions.clear()

    def add(self, wx: float, wy: float) -> None:
        self._positions.append((wx, wy))

    def translate(self, dwx: float, dwy: float) -> None:
        self._positions = [(wx + dwx, wy + dwy) for wx, wy in self._positions]

    def rotate(self, cos_d: float, sin_d: float) -> None:
        self._positions = [rotate(wx, wy, cos_d, sin_d) for wx, wy in self._positions]

    def contains(
        self, cx: int, cy: int, altitude_m: float, frame_cx: int, frame_cy: int, f_px: float
    ) -> bool:
        if not self._positions:
            return False
        wx, wy = pixel_to_world(cx, cy, frame_cx, frame_cy, altitude_m, f_px)
        r2 = self._radius_m**2
        return any((wx - sx) ** 2 + (wy - sy) ** 2 < r2 for sx, sy in self._positions)
