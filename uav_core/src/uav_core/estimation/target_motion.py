"""Smoothed target-centre history for the finals feedforward (moving F2)."""

from __future__ import annotations

from collections import deque

import numpy as np

_MIN_FIT_SAMPLES = 4


class TargetMotionHistory:
    """(x_px, y_px, t_s) samples. Drone motion is removed incrementally every step (shift
    and rotate), so the fit sees only the target's own motion."""

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._hist)

    def clear(self) -> None:
        self._hist.clear()

    def append(self, x: float, y: float, t: float) -> None:
        self._hist.append((x, y, t))

    def shift(self, dx: float, dy: float) -> None:
        self._hist = deque(((x + dx, y + dy, t) for x, y, t in self._hist), maxlen=self._maxlen)

    def rotate_about(self, cx: float, cy: float, cos_d: float, sin_d: float) -> None:
        """Rotate every sample by -dpsi about (cx, cy): the camera yawed +dpsi clockwise, so
        static points appear to turn the other way."""
        self._hist = deque(
            (
                (
                    cx + (x - cx) * cos_d + (y - cy) * sin_d,
                    cy - (x - cx) * sin_d + (y - cy) * cos_d,
                    t,
                )
                for x, y, t in self._hist
            ),
            maxlen=self._maxlen,
        )

    def velocity(self, max_px_s: float) -> tuple[float, float]:
        """Weighted least-squares slope (newest sample weight 1.0, oldest 0.2), clamped."""
        if len(self._hist) < _MIN_FIT_SAMPLES:
            return 0.0, 0.0
        xs = np.array([p[0] for p in self._hist])
        ys = np.array([p[1] for p in self._hist])
        ts = np.array([p[2] for p in self._hist])
        ts = ts - ts[0]
        if ts[-1] < 1e-4:
            return 0.0, 0.0
        w = np.linspace(0.2, 1.0, len(ts))
        vx = float(np.polyfit(ts, xs, 1, w=w)[0])
        vy = float(np.polyfit(ts, ys, 1, w=w)[0])
        return (float(np.clip(vx, -max_px_s, max_px_s)), float(np.clip(vy, -max_px_s, max_px_s)))
