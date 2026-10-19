"""Outer-loop PID: pixel error -> velocity setpoint (ArduPilot runs the inner loops)."""

from __future__ import annotations

import numpy as np

from uav_core.config import PidGains

_MIN_DT_S = 1e-4


class PIDController:
    def __init__(self, gains: PidGains, output_limit: float, first_dt_s: float = 1.0 / 30.0):
        self.kp, self.ki, self.kd = gains.kp, gains.ki, gains.kd
        self.limit = output_limit
        self._first_dt_s = first_dt_s
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float, timestamp: float, freeze_integral: bool = False) -> float:
        """One sample -> output in [-limit, limit]. freeze_integral holds the integral (used
        while the attitude gate blocks a drop) while P and D keep running."""
        dt = timestamp - self._prev_time if self._prev_time is not None else self._first_dt_s
        dt = max(dt, _MIN_DT_S)
        if not freeze_integral:
            self._integral += error * dt
        integral_limit = self.limit / max(self.ki, 1e-9)  # anti-windup
        self._integral = float(np.clip(self._integral, -integral_limit, integral_limit))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        self._prev_time = timestamp
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(out, -self.limit, self.limit))
