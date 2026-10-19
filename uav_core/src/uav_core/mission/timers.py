"""Time-integrated timers (ADR-0001): `held += dt` while a condition holds.

At a fixed frame rate this reproduces the legacy frame counters exactly; at a variable
rate (thermal throttling, slower-than-real-time sim) it stays correct in seconds. EPS
absorbs float accumulation error so that 33 steps of 0.01 s reach 0.33 s.
"""

from __future__ import annotations

EPS = 1e-6


def reached(value: float, threshold: float) -> bool:
    """value >= threshold, tolerant to accumulation error."""
    return value >= threshold - EPS


def exceeded(value: float, threshold: float) -> bool:
    """value > threshold, tolerant to accumulation error."""
    return value > threshold + EPS


def below(value: float, threshold: float) -> bool:
    """value < threshold, tolerant to accumulation error."""
    return value < threshold - EPS


class HoldTimer:
    """Seconds a condition has held continuously."""

    __slots__ = ("held_s",)

    def __init__(self) -> None:
        self.held_s = 0.0

    def tick(self, dt: float) -> float:
        self.held_s += dt
        return self.held_s

    def reset(self) -> None:
        self.held_s = 0.0


def scan_step_index(elapsed_s: float, step_s: float, n_steps: int) -> int:
    """Index of the yaw-scan step active after elapsed_s; the last step holds."""
    return min(int((elapsed_s + EPS) // step_s), n_steps - 1)
