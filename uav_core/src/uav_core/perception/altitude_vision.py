"""Pinhole altitude from a target of known size: Z = f_px * D / d_px."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def altitude_from_diameter(d_px: float, real_diameter_m: float, f_px: float) -> float:
    """Altitude in metres; inf when the circle is smaller than one pixel."""
    if d_px < 1.0:
        return float("inf")
    return f_px * real_diameter_m / d_px


def f_px_from_calibration(path: str | Path) -> float:
    """Mean focal length from the calibration.npz written by camera_calibration.py."""
    cam = np.load(path)["camera_matrix"]
    return float((cam[0, 0] + cam[1, 1]) / 2.0)
