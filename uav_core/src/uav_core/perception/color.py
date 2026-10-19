"""HSV colour identification of fire circles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from uav_core.config import HsvRange
from uav_core.types import Color


def color_of_mask(
    mask: np.ndarray,
    roi_bgr: np.ndarray,
    hsv_ranges: Mapping[Color, Sequence[HsvRange]],
    threshold: float,
) -> Color | None:
    """Dominant colour inside a uint8 mask (255 = object) over a BGR region of the same size.

    The first colour with the largest in-range fraction wins; below `threshold` -> None.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi = cv2.bitwise_and(hsv, hsv, mask=mask)
    total = max(1, int(mask.sum() / 255))
    best, best_frac = None, 0.0
    for color, ranges in hsv_ranges.items():
        color_mask = np.zeros_like(mask)
        for lo, hi in ranges:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(roi, np.array(lo), np.array(hi)))
        color_mask = cv2.bitwise_and(color_mask, mask)
        frac = color_mask.sum() / 255 / total
        if frac > best_frac:
            best, best_frac = color, frac
    return best if best_frac >= threshold else None


def color_in_contour(
    frame_bgr: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    contour: np.ndarray,
    hsv_ranges: Mapping[Color, Sequence[HsvRange]],
    threshold: float,
) -> Color | None:
    """Colour of the region enclosed by `contour`, read from the circle's bounding square."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(frame_w, cx + radius), min(frame_h, cy + radius)
    if not (x2 > x1 and y2 > y1):
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    pts = contour.reshape(-1, 2) - np.array([x1, y1], dtype=np.int32)
    pts = np.clip(pts, 0, [roi.shape[1] - 1, roi.shape[0] - 1])
    cv2.fillPoly(roi_mask, [pts.reshape(-1, 1, 2)], 255)
    return color_of_mask(roi_mask, roi, hsv_ranges, threshold)


def load_hsv_calibration(path: str | Path) -> dict[Color, tuple[HsvRange, ...]]:
    """Read the JSON written by tools/hsv_calibrator.py: {"red": [{"lo": [..], "hi": [..]}]}."""
    raw = json.loads(Path(path).read_text())
    ranges = {
        Color(name): tuple((tuple(r["lo"]), tuple(r["hi"])) for r in entries)
        for name, entries in raw.items()
    }
    missing = set(Color) - set(ranges)
    if missing:
        raise ValueError(f"{path}: missing colours {sorted(missing)}; rerun hsv_calibrator.py")
    return ranges
