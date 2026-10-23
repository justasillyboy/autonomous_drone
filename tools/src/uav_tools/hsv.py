"""HSV colour ranges: writing the calibration JSON, and fitting ranges from labelled circles.

Read the JSON with uav_core.perception.color.load_hsv_calibration. Layout (unchanged from the
legacy hsv_calibrator.py, so old files still load):
    {"red": [{"lo": [0, 120, 80], "hi": [10, 255, 255]}, ...], "yellow": [...], "blue": [...]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from uav_core.config import HsvRange
from uav_core.types import Color

HsvRanges = dict[Color, tuple[HsvRange, ...]]

HUE_CENTRES = {Color.RED: 0, Color.YELLOW: 28, Color.BLUE: 122}  # OpenCV hue, 0..179
_HUE_VOTE_HALF_WIDTH = 15
_FIT_HALF_WIDTH = 25
_MIN_S, _MIN_V = 30, 50  # pixels paler or darker than this carry no hue information


def save_hsv_json(ranges: HsvRanges, path: str | Path) -> None:
    data = {
        color.value: [{"lo": list(lo), "hi": list(hi)} for lo, hi in rs]
        for color, rs in ranges.items()
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _hue_distance(h: np.ndarray, centre: int) -> np.ndarray:
    d = np.abs(h.astype(int) - centre)
    return np.minimum(d, 180 - d)


def classify_sample(hsv_pixels: np.ndarray) -> Color | None:
    """The colour whose hue window holds most of the saturated pixels of one circle."""
    s, v = hsv_pixels[:, 1], hsv_pixels[:, 2]
    h = hsv_pixels[(s >= _MIN_S) & (v >= _MIN_V), 0]
    if h.size == 0:
        return None
    votes = {
        c: int((_hue_distance(h, ctr) <= _HUE_VOTE_HALF_WIDTH).sum())
        for c, ctr in HUE_CENTRES.items()
    }
    best = max(votes, key=votes.__getitem__)
    return best if votes[best] > 0 else None


@dataclass(frozen=True)
class FitResult:
    ranges: HsvRanges
    samples: dict[Color, int]  # circles used per colour
    defaulted: tuple[Color, ...]  # colours with no sample, kept from the defaults


def _hue_offset(h: np.ndarray, centre: int) -> np.ndarray:
    """Signed hue difference to centre, in [-90, 90) (hue is circular over 180)."""
    return (h.astype(int) - centre + 90) % 180 - 90


def hue_ranges(h_lo: int, h_hi: int, s_lo: int, v_lo: int) -> tuple[HsvRange, ...]:
    """One inclusive hue interval, possibly crossing 0/180, as OpenCV ranges."""
    if h_hi < 0:
        h_lo, h_hi = h_lo + 180, h_hi + 180
    elif h_lo > 179:
        h_lo, h_hi = h_lo - 180, h_hi - 180
    if h_lo < 0:
        return ((0, s_lo, v_lo), (h_hi, 255, 255)), ((h_lo + 180, s_lo, v_lo), (180, 255, 255))
    if h_hi > 180:
        return ((0, s_lo, v_lo), (h_hi - 180, 255, 255)), ((h_lo, s_lo, v_lo), (180, 255, 255))
    return (((h_lo, s_lo, v_lo), (h_hi, 255, 255)),)


def _s_floor_for_background(bg: np.ndarray, in_window: np.ndarray, v_lo: int, max_fp: float) -> int:
    """Lowest S bound that keeps the matched share of ALL background pixels <= max_fp."""
    cand = np.sort(bg[in_window & (bg[:, 2] >= v_lo), 1])[::-1]
    allowed = int(max_fp * len(bg))
    return int(cand[allowed]) + 1 if len(cand) > allowed else 0


def _fit_one(
    pixels: np.ndarray,
    background: np.ndarray,
    color: Color,
    pct: tuple[float, float],
    margin: tuple[int, int, int],
    max_bg_fp: float,
    max_circle_loss: float,
) -> tuple[HsvRange, ...]:
    centre = HUE_CENTRES[color]
    d = _hue_offset(pixels[:, 0], centre)
    px, d = pixels[np.abs(d) <= _FIT_HALF_WIDTH], d[np.abs(d) <= _FIT_HALF_WIDTH]
    d_lo = int(np.floor(np.percentile(d, pct[0]))) - margin[0]
    d_hi = int(np.ceil(np.percentile(d, pct[1]))) + margin[0]
    s_lo = max(0, int(np.percentile(px[:, 1], pct[0])) - margin[1])
    s_cap = int(np.percentile(px[:, 1], 100 * max_circle_loss))
    v_lo = max(0, int(np.percentile(px[:, 2], pct[0])) - margin[2])
    if len(background):
        bd = _hue_offset(background[:, 0], centre)
        in_window = (bd >= d_lo) & (bd <= d_hi)
        s_bg = _s_floor_for_background(background, in_window, v_lo, max_bg_fp)
        # Tighten for the background, but drop at most max_circle_loss of the circle pixels
        # (the matcher needs only color_match_threshold of them): an unlabelled circle
        # counted as background must not make the labelled ones unmatchable.
        s_lo = max(s_lo, min(s_bg, s_cap))
    return hue_ranges(centre + d_lo, centre + d_hi, s_lo, v_lo)


def fit_ranges(
    circles: Iterable[np.ndarray],
    defaults: HsvRanges,
    background: np.ndarray | None = None,
    pct: tuple[float, float] = (2.0, 98.0),
    margin: tuple[int, int, int] = (3, 20, 20),
    max_bg_fp: float = 0.01,
    max_circle_loss: float = 0.2,
) -> FitResult:
    """Fit per-colour ranges from the HSV pixels of labelled circles (one array per circle).

    Each circle gets a colour by hue vote. A colour's hue interval is the [pct] percentile
    span of its pixels within +/-25 hue of the colour centre (label polygons also cover some
    floor), plus a margin. S and V lower bounds come from the same percentiles minus a
    margin; S is then raised so that at most max_bg_fp of the `background` pixels (outside
    every label) would match, but never so far that more than max_circle_loss of the circle
    pixels fall out (the matcher needs only color_match_threshold of them). Upper S/V bounds
    stay open at 255 like the defaults. Colours with no sample keep their default ranges.
    """
    bg = background if background is not None else np.zeros((0, 3), np.uint8)
    groups: dict[Color, list[np.ndarray]] = {c: [] for c in Color}
    for px in circles:
        color = classify_sample(px)
        if color is not None:
            keep = (px[:, 1] >= _MIN_S) & (px[:, 2] >= _MIN_V)
            groups[color].append(px[keep])
    ranges: HsvRanges = {}
    defaulted = []
    for color in Color:
        if groups[color]:
            pixels = np.concatenate(groups[color])
            limits = (max_bg_fp, max_circle_loss)
            ranges[color] = _fit_one(pixels, bg, color, pct, margin, *limits)
        else:
            ranges[color] = defaults[color]
            defaulted.append(color)
    return FitResult(ranges, {c: len(g) for c, g in groups.items()}, tuple(defaulted))


def in_ranges(hsv_pixels: np.ndarray, ranges: tuple[HsvRange, ...]) -> np.ndarray:
    """Boolean per pixel: inside any of the (inclusive) ranges."""
    hit = np.zeros(len(hsv_pixels), bool)
    for lo, hi in ranges:
        hit |= np.all((hsv_pixels >= lo) & (hsv_pixels <= hi), axis=1)
    return hit
