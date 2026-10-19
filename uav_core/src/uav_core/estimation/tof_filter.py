"""Median filter with spike rejection for the downward ToF rangefinder."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

_log = logging.getLogger(__name__)
_MIN_SAMPLES = 3  # median is used once the buffer holds this many readings


class TofMedianFilter:
    """Reject single-frame obstacles (a TZ gate reads ~0.9 m at 3.5 m) without freezing on
    a genuine step change: after `spike_reset_count` consecutive rejections the buffer is
    flushed and re-seeded with the new level."""

    def __init__(self, window: int, spike_threshold_m: float, spike_reset_count: int):
        self._buf: deque[float] = deque(maxlen=window)
        self._threshold = spike_threshold_m
        self._reset_count = spike_reset_count
        self._spikes = 0

    def reset(self) -> None:
        self._buf.clear()
        self._spikes = 0

    def update(self, raw: float | None) -> float | None:
        """Filtered altitude, or None for no reading or a rejected spike."""
        if raw is None:
            return None
        if len(self._buf) >= _MIN_SAMPLES:
            median = float(np.median(list(self._buf)))
            if abs(raw - median) > self._threshold:
                self._spikes += 1
                if self._spikes < self._reset_count:
                    return None
                _log.info("ToF step change: buffer re-seeded with %.2f m", raw)
                self._buf.clear()
        self._buf.append(raw)
        self._spikes = 0
        if len(self._buf) < _MIN_SAMPLES:
            return raw
        return float(np.median(list(self._buf)))
