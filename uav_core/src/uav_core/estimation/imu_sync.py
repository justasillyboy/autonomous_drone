"""Match IMU/sensor readings to a camera frame's capture time."""

from __future__ import annotations

from collections import deque

from uav_core.types import SensorSnapshot


class ImuSyncBuffer:
    """Keep recent (stamp, snapshot) pairs; return the one closest to a frame stamp.

    A frame is 30-100 ms old by the time the mission sees it; using the newest IMU sample
    instead would compensate an old image with a fresh attitude."""

    def __init__(self, maxlen: int):
        self._buf: deque[tuple[float, SensorSnapshot]] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, stamp_s: float, snapshot: SensorSnapshot) -> None:
        self._buf.append((stamp_s, snapshot))

    def nearest(self, stamp_s: float) -> SensorSnapshot | None:
        if not self._buf:
            return None
        return min(self._buf, key=lambda item: abs(item[0] - stamp_s))[1]
