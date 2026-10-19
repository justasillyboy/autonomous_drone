"""Altitude source priority: flight controller > ToF > vision (vision never outranks a
hardware sensor: it underestimates Z when the drone pitches)."""

from __future__ import annotations

from uav_core.types import AltSource


def resolve_altitude(
    fc_alt_m: float | None, tof_alt_m: float | None
) -> tuple[float | None, AltSource]:
    if fc_alt_m is not None:
        return fc_alt_m, AltSource.FC
    if tof_alt_m is not None:
        return tof_alt_m, AltSource.TOF
    return None, AltSource.VIS
