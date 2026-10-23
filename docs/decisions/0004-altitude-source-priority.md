# ADR-0004: Altitude comes from the flight controller, then the rangefinder, then vision

- **Status:** Accepted (migrated from legacy decision 3)
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/estimation/altitude.py`, `MissionStatus.alt_source`

## Context

Three altitude sources exist: the flight controller (`/mavros/global_position/rel_alt`), the
MTF-01 ToF rangefinder, and a pinhole estimate from a circle's pixel diameter
(`Z = f_px * D / d_px`). The vision estimate is biased low whenever the drone pitches or
rolls: the circle projects as an ellipse, its enclosing circle grows, and Z shrinks. A low Z
inside the drop window would release the ball below the 3.0 m rule.

## Decision

`resolve_altitude` returns the FC altitude when present, else the filtered ToF, else nothing
(`AltSource.VIS`), in which case the mission falls back to the vision estimate. Vision never
outranks a hardware sensor.

## Consequences

- `alt_source` is published in `MissionStatus` and shown on the HUD; the hardware checklist
  requires `FC` or `TOF` before arming.
- Vision-only runs (offline videos without `--altitude`) still work but are labelled `VIS`.
