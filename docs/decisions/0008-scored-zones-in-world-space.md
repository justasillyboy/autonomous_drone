# ADR-0008: Circles already hit are excluded by a radius in metres

- **Status:** Accepted (migrated from legacy decision 10)
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/mission/scoring.py`, `DropConfig.scored_world_radius_m`

## Context

After a drop the drone must not lock onto the same circle again while looking for the next
one. A pixel radius around the old position (the legacy `SCORED_RADIUS_PX = 80`) shrinks and
grows with altitude.

## Decision

Scored circles are stored as ground offsets in metres in the camera-heading frame and moved
with the drone's own motion (body velocity and yaw changes). A detection within
`drop.scored_world_radius_m` (0.60 m) of a stored position is ignored.

## Consequences

- The exclusion is altitude-invariant.
- It depends on correct signs for body velocity and yaw. The legacy ROS node fed both
  mirrored (ADR-0003); `mission_node` converts them in `uav_core.frames`.
