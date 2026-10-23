# ADR-0007: The F2 feedforward is off while the drone manoeuvres

- **Status:** Accepted (migrated from legacy decision 9)
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/control/feedforward.py`,
  `uav_core/src/uav_core/estimation/target_motion.py`, `FeedforwardConfig`

## Context

In the finals, F2 moves at 0.1 m/s. The mission estimates its image velocity from a short
position history and adds a proportional velocity to the PID command, aiming where the
target will be when the ball lands. While the drone itself pitches or rolls, the image
motion is mostly camera tilt, not target motion.

## Decision

The feedforward is disabled when the IMU pitch or roll rate exceeds
`feedforward.angular_rate_gate_dps` (12 deg/s). Without IMU rates, a proxy is used: the
speed the PID would command exceeds `feedforward.cmd_threshold_ms` (0.35 m/s). The added
velocity is clamped (`max_px_s`, `max_add_ms`).

## Consequences

- The rate gate uses magnitudes only, so it is immune to the FLU/FRD sign issue of
  ADR-0003.
- The proxy is weaker than the IMU gate; flying the finals without the BNO080 is a known
  degradation.
