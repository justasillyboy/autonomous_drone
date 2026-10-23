# Architecture decision records

Short records of decisions that are expensive to reverse or easy to undo by accident. Each one
states the context, the decision and its consequences. Code that depends on a decision cites
it (for example `ADR-0001` in `mission/timers.py`).

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-timers-in-seconds.md) | Mission timers measure seconds, not frames | Accepted |
| [0002](0002-fail-fast-config.md) | Invalid configuration fails at startup | Accepted |
| [0003](0003-frame-conventions.md) | One frame convention per layer, conversions only in `frames.py` | Accepted |
| [0004](0004-altitude-source-priority.md) | Altitude from the FC, then the rangefinder, then vision | Accepted |
| [0005](0005-contours-on-the-small-mask.md) | Circle geometry from contours on the low-resolution mask | Accepted |
| [0006](0006-classes-matched-by-name.md) | Model classes matched to TargetClass by name | Accepted |
| [0007](0007-moving-target-feedforward-gating.md) | F2 feedforward off while the drone manoeuvres | Accepted |
| [0008](0008-scored-zones-in-world-space.md) | Circles already hit excluded by a radius in metres | Accepted |
| [0009](0009-ros2-jazzy-mavros-setpoint-raw.md) | ROS 2 Jazzy, Gazebo Harmonic, MAVROS with body-frame setpoints | Accepted |

## Still to record

- Legacy decisions 5 and 6 (`mask_ratio: 4` and "only YOLO11n fits the Hailo-8L") are
  hypotheses until measured on the device (experiments E3 and E4, M4). The deployed legacy
  ONNX is YOLO11s, which already contradicts decision 6 (defect D11).
- Legacy decision 7 (boundary + Dice loss through an Ultralytics callback) moves with the
  training code in M3, together with experiment E2 that tests whether the loss helps.
- One ADR per legacy quirk fixed after the port (listed in the design spec, section 6).

## Template

```markdown
# ADR-NNNN: <decision as a sentence>

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD
- **Code:** <files that implement it>

## Context
## Decision
## Consequences
```
