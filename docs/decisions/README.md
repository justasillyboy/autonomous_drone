# Architecture decision records

Short records of decisions that are expensive to reverse or easy to undo by accident. Each one
states the context, the decision and its consequences. Code that depends on a decision cites
it (for example `ADR-0001` in `mission/timers.py`).

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-timers-in-seconds.md) | Mission timers measure seconds, not frames | Accepted |
| [0002](0002-fail-fast-config.md) | Invalid configuration fails at startup | Accepted |
| [0003](0003-frame-conventions.md) | One frame convention per layer, conversions only in `frames.py` | Accepted |

## Still to record

- The ten "Architecture Decisions" of the legacy
  [`legacy/vision/CLAUDE.md`](../../legacy/vision/CLAUDE.md) that still apply (altitude source
  priority, class names defined in one place, contours on the small mask, `mask_ratio: 4`,
  YOLO11n for the Hailo-8L, F2 feedforward gating, the boundary + Dice loss callback,
  world-space scored zones). Decisions 1 and 8 are superseded by ADR-0002.
- ROS 2 Jazzy + Gazebo Harmonic, and MAVROS rather than a direct MAVLink bridge.
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
