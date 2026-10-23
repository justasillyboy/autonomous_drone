# ADR-0001: Mission timers measure seconds, not frames

- **Status:** Accepted
- **Date:** 2026-10-24
- **Code:** `uav_core/src/uav_core/mission/timers.py`, `MissionStateMachine.step(t_s, ...)`,
  `TimingConfig`

## Context

The legacy `mission.py` counted frames. Every time window (no-detection timeout, recovery scan,
scan step, tilt relaxation, geofence warning, altitude penalty) was a frame count derived from a
`fps_hint` that the caller had to set from the video FPS (default 30):

```python
self._max_no_detect = max(3, int(0.33 * fps))  # 0.33 s -> 10 frames at 30 fps
```

That has three problems:

1. **The real rate is not constant.** On the Pi 5 the perception rate drops under thermal
   throttling and Hailo load, and Gazebo SITL often runs slower than real time. A window sized
   from a 30 fps hint lasts 1.5x longer when the pipeline actually runs at 20 fps.
2. **It was easy to get wrong.** The legacy code already had two bugs of this kind: the
   tilt-relax and geofence thresholds were computed once at 30 fps and never refreshed, and
   assigning `mission.fps_hint` silently created a new attribute until a property was added.
3. **`int()` truncation and the `max(n, ...)` floors** make the effective windows differ from
   the documented seconds at most frame rates.

## Decision

- `step()` takes a timestamp `t_s` in seconds. It can be wall time or ROS sim time
  (`use_sim_time`); only differences are used.
- Each window is a `HoldTimer` that accumulates `dt` while its condition holds, compared against
  a threshold in seconds from `TimingConfig` / `DropConfig`.
- Comparisons go through `reached` / `exceeded` / `below` with `EPS = 1e-6`, so 33 steps of
  0.01 s reach 0.33 s despite float accumulation error.
- The first step assumes `timing.first_step_dt_s` (legacy: `1 / fps_hint`). The odometry `dt`
  is clamped to [1 ms, 0.5 s] so a stall or a debugger breakpoint does not corrupt the dead
  reckoning.

## Consequences

- At a fixed frame rate the behaviour equals the legacy frame counters: the characterisation
  goldens (10 scenarios at 100 Hz) are reproduced bit-exactly.
- At a variable rate the windows stay correct in seconds. This is a deliberate departure from
  the legacy behaviour, by up to one frame per window.
- Callers must pass monotonic timestamps. The ROS node will pass message stamps, so the same
  code works in SITL at any real-time factor.
- Some quantities are still counted in **samples** on purpose, because they size filters rather
  than time windows: `feedforward.history_len`, `estimation.tof_window`,
  `estimation.tof_spike_reset_count`, `estimation.imu_buffer_len`.
