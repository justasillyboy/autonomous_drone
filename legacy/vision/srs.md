***

# Software Requirements Specification (SRS)
## Project: Autonomous UAV Vision & Control System for Fire-Fighting 2026

**Target Hardware**: Raspberry Pi 5 + Hailo-8L NPU (13 TOPS) + IMX219 Camera
**Core Software**: YOLO11-seg, MAVLink/ArduPilot, Python-based Mission State Machine

---

## 1. System Overview
This software serves as the high-level decision-making system for an autonomous firefighting UAV. The system processes real-time video to detect fire targets and helipads, fuses data from ToF and IMU sensors, and executes an autonomous state machine to navigate the competition field according to the 2026 Rulebook constraints.

---

## 2. Environment & Hardware Requirements
- **[REQ-HW-01]**: **Processor**: The system must operate on a Raspberry Pi 5 equipped with a Hailo-8L NPU to handle real-time AI inference.
- **[REQ-HW-02]**: **Camera**: An IMX219 wide-angle camera is required, utilizing a fallback focal length configuration of $f_{px} = 500.0$.
- **[REQ-HW-03]**: **Altitude & Odometry**: An MTF-01 sensor must be integrated to provide Time-of-Flight (ToF) laser rangefinding and Optical Flow measurements.
- **[REQ-HW-04]**: **Attitude Tracking**: A BNO080 IMU is required to supply pitch, roll, yaw, and angular rate data.

---

## 3. Functional Requirements

### 3.1. Computer Vision (CV) Subsystem
- **[REQ-CV-01]**: **Instance Segmentation**: The YOLO11-seg model must detect three distinct classes: `circle` (fire targets), `semicircle` (TZ obstacles), and `H_landing` (Safe Zone helipad).
- **[REQ-CV-02]**: **Dynamic Color Filtering**: The system must apply HSV masking to the `circle` class to classify targets in real-time as Red (`#FC2A1B`), Yellow (`#FAFF37`), or Blue (`#3C00FB`).
- **[REQ-CV-03]**: **Vision-Based Altitude Fallback**: In the event of ToF sensor failure, the system must compute altitude $Z$ using the pinhole camera equation: $Z = f_{px} \times \frac{D_{real}}{d_{px}}$.
- **[REQ-CV-04]**: **False-Positive Suppression**: The CV pipeline must reject hollow arcs using `fill_ratio_min_circle = 0.40`, reject boundary lines using `max_fire_diameter_ratio = 0.55`, and suppress overlapping geometries via Cross-class NMS `h_landing_overlap_thresh = 0.45`.

### 3.2. Mission Control & State Machine
The system must navigate the UAV through an eight-phase state machine:
- **[REQ-MS-01]**: **TAKEOFF**: Command a controlled vertical ascent until the UAV reaches the minimum safe flight altitude of $\ge 3.0$ m.
- **[REQ-MS-02]**: **SEARCHING**: Maintain a stable hover while scanning for the next target color defined in the competition sequence.
- **[REQ-MS-03]**: **TARGETING**: Execute an outer-loop PID controller to align the drone with the target, converting pixel error to velocity using `px_to_ms_fire = 0.005`.
- **[REQ-MS-04]**: **DROPPING**: Activate the ball release mechanism strictly when all three following conditions are met simultaneously:
    1. Altitude is strictly within the safe window: $3.05$ m $\le Z \le 4.5$ m.
    2. Alignment error is minimal: $|pixel\_offset| < 25$ pixels.
    3. Attitude is level: $|pitch| \le 8^\circ$ and $|roll| \le 8^\circ$.
- **[REQ-MS-05]**: **RECOVERY**: If the target is lost for $> 10$ frames, execute a structured yaw scan pattern ($+30^\circ/s \rightarrow -60^\circ/s \rightarrow +30^\circ/s \rightarrow 0^\circ/s$) to reacquire the target.
- **[REQ-MS-06]**: **FIND_HELIPAD & LANDING**: Search for the `H_landing` pad and transition to descent mode when the bounding box ratio exceeds `drop_threshold = 0.70`.

### 3.3. Sensor Fusion & Advanced Control
- **[REQ-SF-01]**: **IMU Time-Synchronization**: Maintain a timestamped buffer of IMU readings to synchronize with the estimated $0.070$ s camera pipeline latency, preventing over-correction during braking maneuvers.
- **[REQ-SF-02]**: **Spike Rejection Filter**: Apply a 5-sample median filter to the ToF altitude data to reject sudden anomalous readings (e.g., when flying over a 2.6 m TZ gate). 
- **[REQ-SF-03]**: **Motion Isolation & Feedforward**: Subtract the drone's optical flow velocity from the target's pixel history to estimate the pure translational velocity of the moving F2 target ($10$ cm/s), applying this as a feedforward term to the PID controller.
- **[REQ-SF-04]**: **Zero-Order Hold (ZOH)**: If the target is briefly occluded, the system must hold the last known velocity vector (`cmd_vx`, `cmd_vy`, `cmd_vz`) constant rather than decelerating rapidly, preserving camera stability.

### 3.4. Safety & Geofencing
- **[REQ-SAFE-01]**: **Dead-Reckoning Geofence**: Integrate optical flow data to continuously estimate the UAV's position. If the UAV drifts outside the physical boundaries ($X \in [-1.0, 23.0]$ m, $Y \in [-5.0, 5.0]$ m), automatically trigger a Return-to-Home (RTH) velocity command.
- **[REQ-SAFE-02]**: **Geofence Yaw Rotation**: Ensure dead-reckoning vectors are rotated by the inverse yaw angle before sending body-frame RTH commands to ArduPilot.

---

## 4. Non-Functional Requirements
- **[REQ-NF-01]**: **System Performance**: The end-to-end pipeline (capture, inference, state update) must sustain a frequency of $\ge 20$ Hz to ensure PID control loop stability. `mission.fps_hint` must be set from the actual measured FPS (e.g., `cap.get(cv2.CAP_PROP_FPS)`) before the inference loop begins — all frame-count thresholds (altitude penalty window, drop cooldown, recovery timer, tilt-relax counter) scale as `seconds × fps_hint`. An incorrect `fps_hint` silently shifts all timing gates.
- **[REQ-NF-02]**: **Bandwidth Optimization**: The YOLO11-seg model must be deployed using `mask_ratio: 4` to prevent PCIe bus saturation on the Raspberry Pi 5.
- **[REQ-NF-03]**: **Field Modularity**: All physical constraints, PID gains, and color hex codes must reside externally in `config.yaml` to permit rapid on-site tuning without recompiling the Python source. This is implemented via `load_mission_config_yaml(yaml_path)` in `mission.py`, which must be called before `MissionStateMachine()` is instantiated.
- **[REQ-NF-04]**: **Config Safety Assertion**: `load_mission_config_yaml()` must enforce a hard floor of $3.0$ m on `DROP_ALTITUDE_MIN` regardless of `config.yaml` content. If an operator sets `drop_altitude_min < 3.0`, the function must override it to $3.05$ m and emit a visible warning log.

---

## 5. Competition Rules & Constraints Matrix

| Parameter | Value | Source Constraint |
| :--- | :--- | :--- |
| **Max Flight Weight** | $\le 5.0$ kg | |
| **Max Match Time** | 300 seconds (`MATCH_TIME_S`) | |
| **Auto-Land Trigger** | Force `FIND_HELIPAD` at $t = 285$ s (`LAND_EARLY_AT_S`) | |
| **Helipad Force-Land** | Emergency 1.0 m/s descent when remaining time $< 5$ s (`HELIPAD_FORCE_LAND_S`) | |
| **Drop Altitude Min** | $3.05$ m (`DROP_ALTITUDE_MIN`) — measured from floor | |
| **Drop Altitude Target** | $3.25$ m (`DROP_ALTITUDE_TARGET`) — optimal drop height | |
| **Drop Altitude Max** | $4.50$ m (`DROP_ALTITUDE_MAX`) — above this, ball drift unacceptable | |
| **Scored Exclusion Radius** | $0.60$ m world space (`SCORED_WORLD_RADIUS_M`) | |
| **Altitude Penalty** | $-5$ pts if $Z < 3.0$ m for $> 5$ s (`ALT_PENALTY_TIME_S = 5`) | |
| **F2 Target Velocity** | $10$ cm/s (Finals Round) | |
| **Drop Cooldown** | $2.0$ s between releases (`DROP_COOLDOWN_S`) — prevents double-drop | |
| **Edge-Clip Margin** | $5$ px from frame boundary (`EDGE_CLIP_MARGIN_PX`) — radius untrustworthy if clipped | |
| **ToF Spike Threshold** | $1.5$ m deviation from median (`tof_spike_threshold_m`) — rejects TZ gate reflections | |
| **ToF Spike Reset Count** | $4$ consecutive rejections (`tof_spike_reset_count`) before accepting step-change | |
| **Relaxed Tilt Threshold** | $15.0°$ (`TILT_MAX_RELAXED_DEG`) — crosswind deadlock escape after 1.5 s tilt-block | |
| **FF Scale (Finals F2)** | $0.003$ px/s → m/s (`FF_SCALE`) — feedforward gain for moving target | |
| **FF Max Pixel Rate** | $60.0$ px/s (`MAX_FF_PX_S`) — clamp above this to reject noise/tilt artefacts | |
| **FF Maneuvering Proxy** | $0.35$ m/s (`FF_CMD_THRESHOLD_MS`) — suppresses feedforward when BNO080 absent | |

---

## 6. Glossary & Definitions
* **ToF**: Time-of-Flight (specifically referencing the MTF-01 laser rangefinder).
* **FC**: Flight Controller (hardware executing ArduPilot/PX4 inner loops).
* **NMS**: Non-Maximum Suppression (algorithm used to filter overlapping bounding boxes).
* **FF**: Feedforward (predictive velocity control term used for moving targets).