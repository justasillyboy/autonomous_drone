# Deployment Guide — UAV Fire-Fighting Vision & Control System (2026)

**Hardware target:** Raspberry Pi 5 + Hailo-8L (13 TOPS) + IMX219 wide-angle camera  
**Flight controller:** ArduPilot (inner PID loops, MAVLink)  
**Competition:** UAV Fire-Fighting & Rescue 2026 — Rulebook 27/02/2026

---

## Table of Contents

1. [Deployment File Inventory](#1-deployment-file-inventory)
2. [Software Dependencies](#2-software-dependencies)
3. [Stage 1 — Camera Calibration](#3-stage-1--camera-calibration)
4. [Stage 2 — Model Training & Export](#4-stage-2--model-training--export)
5. [Stage 3 — HSV Colour Calibration (at venue)](#5-stage-3--hsv-colour-calibration-at-venue)
6. [Stage 4 — Standalone Python Deployment](#6-stage-4--standalone-python-deployment)
7. [Stage 5 — ROS 2 Node Deployment](#7-stage-5--ros-2-node-deployment)
8. [Sensor Wiring Reference](#8-sensor-wiring-reference)
9. [Constants & Tuning Reference](#9-constants--tuning-reference)
10. [Competition Day Checklist](#10-competition-day-checklist)
11. [Post-Flight Telemetry Analysis](#11-post-flight-telemetry-analysis)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Deployment File Inventory

### Core deployment files (must be present on the drone)

| File | Role | Required? |
|---|---|---|
| `mission.py` | **Single source of truth** — all runtime constants, `MissionStateMachine`, `AltitudeEstimator`, geometry helpers, colour detection | Always |
| `test_video.py` | Main entry point for standalone deployment (video file or live camera) | Standalone mode |
| `postprocess.py` | Single-frame debug tool — verify detections on still images pre-flight | Debug / staging |
| `logger.py` | Per-frame telemetry CSV — wire into `test_video.py` via `--log` | Strongly recommended |
| `camera_calibration.py` | Undistortion + focal-length extraction from `calibration.npz` | Required with IMX219 |
| `hsv_calibrator.py` | Real-time HSV threshold tuning tool — run at venue before the match | Required |
| `config.yaml` | Documented snapshot of all constants — **human-readable reference, not loaded at runtime** | Reference |

### Artefacts produced during setup (must be present on the drone)

| Artefact | Produced by | Required? |
|---|---|---|
| `best.pt` | `train.py` | Standalone (PyTorch) |
| `best.hef` | Hailo DFC toolchain (after `export.py` → ONNX) | ROS 2 / Hailo-8L mode |
| `calibration.npz` | `camera_calibration.py calibrate` | Always (without it, `DEFAULT_F_PX = 500.0` is used — never acceptable in competition) |
| `hsv_calib.json` | `hsv_calibrator.py` | Required at venue |

### Training-only files (NOT deployed to drone)

| File | Role |
|---|---|
| `train.py` | Training loop with boundary + Dice auxiliary loss |
| `loss.py` | Custom loss function |
| `dataset.py` | Synthetic dataset generator |
| `evaluate.py` | mAP / boundary F1 / Hausdorff evaluation |
| `export.py` | Export to ONNX / TensorRT / Hailo |

---

## 2. Software Dependencies

### Python packages (Pi 5 environment)

```bash
pip install ultralytics opencv-python-headless numpy
pip install pymavlink            # for MAVLink / sensor integration
```

### Hailo-8L runtime (PCIe)

```bash
# Install Hailo PCIe driver + Runtime (from Hailo Developer Zone)
# https://hailo.ai/developer-zone/
pip install hailort              # Python HailoRT bindings
# hailo_platform module is included in hailort package
```

### ROS 2 (if using node-split architecture)

```bash
# Install ROS 2 Humble or Iron on Pi 5 (Ubuntu 22.04 arm64)
sudo apt install ros-humble-desktop ros-humble-mavros ros-humble-mavros-extras
sudo apt install python3-colcon-common-extensions
```

### BNO080 IMU

```bash
pip install adafruit-circuitpython-bno08x
```

---

## 3. Stage 1 — Camera Calibration

**When:** Before leaving for the venue. Re-run if the camera is moved or re-mounted.  
**What it produces:** `calibration.npz` — camera matrix, distortion coefficients, focal length (`f_px`).

### Step-by-step

**A. Print a checkerboard**

Print a checkerboard with **9×6 inner corners** (10×7 squares on paper) and 25 mm square size on A4.
Count inner corners (where four squares meet), not squares — passing the wrong count to `--rows/--cols`
causes calibration failure. Verify squares measure exactly 25 mm before shooting.

**B. Collect calibration frames**

```bash
# Capture frames live from IMX219 (camera index 0) and calibrate
python camera_calibration.py calibrate \
    --source 0 \
    --rows 9 --cols 6 \
    --square-mm 25.0 \
    --output calibration.npz

# OR calibrate from a folder of pre-captured checkerboard images
python camera_calibration.py calibrate \
    --source calib_imgs/ \
    --rows 9 --cols 6 \
    --square-mm 25.0 \
    --output calibration.npz
```

- Minimum **20 frames** with detected checkerboard corners (`MIN_GOOD_FRAMES = 20`).
- Cover all corners of the frame; tilt the board at various angles.
- Good reprojection error: < 0.5 px. Acceptable: < 1.0 px. Redo if > 1.0 px.

**C. Verify the result**

```bash
# Preview undistorted live feed — confirm no barrel distortion on straight lines
python camera_calibration.py preview --calib calibration.npz --source 0

# Undistort a test image
python camera_calibration.py undistort \
    --calib calibration.npz \
    --input test_frame.jpg \
    --output corrected.jpg
```

**D. Extract the focal length**

The focal length `f_px` is embedded in `calibration.npz` as `camera_matrix[0,0]` (fx).  
`test_video.py` and `postprocess.py` extract it automatically via `AltitudeEstimator.from_calibration()` when `--calib calibration.npz` is passed.

> **CRITICAL:** Never run the competition without `--calib calibration.npz`. Without it,
> `DEFAULT_F_PX = 500.0` is used, which gives wrong altitude estimates on any real camera
> and may cause premature drops below the 3.0 m rule limit.

---

## 4. Stage 2 — Model Training & Export

### 4a. Train

```bash
python train.py --config config.yaml --boundary-loss
```

- Model: `yolo11n-seg` (Nano — mandatory for Hailo-8L 13 TOPS budget).
- Target metrics before deployment:
  - Mask mAP50-95 ≥ 0.75
  - Boundary F1 (2 px tolerance) ≥ 0.85
  - Hausdorff distance ≤ 3 px

```bash
# Evaluate after training
python evaluate.py --weights runs/circle_seg/yolo11_v1/weights/best.pt \
                   --data data/circle_seg.yaml \
                   --config config.yaml
```

### 4b. Export to ONNX (for Hailo-8L)

```bash
python export.py --weights best.pt --format onnx --validate --benchmark
```

This produces `best.onnx`.

### 4c. Extract real calibration frames for Hailo INT8 calibration

> **MANDATORY:** Never use synthetic frames for Hailo INT8 quantization calibration.
> Use real flight footage — synthetic pixels do not match IMX219 noise/vignetting.

```bash
python export.py --extract-calib-frames flight.mp4 \
                 --calib calibration.npz \
                 --calib-output hailo_calib/
```

### 4d. Compile to HEF (Hailo DFC toolchain)

Run on a machine with the Hailo DFC toolchain installed (not necessarily the Pi 5):

```bash
# Install Hailo DFC (from Hailo Developer Zone)
hailomz compile yolo11n_seg \
    --ckpt best.onnx \
    --calib-path hailo_calib/ \
    --output-dir ./hailo_output/
# → produces best.hef
```

Copy `best.hef` to the drone.

---

## 5. Stage 3 — HSV Colour Calibration (at venue)

**When:** At the competition venue, **15 minutes before the match**, under actual field lighting.  
**Why:** Indoor competition lighting varies significantly from training data; RGB values of the  
red/yellow/blue fire circles may shift enough to break colour classification.

```bash
python hsv_calibrator.py --source 0 --output hsv_calib.json
# With undistortion:
python hsv_calibrator.py --source 0 --output hsv_calib.json --calib calibration.npz
# Resume from a previous calibration:
python hsv_calibrator.py --source 0 --output hsv_calib.json --load hsv_calib.json
```

### Interactive controls

| Key | Action |
|---|---|
| `R` | Tune Red thresholds (F1, F2) |
| `Y` | Tune Yellow thresholds (F3, F4) |
| `B` | Tune Blue thresholds (F5) |
| `S` | Save current calibration to `--output` |
| `L` | Reload saved calibration |
| `C` | Toggle undistorted feed (requires `--calib`) |
| `Q` / ESC | Quit |

### Validation

After saving, run on a still frame captured at the venue:

```bash
python postprocess.py \
    --weights best.pt \
    --image venue_frame.jpg \
    --output annotated.jpg \
    --calib calibration.npz \
    --hsv-calib hsv_calib.json \
    --config-yaml config.yaml \
    --conf 0.4
```

Confirm the annotated image shows correct colour labels on all three fire circle colours.

---

## 6. Stage 4 — Standalone Python Deployment

This mode runs the full mission pipeline as a single Python process — suitable for:
- Video file testing (offline, CI validation)
- Competition flight with direct sensor integration (no ROS 2)

### Basic commands

```bash
# Offline video test
python test_video.py \
    --weights best.pt \
    --source flight.mp4 \
    --output result.mp4 \
    --calib calibration.npz \
    --hsv-calib hsv_calib.json \
    --config-yaml config.yaml \
    --skip-takeoff \
    --log run_001.csv

# Live camera (competition)
python test_video.py \
    --weights best.pt \
    --source 0 \
    --output result.mp4 \
    --calib calibration.npz \
    --hsv-calib hsv_calib.json \
    --config-yaml config.yaml \
    --log run_001.csv

# Finals mode (F2 is moving target, PID tuned higher)
python test_video.py \
    --weights best.pt \
    --source 0 \
    --output result.mp4 \
    --calib calibration.npz \
    --hsv-calib hsv_calib.json \
    --config-yaml config.yaml \
    --finals \
    --log run_001.csv
```

> **`--config-yaml` is strongly recommended.** Without it, `mission.py` uses
> built-in defaults and any edits to `config.yaml` are ignored at runtime
> (operator-tuned drop-altitude, PID gains, feedforward, ToF filter, etc.
> silently do not apply). The flag calls `load_mission_config_yaml()` BEFORE
> `MissionStateMachine()` is instantiated, which is the only ordering that
> propagates overrides to instance variables (`_tof_spike_threshold_m`,
> `_ff_history_len`, etc.). Invoking it after construction leaves those at
> defaults — see CLAUDE.md Danger Zones.

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--weights` | required | Path to `best.pt` |
| `--source` | required | Video file or camera index (`0`, `1`, …) |
| `--output` | required | Output annotated video path |
| `--conf` | `0.5` | YOLO detection confidence threshold |
| `--calib` | None | `calibration.npz` for undistortion + real `f_px` |
| `--f-px` | None | Override focal length in pixels (overrides `--calib` value) |
| `--finals` | False | Enable Finals mode (F2 = moving target, higher PID) |
| `--takeoff-alt` | `3.3` | Target takeoff altitude in metres (rule min: 3.0 m) |
| `--skip-takeoff` | False | Skip TAKEOFF phase (use for video testing or when FC handles takeoff) |
| `--skip-frames` | `0` | Process 1 in every N+1 frames — reduces GPU load at cost of latency |
| `--smooth` | `0.4` | Temporal smoothing alpha 0–1 |
| `--log` | None | Path for telemetry CSV (e.g. `run_001.csv`) — omit to skip logging |
| `--hsv-calib` | None | HSV calibration JSON from `hsv_calibrator.py` |
| `--config-yaml` | None | Path to `config.yaml` — overrides mission constants (drop altitude, PID gains, ToF filter, feedforward, etc.) at runtime. **Must be passed to apply any on-field tuning** — edits to `config.yaml` have no effect without this flag. |

### Sensor integration (MUST wire before competition flight)

`test_video.py` contains STUB variables for all sensor inputs. These are harmless for  
offline video testing but **all must be wired to real hardware before competition**:

```python
# In test_video.py — find the "COMPETITION WIRING" comment block and replace stubs:

from mission import SensorData
import pymavlink.mavutil as mavutil

# FC barometer — highest priority altitude source
fc_alt = mavlink_conn.altitude_m()          # MAVLink GLOBAL_POSITION_INT

# All sensor data for attitude gate, ToF gate, geofence, feedforward
sensors = SensorData(
    tof_altitude_m  = tof_sensor.read_m(),  # MTF-01 rangefinder
    of_vx_ms        = flow.vx_ms(),         # MTF-01 optical flow
    of_vy_ms        = flow.vy_ms(),
    pitch_deg       = imu.pitch_deg,         # BNO080
    roll_deg        = imu.roll_deg,
    pitch_rate_dps  = imu.pitch_rate_dps,
    roll_rate_dps   = imu.roll_rate_dps,
    yaw_deg         = imu.yaw_deg,          # required for geofence + feedforward rotation
)

# Use actual grab time, not inference time, as frame timestamp
frame_ts = frame_grab_timestamp  # capture BEFORE model() call

frame = mission.update(frame, result,
                        current_altitude=fc_alt,
                        sensor_data=sensors,
                        frame_timestamp=frame_ts)
```

**Without wiring:**  
- `current_altitude=None` → altitude priority ladder falls back to vision-only (`Z_est`) —
  unreliable under pitch, may cause drops below 3.0 m rule limit.
- `sensor_data=None` → attitude gate (pitch/roll < 8°), ToF low-altitude block, geofence,
  and Finals feedforward are all inactive.

### Reading the FlightCommand output

```python
cmd = mission.last_command  # updated every frame by mission.update()
# cmd.vx, cmd.vy, cmd.vz  — body-frame velocity (m/s)
# cmd.yaw_rate             — clockwise positive (deg/s) — convert with np.deg2rad() before sending to MAVROS cmd_vel
# cmd.drop_ball            — True = release servo
# cmd.phase, cmd.action    — current mission state for logging

mavlink_conn.set_velocity_body(vx=cmd.vx, vy=cmd.vy, vz=cmd.vz,
                                yaw_rate=cmd.yaw_rate)
if cmd.drop_ball:
    servo.release()
if mission.phase == Phase.COMPLETE:
    mavlink_conn.disarm()
```

### Retry (referee signals retry)

```python
mission.reset()
mission.skip_takeoff()  # call only if drone is already at altitude
```

---

## 7. Stage 5 — ROS 2 Node Deployment

See `node_split_deploy.md` for the complete 5-node ROS 2 architecture. This section  
provides the deployment procedure and startup sequence.

### 7a. Build the `uav_interfaces` package first

```bash
cd ~/ros2_ws/src
ros2 pkg create uav_interfaces --build-type ament_cmake
# Create the following message files inside uav_interfaces/msg/:
```

**`VisionTargetsMsg.msg`:**
```
std_msgs/Header header
uav_interfaces/Detection[] detections
```

**`Detection.msg`:**
```
uint8   cls_id
string  color
float32 cx
float32 cy
float32 radius
float32 fill_ratio
float32 z_est_raw
float32 conf
bool    edge_clipped
```

**`FlightCommandMsg.msg`:**
```
std_msgs/Header header
float32 vx           # m/s, body-frame forward
float32 vy           # m/s, body-frame right
float32 vz           # m/s, body-frame down (+vz = descend)
float32 yaw_rate     # deg/s, clockwise positive — convert with np.deg2rad() before publishing to /cmd_vel
bool    yaw_lock     # True = maintain current heading
bool    drop_ball    # True = release servo
string  phase        # TAKEOFF / SEARCHING / TARGETING / DROPPING / FIND_HELIPAD / LANDING / COMPLETE
string  action       # human-readable action label: ALIGN / DESCEND / DROP / ASCEND / EDGE_CLIP etc.
int32   score        # current competition score
int8    balls_dropped  # populated from mission.balls_dropped, NOT from FlightCommand dataclass
```

> **Note on `balls_dropped`:** This field is not part of the `FlightCommand` dataclass in
> `mission.py`. Populate it from `mission.balls_dropped` (a `MissionStateMachine` attribute)
> before publishing the message.

```bash
cd ~/ros2_ws
colcon build --packages-select uav_interfaces
source install/setup.bash
```

### 7b. Node startup order

The launch file must start nodes in this exact order to avoid race conditions:

```
1. MAVROS          — must connect to FC before mission_control_node subscribes
2. camera_node     — must publish CameraInfo before perception_node initializes f_px
3. sensor_bridge_node — BNO080 must be streaming before mission_control_node needs IMU
4. perception_node — after CameraInfo is available
5. mission_control_node — after all publishers are ready
6. action_mavros_node
```

### 7c. Launch file parameters

```python
# bringup.launch.py
launch_args = [
    DeclareLaunchArgument('takeoff_alt',   default_value='3.3'),
    DeclareLaunchArgument('is_finals',     default_value='false'),
    DeclareLaunchArgument('hsv_calib',     default_value='/home/pi/hsv_calib.json'),
    DeclareLaunchArgument('calib_npz',     default_value='/home/pi/calibration.npz'),
    DeclareLaunchArgument('weights_hef',   default_value='/home/pi/best.hef'),
    DeclareLaunchArgument('log_csv',       default_value='/home/pi/run_001.csv'),
]
```

### 7d. Hailo-8L initialization in `perception_node`

```python
# perception_node.__init__()
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams

hef = HEF(self.get_parameter('weights_hef').value)
target = VDevice()
network_group = target.configure(hef)[0]
# ... initialize input/output vstreams
```

> **`mask_ratio = 4` is a hardware constraint**, not a quality setting. The PCIe bus
> between Pi 5 and Hailo-8L saturates at `mask_ratio = 2`, dropping FPS below 20 Hz.

### 7e. QoS profiles

| Topic | QoS |
|---|---|
| `/camera/image_raw` | Best Effort, depth=1 |
| `/camera/camera_info` | Reliable, transient local (latched) |
| `/perception/targets` | Reliable, depth=1, keep-last |
| `/perception/debug_image` | Best Effort, depth=1 |
| `/bno080/imu` | Best Effort, depth=5 |
| `/mission/flight_command` | **Reliable**, depth=1 — never Best Effort; `drop_ball=True` packets cannot be dropped |

### 7f. ROS 2 services

| Service | Type | When to call |
|---|---|---|
| `/mission/reset` | `std_srvs/srv/Trigger` | Referee allows retry — resets score, phase, ball count |
| `/mission/e_stop` | `std_srvs/srv/Trigger` | Software kill switch — publishes hover command, pauses state machine |

```bash
# Retry
ros2 service call /mission/reset std_srvs/srv/Trigger

# Emergency stop
ros2 service call /mission/e_stop std_srvs/srv/Trigger
```

### 7g. Match timer

The match timer in `mission_control_node` must start when `mavros_msgs/State.armed == True`,  
**not** at node startup. Subscribe to `/mavros/state` and begin the timer on the ARM event.

```python
def mavros_state_cb(self, msg):
    if msg.armed and not self._timer_started:
        self._match_start = self.get_clock().now()
        self._timer_started = True
```

When `time_elapsed_s >= LAND_EARLY_AT_S (285 s)`, transition to `FIND_HELIPAD`.  
When remaining time `< HELIPAD_FORCE_LAND_S (5 s)`, trigger emergency 1.0 m/s descent.

### 7h. Watchdog

If no `/perception/targets` message arrives within **0.5 s** (10 frames at 20 Hz),  
`mission_control_node` publishes a hover command (`vx=vy=vz=0`) and transitions to  
`RECOVERY` phase. This prevents uncontrolled flight on camera failure.

### 7i. ROS 2 bag recording (for post-flight analysis)

```bash
ros2 bag record \
    /mission/flight_command \
    /perception/targets \
    /bno080/imu \
    /mavros/distance_sensor/rangefinder_sub \
    /mavros/global_position/rel_alt \
    /camera/camera_info
```

Run `logger.py summary()` on the extracted CSV after landing for competition-specific metrics  
(score, balls_dropped, phase transitions, mean FPS, drop events).

---

## 8. Sensor Wiring Reference

### Altitude priority ladder

```
fc_altitude  (MAVLink GLOBAL_POSITION_INT)   ← always prefer
    ↓ if None
tof_altitude_m  (MTF-01 laser rangefinder)
    ↓ if None
z_est_vision  (f_px × D_real / d_px)         ← fallback only — DO NOT use on competition drone
```

> **Never promote vision-estimated altitude** above a hardware sensor. Vision altitude is
> invalid when the drone pitches — the circle projects as an ellipse, `d_px` inflates,
> and `Z_est` underestimates altitude → premature drop below 3.0 m.
>
> The HUD shows `[FC]`, `[TOF]`, or `[VIS]` as the altitude source.
> **If the HUD shows `[VIS]` during flight, stop and investigate.**

### MTF-01 (ToF laser + optical flow)

| Signal | Connection | MAVLink topic |
|---|---|---|
| `tof_altitude_m` | ArduPilot SERIAL_PROTOCOL=1 | `/mavros/distance_sensor/rangefinder_sub` |
| `of_vx_ms`, `of_vy_ms` | Same UART port | `/mavros/local_position/velocity_body` |

### BNO080 (attitude IMU)

| Signal | Connection | Used for |
|---|---|---|
| `pitch_deg`, `roll_deg` | I2C `0x4A` or UART | Attitude drop gate (< 8°) |
| `pitch_rate_dps`, `roll_rate_dps` | Same | Finals F2 feedforward suppression guard |
| `yaw_deg` | Same (full quaternion) | Geofence world-frame velocity rotation |

Library: `adafruit-circuitpython-bno08x`

### SensorData fields summary

```python
from mission import SensorData

SensorData(
    tof_altitude_m  = ...,   # float | None — rangefinder altitude (metres)
    of_vx_ms        = ...,   # float | None — optical flow X velocity (m/s)
    of_vy_ms        = ...,   # float | None — optical flow Y velocity (m/s)
    pitch_deg       = ...,   # float | None — current pitch angle (degrees)
    roll_deg        = ...,   # float | None — current roll angle (degrees)
    pitch_rate_dps  = ...,   # float | None — pitch angular rate (deg/s)
    roll_rate_dps   = ...,   # float | None — roll angular rate (deg/s)
    yaw_deg         = ...,   # float | None — current yaw angle (degrees)
)
# NOTE: FC barometer altitude is NOT a SensorData field.
# Pass it as current_altitude= in mission.update(), not inside SensorData.
```

All fields default to `None` — the state machine degrades gracefully when a sensor is absent,
but degraded operation does not meet competition requirements.

---

## 9. Constants & Tuning Reference

All constants are defined in `mission.py` (authoritative) and documented in `config.yaml`.  
**Never edit `config.yaml` values independently — always change `mission.py` first.**

### Detection filters

| Constant | Value | What it rejects |
|---|---|---|
| `FILL_RATIO_MIN_CIRCLE` | `0.40` | Hollow rings (H_landing), court arcs |
| `MAX_FIRE_DIAMETER_RATIO` | `0.55` | Giant court boundary arc lines |
| `H_LANDING_OVERLAP_THRESH` | `0.45` | Circles detected inside H_landing bbox |
| `COLOR_MATCH_THRESHOLD` | `0.25` | Weak HSV colour matches |
| `EDGE_CLIP_MARGIN_PX` | `5 px` | Circles partially off-screen (radius untrustworthy) |

### Altitude gate (drop safety)

| Constant | Value | Meaning |
|---|---|---|
| `DROP_ALTITUDE_MIN` | `3.05 m` | Lowest safe drop height (3.0 m rule + 5 cm margin) |
| `DROP_ALTITUDE_TARGET` | `3.25 m` | Optimal drop altitude |
| `DROP_ALTITUDE_MAX` | `4.50 m` | Too high — ball drift becomes unacceptable |

> **`DROP_ALTITUDE_MIN = 3.05 m` is a competition rule, not a tuning parameter.**
> Never lower below 3.05 m. Violation: −5 pts per 5 seconds below 3.0 m.

### Attitude gate (drop safety)

| Constant | Value | Meaning |
|---|---|---|
| `ALIGN_PIXEL_THRESH` | `25 px` | Max centroid offset from frame centre before drop |
| `DROP_MAX_PITCH_DEG` | `8.0°` | Max pitch for ball release |
| `DROP_MAX_ROLL_DEG` | `8.0°` | Max roll for ball release |
| `FF_ANGULAR_RATE_GATE_DPS` | `12.0°/s` | Feedforward suppression — drone actively pitching |

At 8° pitch + 3.5 m altitude, a released ball lands ~49 cm off-centre — outside the F5 target radius.  
Tighten to 5° for higher accuracy requirements.

### Velocity gains

| Constant | Value | Meaning |
|---|---|---|
| `PX_TO_MS_FIRE` | `0.005` | Pixel error → m/s for fire circle targeting |
| `PX_TO_MS_HELIPAD` | `0.004` | Pixel error → m/s for helipad approach |

### Match timing

| Constant | Value | Meaning |
|---|---|---|
| `LAND_EARLY_AT_S` | `285.0 s` | Force `FIND_HELIPAD` transition (15 s before T=300 s) |
| `HELIPAD_FORCE_LAND_S` | `5.0 s` | Emergency 1.0 m/s descent when remaining time < this |
| `MATCH_TIME_S` | `300 s` | Total match time (5 minutes) |

### Exclusion zones

| Constant | Value | Meaning |
|---|---|---|
| `SCORED_WORLD_RADIUS_M` | `0.60 m` | Altitude-invariant exclusion radius around already-scored circles |
| `SCORED_RADIUS_PX` | `80 px` | Legacy reference only — code uses world-space version |

### Drop timing

| Constant | Value | Meaning |
|---|---|---|
| `drop_cooldown_s` | `2.0 s` | Minimum gap between successive ball releases (prevents double-drop) |

### ToF spike filter

| Constant | Value | Meaning |
|---|---|---|
| `tof_spike_threshold_m` | `1.5 m` | Max deviation from median before rejection as spike |
| `tof_spike_reset_count` | `4` | Consecutive rejections before accepting step-change |

---

## 10. Competition Day Checklist

### Hardware checks (before arriving at venue)

```
□ calibration.npz generated from actual IMX219 camera (not synthetic)
□ Reprojection error < 1.0 px (printed during camera_calibration.py calibrate)
□ best.hef compiled with real flight frames (not synthetic) for INT8 calibration
□ BNO080 responds on I2C: python -c "import adafruit_bno08x; print('OK')"
□ MTF-01 telemetry visible in MAVProxy: DISTANCE_SENSOR, OPTICAL_FLOW_RAD
□ ArduPilot SERIAL_PROTOCOL=1 set for MTF-01 port
□ Servo AUX output tested with manual CommandLong: ball releases cleanly
```

### Software checks (at venue, before calibration)

```
□ Verify DROP_ALTITUDE_MIN = 3.05 in config.yaml (hard-floor override activates if < 3.0)
□ Verify model is yolo11n-seg (nano) — not s/m/l
□ Verify mask_ratio = 4 in config.yaml
□ Confirm --config-yaml config.yaml is in the test_video.py command line — without it, config.yaml edits do NOT apply
□ Check startup log shows "[CONFIG] Loaded N constants from config.yaml" before "Processing..."
□ Test HSV calibrator on live camera before tuning: python hsv_calibrator.py --source 0
```

### HSV calibration (at venue, 15 min before match)

```
□ Lock AWB and AEC on camera before calibrating (V4L2)
□ Hold red, yellow, blue fire circle samples in frame — tune each colour
□ Save with [S] key → hsv_calib.json
□ Validate with postprocess.py on a sample frame — confirm correct colour labels
```

### Final pre-flight checks

```
□ HUD shows [TOF] or [FC] as altitude source — NEVER [VIS] during flight
□ Test run with: python test_video.py --weights best.pt --source 0
                 --calib calibration.npz --hsv-calib hsv_calib.json
                 --config-yaml config.yaml --log run_001.csv
□ Confirm drop only triggers when HUD shows DROPPING (not ALIGN or DESCEND)
□ Confirm ball sequence: F1(red) → F2(red) → F3(yellow) → F4(yellow) → F5(blue)
□ Confirm Finals flag: --finals added if competing in Finals round
□ Confirm takeoff altitude: --takeoff-alt 3.3 (or mission-specific value)
□ fc_alt, sensors wired to real hardware (NOT stubs) in test_video.py
```

### Retry procedure (referee signals retry)

```python
# Standalone:
mission.reset()
mission.skip_takeoff()  # only if drone is already at altitude

# ROS 2:
ros2 service call /mission/reset std_srvs/srv/Trigger
```

> **Always call `mission.reset()` before retry.** Without it, score, phase, PIDs,
> ball count, and penalty counter remain dirty from the previous attempt.

---

## 11. Post-Flight Telemetry Analysis

### Standalone mode

```python
from logger import TelemetryLogger

# Wire during flight
tlog = TelemetryLogger("run_001.csv")
# (inside loop) tlog.record(frame_idx, inference_ms, result, mission)

# After flight
tlog.close()
metrics = tlog.summary()
# Prints: Duration, Mean/Min FPS, Mean/Max latency, Balls dropped, Drop triggers,
#         Landing frames, Recovery frames, Final score, CSV path
# Returns dict: {mean_fps, min_fps, mean_lat_ms, max_lat_ms, balls, score, frames, duration_s}
```

### CSV columns

| Column | Description |
|---|---|
| `frame_idx` | Frame counter |
| `wall_time_s` | Elapsed wall-clock time from start |
| `inference_ms` | YOLO inference time in milliseconds |
| `fps` | Rolling 30-frame FPS |
| `n_detections` | Number of detections this frame |
| `phase` | Mission phase name |
| `action` | Current flight action (ALIGN, DESCEND, DROP, etc.) |
| `balls_dropped` | Cumulative balls released |
| `score` | Current competition score |
| `top_class` | Class of highest-confidence detection |
| `top_conf` | Confidence score of top detection |
| `top_d_ratio` | Diameter ratio of top detection |
| `cmd_vx/vy/vz` | Velocity command (m/s) |
| `cmd_yaw_rate` | Yaw rate command |
| `cmd_drop_ball` | 1 if drop command was issued this frame |

### Performance thresholds

| Metric | Target | Warning trigger |
|---|---|---|
| Mean FPS | ≥ 20 Hz | `< 15` → consider `--skip-frames 1` or smaller model |
| Peak inference latency | < 150 ms | `> 150 ms` → check GPU/Hailo load |
| Mask mAP50-95 | ≥ 0.75 | Retrain with more data or step to `yolo11s-seg` |

---

## 12. Troubleshooting

### FPS below 20 Hz

1. Confirm `mask_ratio: 4` in `config.yaml` — if `2`, PCIe bus is saturated.
2. Confirm model is `yolo11n-seg` (not `s` or `m`).
3. Add `--skip-frames 1` to halve GPU load (processes every 2nd frame).
4. Check CPU governor: `sudo cpupower frequency-set -g performance`.

### HUD shows `[VIS]` during flight

FC barometer and MTF-01 ToF are not wired to the state machine. Wire sensor stubs:
- `current_altitude=fc_alt` from `mavlink_conn.altitude_m()`
- `sensor_data.tof_altitude_m` from MTF-01 rangefinder

### Altitude estimate wrong / premature drop below 3.0 m

1. Confirm `--calib calibration.npz` is passed — without it, `DEFAULT_F_PX = 500.0` is used.
2. Confirm FC/ToF altitude is wired (not vision-only fallback).
3. Check `DROP_ALTITUDE_MIN = 3.05` has not been lowered.

### Ball drops outside the target

1. Check attitude gate: `|pitch| < DROP_MAX_PITCH_DEG = 8°`, `|roll| < DROP_MAX_ROLL_DEG = 8°`.
   If violated, tighten to 5° or check drone oscillation (inner-loop PID tuning).
2. Check alignment gate: `ALIGN_PIXEL_THRESH = 25 px`. Confirm centre offset visible in HUD.
3. Confirm altitude is in `[DROP_ALTITUDE_MIN=3.05, DROP_ALTITUDE_TARGET=3.25]`.

### Colour detection wrong (red classified as yellow, etc.)

Run HSV calibration at the venue:
```bash
python hsv_calibrator.py --source 0 --output hsv_calib.json
```
Always pass `--hsv-calib hsv_calib.json` at startup. Without it, built-in defaults are used.

### State machine stuck in SEARCHING

1. Verify YOLO model detects fire circles: run `postprocess.py` on a sample frame.
2. Verify colour classification is correct (see above).
3. Check ball sequence: the current target colour must match the detected colour.
   Sequence: F1(red) → F2(red) → F3(yellow) → F4(yellow) → F5(blue).

### Double-drop (two balls released for one target)

Check `drop_cooldown_s = 2.0 s` in `config.yaml`. The cooldown prevents this; if the
servo is slow, increase to 3.0 s by updating the constant in `mission.py` and `config.yaml`.

### ROS 2: `drop_ball=True` command silently ignored

Check QoS on `/mission/flight_command`. Must be **Reliable**, not Best Effort.
Best Effort can silently drop packets — `drop_ball=True` frames may never arrive at
`action_mavros_node`, and the servo will not activate.

### ROS 2: wrong f_px / altitude estimates incorrect

Confirm `/camera/camera_info` is being published by `camera_node` and that
`perception_node` subscribes to it. Until `CameraInfo` is received, the fallback
`DEFAULT_F_PX = 500.0` is used with a warning log.

### ROS 2: match timer never starts

Confirm `mission_control_node` subscribes to `/mavros/state` and starts the timer
on `state.armed == True`. If the timer starts at node launch, the drone has extra
time budget before the real match clock begins — the forced `FIND_HELIPAD` at 285 s
will trigger too early.

### Retry does not reset score / ball count

Call `mission.reset()` — or `/mission/reset` service in ROS 2 — before the next attempt.
Without it, the state machine retains all previous round data.
