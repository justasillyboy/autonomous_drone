# YOLO11-seg — UAV Fire-Fighting Competition Vision System

Autonomous instance segmentation for **UAV Fire-Fighting & Rescue** competition
(Rulebook: Bo_Luat_UAV_Cuu_Hoa_Cuu_Ho_27_02_2026).
Hardware target: Raspberry Pi 5 + Hailo-8L (13 TOPS) + IMX219 wide-angle camera.

---

## Project structure

```
yolo11_circle_seg/
├── dataset.py            # Synthetic data generation (all 8 realism fixes)
├── config.yaml           # Training hyperparameters + competition rule constants
├── loss.py               # Boundary-aware BCE + Dice + Focal auxiliary loss
├── train.py              # Training entry point (loss hooked before epoch 1)
├── evaluate.py           # Boundary F1, Hausdorff, circularity, per-class metrics
├── export.py             # ONNX / TensorRT / CoreML / OpenVINO + Hailo calibration
├── mission.py            # Full competition state machine (Mission 2 + Mission 3)
├── postprocess.py        # Single-image inference (imports from mission.py)
├── test_video.py         # Video/camera inference — SM + FPS + telemetry
├── camera_calibration.py # IMX219 undistortion + focal length extraction
├── hsv_calibrator.py     # Real-time HSV colour tuning tool (run at the venue)
└── logger.py             # Runtime telemetry CSV (FPS, latency, phase, score, FlightCommand)
```

---

## Competition rules quick-reference (rulebook 27/02/2026)

### Field layout

| Dimension | Value |
|---|---|
| Total field | 24.0 × 12.0 m |
| Safe Zone | 12.0 × 3.5 m (helipad is here) |
| Manual flight ceiling | ≤ 5.0 m |
| Autonomous flight window | **3.0 – 5.0 m** |
| Ball drop rule | Lowest UAV point ≥ **3.0 m** above floor |

### TZ ring obstacles (Round 1 — rescue)

| Parameter | Value |
|---|---|
| Inner diameter | 1.6 m |
| Ring centre height | 1.7 m above floor |
| Total stand height | 2.6 m |

### Fire circle targets

| ID | Colour | Hex | Diameter | Height above floor |
|---|---|---|---|---|
| F1, F2 | Red | `#FC2A1B` | 1.0 m | 0.1 m (platform) |
| F3, F4 | Yellow | `#FAFF37` | 0.6 m | 0.1 m (platform) |
| F5 | Blue | `#3C00FB` | 0.4 m | 0.1 m (platform) |

> Fire circles are mounted on a 0.1 m platform. Altitude sensor reads from floor.
> Drop height rule (≥ 3.0 m) is measured from floor, so `DROP_ALTITUDE_MIN = 3.05 m` is correct.

### Scoring

| Task | Qualifier / Semi | Finals |
|---|---|---|
| Auto takeoff (≥ 3.0 m) | **20 pts** | **10 pts** |
| F1 red 1.0 m (static) | 20 pts | 10 pts |
| F2 red 1.0 m (static / **MOVING 10 cm/s**) | 20 pts | **50 pts** |
| F3 yellow 0.6 m | 30 pts | 30 pts |
| F4 yellow 0.6 m | 30 pts | 30 pts |
| F5 blue 0.4 m | 50 pts | 50 pts |
| Auto landing (inside helipad) | **30 pts** | **20 pts** |
| **Round 2 total** | **200 pts** | **200 pts** |

### Penalties (§VII)

| Violation | Penalty |
|---|---|
| UAV touches TZ gate frame | **−5 pts** per occurrence |
| Outside field boundary > 5 seconds | **−5 pts** per occurrence |
| Below 3.0 m during auto flight > **5 seconds** | **−5 pts** per occurrence |
| Ball outside target circle | 0 pts for that ball |
| Second ball into already-scored target | 0 pts for that ball |

### Timing (§XI)

| Parameter | Value |
|---|---|
| Total match time | **5 minutes** (Round 1 + Round 2 combined) |
| Prep time at field | **60 seconds** before clock starts |
| Post-time tasks | Not scored |

### Retry rules (§VII)

- Maximum **1 retry** per round
- Clock **does not stop** during retry
- **All scores from before the retry are reset to zero** (nothing carries over)
- Team must manually return drone to start position

```python
# In your MAVLink integration:
if referee_signals_retry:
    mission.reset()        # resets score, balls, phase, PIDs, penalty counter
    # fly drone back manually
    mission.skip_takeoff() # if already at altitude
```

### UAV hardware requirements (§III)

| Parameter | Limit |
|---|---|
| Max size (incl. props + cage) | 1.0 × 1.0 × 1.0 m |
| Max weight (incl. battery) | ≤ 5.0 kg |
| Battery type | LiPo / Li-ion max 6S |
| Max batteries | ≤ 3 |
| Max total capacity | ≤ 300 Wh |
| Max balls at takeoff | **5** |

---

## YOLO classes

| ID | Name | Colour | Competition role |
|---|---|---|---|
| 0 | `circle` | target colour | Fire circles F1–F5 |
| 1 | `semicircle` | Red | TZ obstacle ring outline |
| 2 | `H_landing` | Orange | Helipad in Safe Zone |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate dataset (3 000 images recommended)
python dataset.py --output data/synthetic --n-images 3000

# 2. Train  (custom loss hooked before epoch 1)
python train.py --config config.yaml --boundary-loss

# 3. Calibrate IMX219 camera (extracts f_px for altitude estimator)
python camera_calibration.py calibrate --source 0 --rows 9 --cols 6 \
       --square-mm 25.0 --output calibration.npz

# 4. Evaluate
python evaluate.py --weights runs/circle_seg/yolo11_v1/weights/best.pt \
                   --data data/synthetic/circle_seg.yaml --config config.yaml

# 5. Export for Hailo-8 deployment
python export.py --weights best.pt --format onnx --validate --benchmark
# Extract real calibration frames for Hailo INT8 quantization:
python export.py --extract-calib-frames flight.mp4 \
                 --calib calibration.npz --calib-output hailo_calib/

# 6. Tune HSV colour thresholds at the venue (run 15 min before the match)
#    Position the drone above the fire circles, let AWB/AEC settle, then lock them.
python hsv_calibrator.py --source 0 --output hsv_calib.json
#    Controls: [R] Red  [Y] Yellow  [B] Blue  [S] Save  [Q] Quit

# 7. Run on pre-recorded video (--skip-takeoff since drone already airborne)
python test_video.py --weights best.pt \
       --source flight.mp4 --output result.mp4 \
       --calib calibration.npz --log run_001.csv \
       --hsv-calib hsv_calib.json --skip-takeoff

# 8. Run on live camera (real competition — TAKEOFF phase active)
python test_video.py --weights best.pt \
       --source 0 --output result.mp4 \
       --calib calibration.npz --log run_001.csv \
       --hsv-calib hsv_calib.json --takeoff-alt 3.3 --finals

# 9. Annotate a single frame (field-tuned HSV is respected)
python postprocess.py --weights best.pt --image frame.jpg --output annotated.jpg \
       --hsv-calib hsv_calib.json
```

---

## Mission state machine

```
TAKEOFF → SEARCHING → TARGETING → DROPPING ─┐
               ↑                              │ × 5 balls
               └──────── RECOVERY ◄──────────┘
                                              │ (all balls)
                        FIND_HELIPAD → LANDING → COMPLETE
```

| Phase | Entry | `FlightCommand` output |
|---|---|---|
| `TAKEOFF` | System start | `vz=−0.8` until `current_altitude ≥ takeoff_alt` → awards takeoff bonus |
| `SEARCHING` | After takeoff / drop / recovery | Hover, `yaw_lock=True` |
| `TARGETING` | Colour confirmed | PID `vx/vy`, `vz` descend to drop window |
| `DROPPING` | `3.05 m ≤ Z ≤ 4.5 m` AND offset `< 25 px` AND `\|pitch\| < 8°` AND `\|roll\| < 8°` | `drop_ball=True` |
| `RECOVERY` | Target lost 10 frames | Yaw sweep `+30°/s → −60°/s → +30°/s → 0°/s` |
| `FIND_HELIPAD` | All 5 balls dropped | Hover + yaw scan if helipad lost |
| `LANDING` | H_landing `d_ratio ≥ 0.70` | `vz=0.4` descend, proportional XY |
| `COMPLETE` | `signal_landed()` called | All zeros — awards landing bonus |

### Reading `FlightCommand` in your MAVLink loop

> **`fps_hint` must match your actual pipeline FPS.**
> All frame-count thresholds — altitude penalty window (5 s × fps), drop cooldown (2 s × fps),
> recovery timer (3 s × fps), tilt-relax counter (1.5 s × fps) — are computed as
> `seconds × mission.fps_hint`. If `fps_hint=30` but your pipeline runs at 20 Hz, the altitude
> penalty fires every 3.3 s instead of 5 s, causing premature −5 pt penalties.
> `test_video.py` sets this automatically from `cap.get(cv2.CAP_PROP_FPS)`.
> On live camera: verify actual throughput with `--log` and inspect the `fps` column in the CSV.

```python
from mission import MissionStateMachine, Phase, load_mission_config_yaml

# Optional: load tunable constants from config.yaml before constructing the state machine.
# Must be called BEFORE MissionStateMachine() — FF_HISTORY_LEN and ToF filter constants
# are read at __init__ time and won't update if load_mission_config_yaml() is called after.
load_mission_config_yaml("config.yaml")

mission = MissionStateMachine(f_px=490.0, is_finals=False, takeoff_alt=3.3)
mission.fps_hint = cap.get(cv2.CAP_PROP_FPS)   # REQUIRED for correct timing gates

while True:
    frame, result = camera.grab()
    alt   = mavlink.altitude_m()   # REQUIRED: barometer/rangefinder from FC
    frame = mission.update(frame, result, current_altitude=alt)

    cmd = mission.last_command
    mavlink.set_velocity_body(vx=cmd.vx, vy=cmd.vy, vz=cmd.vz,
                               yaw_rate=cmd.yaw_rate)
    if cmd.drop_ball:
        servo.release()
    if mission.phase == Phase.COMPLETE:
        mavlink.disarm()

    # Check net score (score + penalties)
    print(f"Score: {mission.score}  Penalties: {mission.penalties}  "
          f"Net: {mission.score + mission.penalties}")
```

---

## Sensor integration (MTF-01 + BNO080)

`mission.py` accepts an optional `SensorData` dataclass on every `update()` call.
All fields are `None` by default — the state machine degrades gracefully when a sensor is absent.

### MTF-01 (laser ToF rangefinder + optical flow)

| Field | Type | Description |
|---|---|---|
| `tof_altitude_m` | `float` | Laser altitude above floor in metres (±1 cm). Primary altitude source. |
| `of_vx_ms` | `float` | Drone body-frame forward velocity m/s from optical flow. |
| `of_vy_ms` | `float` | Drone body-frame lateral velocity m/s from optical flow. |

**Wiring:** Connect MTF-01 to an ArduPilot serial port (`SERIAL_PROTOCOL=1`). ArduPilot rebroadcasts the data as MAVLink `DISTANCE_SENSOR` and `OPTICAL_FLOW_RAD` messages on the companion-computer telemetry stream.

```python
# Read ToF altitude (pymavlink)
msg = conn.recv_match(type='DISTANCE_SENSOR', blocking=False)
tof_alt = msg.current_distance / 100.0          # cm → m

# Read optical-flow velocity (pymavlink)
msg = conn.recv_match(type='OPTICAL_FLOW_RAD', blocking=False)
flow_vx = msg.integrated_x / msg.integration_time_us * 1e6   # rad/s
of_vx_ms = flow_vx * tof_alt                   # rad/s × altitude → m/s
```

**Why optical flow matters:** The feedforward path for the Finals F2 moving target estimates pure *target* velocity from the pixel history. Without optical flow compensation the history also contains drone-motion, producing a corrupted feedforward signal. Provide `of_vx_ms` / `of_vy_ms` whenever available.

### BNO080 (IMU — attitude + angular rate)

| Field | Type | Description |
|---|---|---|
| `pitch_deg` | `float` | Nose-up pitch in degrees (positive = nose up). |
| `roll_deg` | `float` | Roll in degrees (positive = right wing down). |
| `yaw_deg` | `float` | Yaw in degrees clockwise from north. |
| `pitch_rate_dps` | `float` | Pitch angular rate deg/s. Used for feedforward guard. |
| `roll_rate_dps` | `float` | Roll angular rate deg/s. Used for feedforward guard. |

**Wiring:** Connect BNO080 to I2C (default address `0x4A`) or UART on the Raspberry Pi 5.

```python
# Adafruit CircuitPython BNO08x library
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE

pitch, roll, yaw = euler_from_quaternion(bno.game_quaternion)
gx, gy, gz = bno.gyro    # rad/s body frame
```

**Why the IMU matters:**
- **Attitude gate:** a drop is refused when `|pitch| > 8°` or `|roll| > 8°`. At 8° pitch a ball released at 3.5 m lands ~49 cm off-centre — already at the edge of the 50 cm F5 radius. Without this gate, drops triggered during XY corrections will miss.
- **Feedforward guard:** feedforward is suppressed when `pitch_rate_dps > 12°/s` (`FF_ANGULAR_RATE_GATE_DPS`). When the drone is actively pitching, the camera is tilting and the pixel history is corrupted. The BNO080 angular rate detects this directly; the fallback proxy (`|cmd_vx| > 0.35 m/s`) is less reliable.

### Passing SensorData to the state machine

```python
from mission import MissionStateMachine, SensorData

mission = MissionStateMachine(f_px=490.0, is_finals=False, takeoff_alt=3.3)

while True:
    frame, result = camera.grab()

    sensors = SensorData(
        tof_altitude_m  = tof_sensor.read_m(),      # MTF-01
        of_vx_ms        = flow.vx_ms(),             # MTF-01 optical flow
        of_vy_ms        = flow.vy_ms(),
        pitch_deg       = imu.pitch_deg,             # BNO080
        roll_deg        = imu.roll_deg,
        pitch_rate_dps  = imu.pitch_rate_dps,
        roll_rate_dps   = imu.roll_rate_dps,
    )

    frame = mission.update(frame, result,
                           current_altitude=fc.altitude_m(),  # FC baro (highest priority)
                           sensor_data=sensors)
```

### Altitude source priority

The state machine selects the best available altitude each frame:

```
fc_altitude  (MAVLink GLOBAL_POSITION_INT / barometer)   ← highest priority
    ↓ if None
tof_altitude_m  (MTF-01 laser rangefinder)
    ↓ if None
Z_est_vision  (f_px × D_real / d_px from YOLO mask)      ← fallback, least reliable
```

The HUD labels the active source: `[FC]`, `[TOF]`, or `[VIS]`.
**Always connect at least a ToF sensor for competition flights.** Vision-only altitude is unreliable when the drone pitches during XY correction, because the circle projects as an ellipse and `d_px` is inflated, underestimating `Z_est` and risking a sub-3.0 m drop.

---

## HSV colour calibration (`hsv_calibrator.py`)

Competition venue lighting differs from lab conditions. AWB and AEC shifts can move the red hue from H=0–10 to H=15–25, making the hardcoded defaults in `mission.py` ineffective mid-flight.

**Recommended pre-match procedure (15 min before):**

1. Position the drone above the fire circles (the brightest spot on the field).
2. Let the camera auto-expose for 2–3 seconds, then lock AWB and AEC (`hsv_calibrator.py` does this automatically via V4L2 on Linux).
3. Run the calibrator and drag the trackbars until only the target circle lights up:

```bash
python hsv_calibrator.py --source 0 --output hsv_calib.json
# Controls: [R] Red  [Y] Yellow  [B] Blue  [S] Save  [Q] Quit
```

4. Pass the saved file to `test_video.py` and `postprocess.py`:

```bash
python test_video.py  ... --hsv-calib hsv_calib.json
python postprocess.py ... --hsv-calib hsv_calib.json
```

The JSON is loaded by `mission.load_hsv_calibration()` before the first inference frame, overriding the hardcoded `COLOR_HSV` defaults for the current process. The defaults remain unchanged on disk.

---

## Altitude-based drop control

Vision `Z_est = f_px × D_real / d_px` is the fallback when no sensor is connected.
**When a rangefinder/barometer is connected, it is used as the authoritative altitude source.**
This is critical: drone pitch during XY corrections projects the circle as an ellipse,
inflating `d_px` and underestimating `Z_est`, potentially triggering a drop below 3.0 m.

The HUD shows `[FC]` (flight-controller baro/rangefinder), `[TOF]` (MTF-01 laser), or `[VIS]` (vision fallback) so you can verify which altitude source is active during testing.

| Circle | D_real | Safe drop window |
|---|---|---|
| Red F1/F2 | 1.0 m | 3.05 – 4.50 m |
| Yellow F3/F4 | 0.6 m | 3.05 – 4.50 m |
| Blue F5 | 0.4 m | 3.05 – 4.50 m |

| Constant | Value | Meaning |
|---|---|---|
| `DROP_ALTITUDE_TARGET` | `3.25 m` | Optimal approach altitude (accuracy + safety margin above 3.05 m floor) |
| `DEFAULT_F_PX` | `500.0 px` | Focal length fallback — always override with `camera_calibration.py` output |
| `SCORED_WORLD_RADIUS_M` | `0.60 m` | World-space exclusion radius around already-scored circles (altitude-invariant) |

### Alignment and drop-gate thresholds

| Constant | Value | Meaning |
|---|---|---|
| `ALIGN_PIXEL_THRESH` | `25 px` | Maximum centroid offset from frame centre before drop is permitted |
| `DROP_THRESHOLD` | `0.70` | H_landing bbox width / frame width ratio to transition APPROACH → LAND |
| `DROP_MAX_PITCH_DEG` | `8.0°` | Maximum `\|pitch\|` for ball release (at 8° pitch, ball lands ~49 cm off-centre at 3.5 m) |
| `DROP_MAX_ROLL_DEG` | `8.0°` | Maximum `\|roll\|` for ball release |

---

## False-positive suppression

| Filter | Constant | Rejects |
|---|---|---|
| Fill-ratio | `FILL_RATIO_MIN_CIRCLE = 0.40` | Hollow arcs, H_landing ring |
| Size cap | `MAX_FIRE_DIAMETER_RATIO = 0.55` | Court arc boundary lines |
| Cross-class NMS | `H_LANDING_OVERLAP_THRESH = 0.45` | Circles inside H_landing bbox |
| Edge clip | `EDGE_CLIP_MARGIN_PX = 5` | Partial circles at frame boundary (altitude untrustworthy) |
| Colour match | `COLOR_MATCH_THRESHOLD = 0.25` | Detections whose HSV colour score falls below this fraction |

---

## Temporal smoothing & feedforward (Finals F2)

- `smoothing_alpha=0.4` for Qualifier (static targets)
- Auto-lowered to `0.15` when `--finals` — reduces phase lag on 10 cm/s F2
- Feedforward velocity from `_ff_history_len=8` frames of position history compensates for target motion
- **Feedforward is suppressed when BNO080 angular rate `> 12°/s`** (`FF_ANGULAR_RATE_GATE_DPS`) — drone is actively pitching and camera tilt would corrupt the pixel history
- Falls back to `|cmd_vx| > FF_CMD_THRESHOLD_MS` proxy when BNO080 is not connected

| Constant | Value | Meaning |
|---|---|---|
| `FF_SCALE` | `0.003` | Target velocity (px/s) → drone additive velocity (m/s) |
| `MAX_FF_PX_S` | `60.0 px/s` | Clamp on feedforward estimate — above this indicates noise/tilt |
| `FF_CMD_THRESHOLD_MS` | `0.35 m/s` | Maneuvering proxy gate when BNO080 absent |
| `FF_ANGULAR_RATE_GATE_DPS` | `12.0°/s` | IMU-based feedforward suppression gate |
| `TILT_MAX_RELAXED_DEG` | `15.0°` | Relaxed attitude gate after 1.5 s crosswind tilt-block |
| `DROP_COOLDOWN_S` | `2.0 s` | Minimum interval between ball releases |

---

## ToF median filter

The MTF-01 rangefinder is filtered through a 5-sample rolling median before use as an altitude source. Individual spike readings (e.g. TZ gate passing under the drone) are rejected when they deviate from the running median by more than `_tof_spike_threshold_m`. After `_tof_spike_reset_count` consecutive rejections the filter assumes a genuine terrain step-change and re-seeds the buffer.

| Constant | Value | Meaning |
|---|---|---|
| `_tof_spike_threshold_m` | `1.5 m` | Max deviation from median before a reading is discarded |
| `_tof_spike_reset_count` | `4 frames` | Consecutive rejections before step-change flush |

---

## Match timing

| Constant | Value | Meaning |
|---|---|---|
| `MATCH_TIME_S` | `300 s` | Total match window |
| `LAND_EARLY_AT_S` | `285 s` | Force `FIND_HELIPAD` transition (15 s margin) |
| `ALT_PENALTY_TIME_S` | `5 s` | Seconds below 3.0 m before −5 pt altitude penalty |
| `HELIPAD_FORCE_LAND_S` | `5.0 s` | Remaining-time threshold that triggers emergency 1.0 m/s descent |

---

## Recovery behaviour

Fire target lost → `RECOVERY` phase:
1. Yaw sweep: `+30°/s` × ~0.67 s → `−60°/s` × ~0.67 s → `+30°/s` × ~0.67 s → `0°/s` (each step is `int(0.67 × fps)` frames — fps-adaptive)
2. If target reacquired → `TARGETING`, PIDs + history reset
3. Full sweep without detection → skip target, advance to next

Same yaw-sweep logic applies to `FIND_HELIPAD` after 10 missed frames.

---

## Telemetry (`logger.py`)

CSV columns per frame:
```
frame_idx, wall_time_s, inference_ms, fps, n_detections,
phase, action, balls_dropped, score,
top_class, top_conf, top_d_ratio,
cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate, cmd_drop_ball
```
Use `--log run_001.csv`. After flight: `tlog.summary()` prints FPS stats, drop triggers,
recovery frames, final score, and total penalties.

---

## Model size guide (Hailo-8L deployment)

**Hardware: Hailo-8L = 13 TOPS** (half of Hailo-8 at 26 TOPS).

| Model | Hailo-8L fit | Notes |
|---|---|---|
| `yolo11n-seg` | **Recommended** | Fits 13 TOPS budget. ~2 ms inference, stable FPS. |
| `yolo11s-seg` | Borderline | Only if n mAP < 0.75 after training. |
| `yolo11m-seg` | **Do not use** | Overloads 13 TOPS PCIe transfer bus. |
| `yolo11l-seg` | **Do not use** | Will not run in real-time. |

Why `n` is sufficient: fire targets are large, solid-color, high-contrast discs
on a uniform gray floor. The nano backbone has more than enough capacity.
Reserve `s` only as a fallback if validation mAP falls below 0.75.

Export path: `best.pt → ONNX → Hailo DFC (.hef INT8)`.
Use **real flight frames** (post-undistort, `crop=True`) for Hailo calibration, not synthetic.

---

## Research targets

| Metric | Target |
|---|---|
| Mask mAP50-95 | ≥ 0.75 |
| Boundary F1 (2 px tol.) | ≥ 0.85 |
| Hausdorff distance | ≤ 3 px |
| Predicted mask circularity | ≥ 0.90 |

---

## FC inner-loop PID reference (hardware team, 2026-04)

These are ArduPilot `ATC_RAT` inner-loop gains — **not our Python PID**.
They run at 400 Hz on gyro feedback inside the FC.

| Parameter | Value | Unit |
|---|---|---|
| `ATC_RAT_PIT_P` / `ATC_RAT_RLL_P` | 0.135 | s/rad |
| `ATC_RAT_PIT_I` / `ATC_RAT_RLL_I` | 0.135 | 1/rad |
| `ATC_RAT_PIT_D` / `ATC_RAT_RLL_D` | 0.0036 | s²/rad |

Our Python outer-loop PID (`kp=0.5, ki=0.01, kd=0.15`) converts pixel error → m/s setpoint.

| Constant | Value | Meaning |
|---|---|---|
| `PX_TO_MS_FIRE` | `0.005` | Pixel error → m/s gain for fire-circle targeting (100 px → 0.5 m/s) |
| `PX_TO_MS_HELIPAD` | `0.004` | Pixel error → m/s gain for helipad approach (slower, more precise) |

`PX_TO_MS_FIRE = 0.005` was chosen so 100 px error → 0.5 m/s, which the FC tracks cleanly
with P=0.135 in the inner loop. See `mission.py` constants block for derivation.
