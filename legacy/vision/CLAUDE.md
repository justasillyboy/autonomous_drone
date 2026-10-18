# CLAUDE.md — Project Intelligence for UAV Fire-Fighting Vision System

## Project Identity

Autonomous vision & control system for the **UAV Fire-Fighting & Rescue Competition 2026**
(rulebook: `Bo_Luat_UAV_Cuu_Hoa_Cuu_Ho_27_02_2026.pdf`).

Hardware target: **Raspberry Pi 5 + Hailo-8L (13 TOPS) + IMX219 wide-angle camera**.
Control: **ArduPilot** (inner PID loops) + this Python stack (outer PID + state machine).

Formal requirements document: `srs.md` (Software Requirements Specification).

---

## MANDATORY RULE — Synchronization Invariant

> **Every time you introduce or modify a significant constant, threshold, parameter, or
> logical rule anywhere in the codebase, you MUST update BOTH `config.yaml` AND `README.md`
> in the same change. No exceptions.**

This rule exists because:
- `config.yaml` is the on-site tuning surface — judges and operators read it to understand what the drone will do.
- `README.md` is the team's operational reference — field members rely on it during the match.
- Constants scattered only in Python source are invisible to non-programmers on the team.

**Practical checklist for any change:**
1. Change the Python constant/logic.
2. Find the corresponding entry in `config.yaml` and update it (add one if missing).
3. Find the corresponding entry in `README.md` (table, bullet, or code block) and update it.
4. If the change affects mission behavior, update `srs.md` requirements as well.

---

## File Architecture & Ownership

### Dependency graph (imports flow downward)

```
config.yaml           ← on-site tuning, human-readable snapshot of all constants
srs.md                ← requirements specification (update alongside code)
README.md             ← operational reference (update alongside code)

dataset.py            ← CLASS_NAMES / CLASS_IDS authority; no upstream deps
loss.py               ← pure PyTorch; no project deps
camera_calibration.py ← pure OpenCV; no project deps
hsv_calibrator.py     ← reads/writes JSON; imports mission only for DEFAULT_RANGES mirror

mission.py            ← SINGLE SOURCE OF TRUTH for all runtime constants
                         imports: nothing from this project
                         exports: constants, FlightCommand, SensorData,
                                  MissionStateMachine, AltitudeEstimator, helpers

train.py              ← imports: loss.py
evaluate.py           ← imports: dataset.py (CLASS_NAMES)
export.py             ← imports: camera_calibration.py (optional)
postprocess.py        ← imports: mission.py (everything)
test_video.py         ← imports: mission.py, logger.py
logger.py             ← imports: nothing from this project
```

### Who owns what

| Truth | Authoritative file | Consumers |
|---|---|---|
| Class names & IDs | `dataset.py` (`CLASS_NAMES`, `CLASS_IDS`) | `evaluate.py`, `mission.py` (names only) |
| All runtime constants | `mission.py` | `postprocess.py`, `test_video.py` |
| Focal length fallback | `mission.py` (`DEFAULT_F_PX = 500.0`) | `test_video.py` (imported), `postprocess.py` (imported) |
| HSV colour ranges | `mission.py` (`COLOR_HSV`) | `hsv_calibrator.py` (mirrored as `DEFAULT_RANGES`) |
| Training hyperparams | `config.yaml` + `train.py` (`CONFIG_DEFAULTS`) | `train.py`, `evaluate.py --config` |
| Competition rule constants | `mission.py` + documented in `config.yaml` | everywhere |

---

## Critical Constants Registry

All constants below exist **twice**: authoritative in Python, documented in `config.yaml`.
If you change one, change both.

### Detection filters (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | What it rejects |
|---|---|---|---|
| `FILL_RATIO_MIN_CIRCLE` | `0.40` | `fill_ratio_min_circle` | Hollow arcs, H_landing rings |
| `MAX_FIRE_DIAMETER_RATIO` | `0.55` | `max_fire_diameter_ratio` | Court boundary arc lines |
| `H_LANDING_OVERLAP_THRESH` | `0.45` | `h_landing_overlap_thresh` | Circles inside H_landing bbox |
| `DROP_THRESHOLD` | `0.70` | `drop_threshold` | H_landing approach gate |
| `COLOR_MATCH_THRESHOLD` | `0.25` | `color_match_threshold` | HSV colour confirmation |

### Altitude control (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `DROP_ALTITUDE_MIN` | `3.05 m` | `drop_altitude_min` | Lowest safe drop height (>3.0 m rule) |
| `DROP_ALTITUDE_TARGET` | `3.25 m` | `drop_altitude_target` | Optimal approach altitude |
| `DROP_ALTITUDE_MAX` | `4.50 m` | `drop_altitude_max` | Too high, ball drift too large |
| `DEFAULT_F_PX` | `500.0 px` | `default_f_px` | Focal length fallback (always override) |

### Attitude gates — drop safety (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `ALIGN_PIXEL_THRESH` | `25 px` | `align_pixel_thresh` | Max pixel offset before drop |
| `DROP_MAX_PITCH_DEG` | `8.0°` | `drop_max_pitch_deg` | Max pitch for release |
| `DROP_MAX_ROLL_DEG` | `8.0°` | `drop_max_roll_deg` | Max roll for release |
| `FF_ANGULAR_RATE_GATE_DPS` | `12.0°/s` | `ff_angular_rate_gate_dps` | Feedforward suppression threshold |

### Exclusion zones (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `SCORED_RADIUS_PX` | `80 px` | `scored_radius_px` | Legacy pixel exclusion (reference only) |
| `SCORED_WORLD_RADIUS_M` | `0.60 m` | `scored_world_radius_m` | Altitude-invariant exclusion zone |

### Velocity gains (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `PX_TO_MS_FIRE` | `0.005` | `px_to_ms_fire` | Pixel error → m/s for fire targeting |
| `PX_TO_MS_HELIPAD` | `0.004` | `px_to_ms_helipad` | Pixel error → m/s for helipad approach |

### Feedforward & timing (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `FF_SCALE` | `0.003` | `ff_scale` | Target velocity px/s → drone m/s (Finals F2 only) |
| `MAX_FF_PX_S` | `60.0 px/s` | `max_ff_px_s` | Feedforward clamp — above this is noise/tilt |
| `FF_CMD_THRESHOLD_MS` | `0.35 m/s` | `ff_cmd_threshold_ms` | Maneuvering proxy (suppresses FF when BNO080 absent) |
| `FF_HISTORY_LEN` | `8 frames` | `ff_history_len` | Position history length for linear-velocity fit |
| `TILT_MAX_RELAXED_DEG` | `15.0°` | `tilt_max_relaxed_deg` | Relaxed attitude gate after 1.5 s tilt-block |
| `DROP_COOLDOWN_S` | `2.0 s` | `drop_cooldown_s` | Min interval between ball releases (prevents double-drop) |
| `EDGE_CLIP_MARGIN_PX` | `5 px` | `edge_clip_margin_px` | Frame-edge margin — radius untrustworthy if clipped |
| `LAND_EARLY_AT_S` | `285.0 s` | `land_early_at_s` | Force FIND_HELIPAD (15 s margin before end) |
| `HELIPAD_FORCE_LAND_S` | `5.0 s` | `helipad_force_land_s` | Emergency 1.0 m/s descent threshold |

### ToF filter (`mission.py` → `config.yaml`)

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `TOF_SPIKE_THRESHOLD_M` | `1.5 m` | `tof_spike_threshold_m` | Max deviation from median before spike rejection |
| `TOF_SPIKE_RESET_COUNT` | `4 frames` | `tof_spike_reset_count` | Consecutive rejections before step-change flush |

---

## Competition Rules That Are Hard-Coded as Constants

These are not tuning parameters — they are competition rules. Do **not** relax them.

| Rule | Value | Where enforced |
|---|---|---|
| Minimum drop altitude | ≥ 3.0 m (use 3.05 m with margin) | `DROP_ALTITUDE_MIN` |
| Autonomous flight window | 3.0 – 5.0 m | `DROP_ALTITUDE_MIN`, `DROP_ALTITUDE_MAX` |
| Match time total | 300 s | `MATCH_TIME_S` — triggers forced `FIND_HELIPAD` at t=285 s |
| Altitude penalty threshold | < 3.0 m for > 5 s = −5 pts | `PENALTY_ALTITUDE`, `ALT_PENALTY_TIME_S = 5` |
| Max 1 retry per round | Score resets to zero on retry | `mission.reset()` |

**Referee-applied penalties (NOT self-tracked in code):**
TZ-gate collisions (−5 pts each) and field-boundary excursions (−5 pts per 5 s) are
rulebook penalties (§VII) that the drone cannot self-detect without sensors it does
not carry (impact sensor / GPS fence). Referees deduct them manually; the Python code
does not define constants for them.

**Note on drop order:** The sequence F1(red) → F2(red) → F3(yellow) → F4(yellow) → F5(blue)
encoded in `FIRE_TARGETS_QUALIFIER` / `FIRE_TARGETS_FINALS` is **team strategy, not a rule**.
Per rulebook §V (*"Thứ tự thực hiện các điểm cháy do đội thi tự quyết định"*), the team picks
the order. We chose ascending-difficulty: static red → yellow → blue, with F2 (moving in Finals)
kept second so it triggers before the drone commits to the smaller targets.

### Scoring tables (`mission.py` `FIRE_TARGETS_QUALIFIER` / `FIRE_TARGETS_FINALS`)

Per-target points — rulebook §VI. Finals changes F1 (20→10), F2 (20→50, moving 10 cm/s),
and landing (30→20); takeoff also drops 20→10.

| Target | Colour | Diameter | Qualifier | Finals |
|---|---|---|---|---|
| F1 | Red    | 1.0 m | 20 pts | 10 pts |
| F2 | Red    | 1.0 m | 20 pts | **50 pts** (MOVING 10 cm/s) |
| F3 | Yellow | 0.6 m | 30 pts | 30 pts |
| F4 | Yellow | 0.6 m | 30 pts | 30 pts |
| F5 | Blue   | 0.4 m | 50 pts | 50 pts |
| Takeoff (auto, ≥ 3.0 m) | — | — | 20 pts | 10 pts |
| Landing (auto, inside helipad) | — | — | 30 pts | 20 pts |

Bonus constants: `SCORE_TAKEOFF_QUAL / SCORE_TAKEOFF_FINALS`, `SCORE_LANDING_QUAL /
SCORE_LANDING_FINALS`.

### Circle physical diameters (`mission.py` `FIRE_REAL_DIAMETERS`)

Used by vision Z-estimation: `Z = f_px × D_real / d_px`. Single source of truth is the
4th slot of each `FIRE_TARGETS_QUALIFIER` tuple; `FIRE_REAL_DIAMETERS` is derived from
those tuples by comprehension and must not be hand-edited — update the tuple instead.

| Colour | `D_real` | Targets |
|---|---|---|
| Red    | 1.0 m | F1, F2 |
| Yellow | 0.6 m | F3, F4 |
| Blue   | 0.4 m | F5 |

### Field geometry (`mission.py` → `config.yaml`)

Rulebook §IV. Only these three are used in code (geofencing — see `_GEOFENCE_X_MAX`,
`_GEOFENCE_Y_M`, and the Safe-Zone RTH/scan heuristics). TZ-ring dimensions and
fire-circle mount height are documented in `config.yaml` and `README.md` as rulebook
reference but deliberately not carried as Python constants.

| Python constant | Value | config.yaml key | Meaning |
|---|---|---|---|
| `FIELD_LENGTH`    | `24.0 m` | `field_length_m`    | Along-track dimension |
| `FIELD_WIDTH`     | `12.0 m` | `field_width_m`     | Cross-track dimension |
| `SAFE_ZONE_DEPTH` | `3.5 m`  | `safe_zone_depth_m` | Safe Zone depth along field length |

### Penalty values (`mission.py` → `config.yaml`)

Rulebook §VII. Only the altitude penalty is self-tracked in code (onboard altitude is
always available). TZ and boundary penalties are referee-applied (see the rules table
above) and therefore intentionally have no Python constant.

| Python constant | Value | config.yaml key | Trigger |
|---|---|---|---|
| `PENALTY_ALTITUDE` | `−5` | `penalty_altitude` | Below 3.0 m during auto flight > 5 s |

---

## Architecture Decisions (Do Not Reverse Without Understanding)

### 1. `mission.py` is the single source of truth for constants
All runtime constants live in `mission.py`. `postprocess.py` and `test_video.py` **import** them — they never redefine them. If you find a literal number in those files that matches a known constant, replace it with an import. The previous version had `500.0` hardcoded in `test_video.py` — this was a bug.

### 2. Class names come from `dataset.py`, not string literals
`evaluate.py` derives `class_names` by iterating `CLASS_NAMES` from `dataset.py`. Never hardcode `{0: "circle", 1: "semicircle", 2: "H_landing"}` anywhere — if a class is renamed in `dataset.py`, the whole pipeline must update automatically.

### 3. Altitude source priority ladder
The state machine selects altitude in this order:
```
fc_altitude  (MAVLink GLOBAL_POSITION_INT)   ← always prefer
    ↓ if None
tof_altitude_m  (MTF-01 ToF laser)
    ↓ if None
Z_est_vision  (f_px × D_real / d_px)         ← fallback only, unreliable under pitch
```
**Never promote vision-estimated altitude** above a hardware sensor. Vision altitude is invalid when the drone pitches (circle projects as ellipse, `d_px` inflates, `Z_est` underestimates → premature sub-3.0 m drop).

### 4. Geometry extraction: contours on small mask, scale up
`_extract_geometry()` (in `mission.py`) runs `cv2.findContours` on the small YOLO output mask then scales coordinates to full frame. **Do not** `cv2.resize(mask, full_frame)` — this is the slow path that causes FPS drops on Pi 5 / Hailo-8L.

### 5. `mask_ratio: 4` is a hardware constraint, not a quality choice
`mask_ratio: 4` prevents PCIe bus saturation between Pi 5 and Hailo-8L. Setting it to `2` saturates the bus and drops FPS below 20 Hz. Only change this if running on a different hardware target.

### 6. `yolo11n-seg` (Nano) is the correct model for Hailo-8L
At 13 TOPS, only `n` fits the PCIe budget with stable FPS. `s` is borderline (use only if `mAP < 0.75` with `n`). `m` and larger **will not run in real-time** and must not be used.

### 7. Boundary + Dice auxiliary loss is added via callback, not subclass
The loss is patched onto the Ultralytics trainer via `on_train_start` callback (see `train.py`). The callback is registered **before** training starts — not after — because the Ultralytics trainer calls `init_criterion()` during `on_train_start`. The `loss.py` file also has a `patch_ultralytics_loss()` function, but `train.py`'s callback is the canonical path.

### 8. HSV calibration must be loaded before any inference frame
`mission.load_hsv_calibration(json_path)` must be called before `mission.update()` is ever invoked. In both `test_video.py` and `postprocess.py` this happens immediately after model load and before the video loop. The JSON from `hsv_calibrator.py` overrides `COLOR_HSV` for the current process only — the source file is unchanged.

### 9. Feedforward for Finals F2 is suppressed when drone is actively maneuvering
The F2 target moves at 10 cm/s. The feedforward term estimates target velocity from the pixel history. However, if the drone is pitching (angular rate > `FF_ANGULAR_RATE_GATE_DPS = 12°/s`), the pixel history contains camera tilt, not target motion. The BNO080 angular rate catches this directly; the fallback proxy (`|cmd_vx| > 0.35 m/s`) is less reliable but works without the IMU.

### 10. SCORED_WORLD_RADIUS_M is used in code; SCORED_RADIUS_PX is legacy
The actual exclusion zone for already-scored circles is computed in world space (`SCORED_WORLD_RADIUS_M = 0.60 m`). The pixel value `SCORED_RADIUS_PX = 80` is kept as a historical reference comment only. Don't add new logic using the pixel version.

---

## Development Workflow

### Adding a new competition constant

1. Define it in `mission.py` in the relevant constants block with a comment explaining the rule source.
2. Import it in any file that needs it (`postprocess.py`, `test_video.py`).
3. Add it to `config.yaml` under the appropriate section with a comment.
4. Add or update the relevant table/bullet in `README.md`.
5. If it maps to an SRS requirement, update `srs.md`.

### Adding a new CLI argument to `postprocess.py` or `test_video.py`

Check whether the parallel file already has this argument. These two files must be kept in sync:
- Both have `--conf`, `--calib`, `--f-px`, `--finals`, `--hsv-calib`, `--config-yaml`.
- `test_video.py` additionally has `--skip-frames`, `--smooth`, `--log`, `--takeoff-alt`, `--skip-takeoff` (video-specific features).
- If you add a detection/mission argument, add it to both files.

### Syncing `deploy_files_drone_competition/`

`deploy_files_drone_competition/` is the exact set of files copied to the drone SD card.
It is **not** a git submodule — it must be manually kept in sync with the root after every change.

```bash
# Run after any change to the following files:
cp files/mission.py     files/deploy_files_drone_competition/mission.py
cp files/test_video.py  files/deploy_files_drone_competition/test_video.py
cp files/postprocess.py files/deploy_files_drone_competition/postprocess.py
cp files/logger.py      files/deploy_files_drone_competition/logger.py
cp files/config.yaml    files/deploy_files_drone_competition/config.yaml

# Verify:
diff files/mission.py    files/deploy_files_drone_competition/mission.py
diff files/test_video.py files/deploy_files_drone_competition/test_video.py
```

The deploy directory does **not** contain `train.py`, `dataset.py`, `evaluate.py`, `loss.py`, or `export.py` — those are training-time files not needed on the drone.

### Modifying HSV colour ranges

1. Change `COLOR_HSV` in `mission.py`.
2. Change `DEFAULT_RANGES` in `hsv_calibrator.py` to match (format differs: dict with `"lo"/"hi"` lists vs tuple pairs).
3. Update `config.yaml` comment block `color_*_hex`.
4. Update `README.md` fire circle target table if the hex codes changed.

### Training pipeline changes

1. Change `config.yaml` first.
2. Ensure `train.py`'s `CONFIG_DEFAULTS` dict has matching defaults.
3. Run `python train.py --config config.yaml` to validate the config parses correctly.
4. After training, evaluate with `python evaluate.py --config config.yaml` (syncs confidence threshold).

---

## Danger Zones — Do Not Do These

| Danger | Why | Safe alternative |
|---|---|---|
| Hardcode `500.0` as focal length | Breaks altitude estimation on any other camera; was already a bug fixed once | Import `DEFAULT_F_PX` from `mission.py` |
| Use `cv2.resize(mask, full_frame)` | Slow on Pi 5 — defeats the whole point of using Hailo-8L | Use `_extract_geometry()` from `mission.py` |
| Set `drop_altitude_min < 3.05` | Violates competition rule — −5 pt penalty per 5s below 3.0 m | Never lower below 3.05 m |
| Use `mask_ratio: 2` on Pi 5 + Hailo-8L | Saturates PCIe bus, drops FPS | Keep `mask_ratio: 4` |
| Deploy `yolo11m-seg` or larger | Overloads 13 TOPS budget | Use `yolo11n-seg`; step to `s` only if mAP < 0.75 |
| Trust vision-only altitude for drops | Pitch during XY correction → circle appears as ellipse → `Z_est` underestimated → drop below 3.0 m | Connect MTF-01 or FC barometer |
| Allow drop when `|pitch| > 8°` | Ball lands ~49 cm off-centre at 3.5 m — outside F5 radius | Attitude gate is mandatory |
| Use synthetic frames for Hailo INT8 calibration | Synthetic pixels don't match IMX219 noise/vignetting — quantization degrades boundary features | Use `export.py --extract-calib-frames flight.mp4` |
| Retry without calling `mission.reset()` | Score, phase, PIDs, ball count, penalty counter all remain dirty | Always call `mission.reset()` before retry |
| Duplicate class-name mapping | When `dataset.py` `CLASS_NAMES` changes, other files silently break | Always derive: `{idx: name for idx, name in enumerate(CLASS_NAMES)}` |
| Set `drop_altitude_min < 3.0` in `config.yaml` | `load_mission_config_yaml()` safety assertion overrides to 3.05 m, but the operator sees a confusing log warning mid-flight | Never write a value below 3.05; the floor is non-negotiable |
| Call `load_mission_config_yaml()` after creating `MissionStateMachine()` | `FF_HISTORY_LEN`, `TOF_SPIKE_THRESHOLD_M`, `TOF_SPIKE_RESET_COUNT` are read at `__init__` — late loading leaves them at defaults | Always call `load_mission_config_yaml()` **before** `MissionStateMachine()` |

---

## Sensor Integration Reference

### MTF-01 (ToF laser + optical flow)
- Connected to ArduPilot serial port (`SERIAL_PROTOCOL=1`).
- ArduPilot rebroadcasts as `DISTANCE_SENSOR` and `OPTICAL_FLOW_RAD` MAVLink messages.
- `tof_altitude_m` field of `SensorData` — primary altitude source.
- `of_vx_ms` / `of_vy_ms` — used to isolate target velocity from drone motion in feedforward.

### BNO080 (attitude IMU)
- I2C address `0x4A` (default) or UART on Raspberry Pi 5.
- `pitch_deg`, `roll_deg` → attitude drop gate.
- `pitch_rate_dps`, `roll_rate_dps` → feedforward suppression guard.
- Library: Adafruit CircuitPython BNO08x (`adafruit_bno08x`).

### Passing sensors to the state machine
```python
from mission import MissionStateMachine, SensorData

sensors = SensorData(
    tof_altitude_m = tof_sensor.read_m(),
    of_vx_ms       = flow.vx_ms(),
    of_vy_ms       = flow.vy_ms(),
    pitch_deg      = imu.pitch_deg,
    roll_deg       = imu.roll_deg,
    pitch_rate_dps = imu.pitch_rate_dps,
    roll_rate_dps  = imu.roll_rate_dps,
)
frame = mission.update(frame, result, current_altitude=fc.altitude_m(), sensor_data=sensors)
```

All fields default to `None` — state machine degrades gracefully when a sensor is absent.

---

## Pre-Match Checklist

```
□ Camera calibrated: python camera_calibration.py calibrate → calibration.npz
□ HSV thresholds tuned on field: python hsv_calibrator.py → hsv_calib.json
□ Model validated: python evaluate.py → mAP50-95 ≥ 0.75
□ Hailo export done with real flight frames (not synthetic)
□ Test run with --calib calibration.npz --hsv-calib hsv_calib.json
□ HUD shows [TOF] or [FC] altitude source (NOT [VIS])
□ Confirm DROP_ALTITUDE_MIN = 3.05 in config.yaml
□ Confirm drop only triggers when HUD shows DROPPING (not ALIGN or DESCEND)
□ Telemetry logger running: --log run_001.csv
□ Finals mode: add --finals flag if competing in Finals round
```

---

## Evaluation Targets

| Metric | Target | File |
|---|---|---|
| Mask mAP50-95 | ≥ 0.75 | `evaluate.py` |
| Boundary F1 (2 px tolerance) | ≥ 0.85 | `evaluate.py` |
| Hausdorff distance | ≤ 3 px | `evaluate.py` |
| Predicted mask circularity | ≥ 0.90 | `evaluate.py` |
| End-to-end pipeline frequency | ≥ 20 Hz | `logger.py` summary |
| Peak inference latency | < 150 ms | `logger.py` summary |

---

## MAVLink Integration Pattern

```python
from mission import MissionStateMachine, Phase, SensorData

mission = MissionStateMachine(f_px=490.0, is_finals=False, takeoff_alt=3.3)

while True:
    frame, result = camera.grab()
    alt = mavlink.altitude_m()          # FC barometer — highest priority
    frame = mission.update(frame, result, current_altitude=alt, sensor_data=sensors)

    cmd = mission.last_command          # FlightCommand dataclass
    mavlink.set_velocity_body(vx=cmd.vx, vy=cmd.vy, vz=cmd.vz,
                              yaw_rate=cmd.yaw_rate)
    if cmd.drop_ball:
        servo.release()
    if mission.phase == Phase.COMPLETE:
        mavlink.disarm()

    # Retry (referee signals):
    # mission.reset(); mission.skip_takeoff()  ← if already at altitude
```

`FlightCommand` body-frame convention (NED): `vx > 0` forward, `vy > 0` right, `vz > 0` descend, `yaw_rate > 0` clockwise.

---

## Key File Quick Reference

| Task | Command |
|---|---|
| Generate training data | `python dataset.py --output data/synthetic --n-images 3000` |
| Train with boundary loss | `python train.py --config config.yaml --boundary-loss` |
| Camera calibration | `python camera_calibration.py calibrate --source 0 --rows 9 --cols 6 --square-mm 25.0 --output calibration.npz` |
| Evaluate model | `python evaluate.py --weights best.pt --data data/synthetic/circle_seg.yaml --config config.yaml` |
| Export to ONNX | `python export.py --weights best.pt --format onnx --validate --benchmark` |
| Extract Hailo calib frames | `python export.py --extract-calib-frames flight.mp4 --calib calibration.npz --calib-output hailo_calib/` |
| Tune HSV at venue | `python hsv_calibrator.py --source 0 --output hsv_calib.json` |
| Run on video | `python test_video.py --weights best.pt --source flight.mp4 --output result.mp4 --calib calibration.npz --hsv-calib hsv_calib.json --skip-takeoff --log run.csv` |
| Debug single frame | `python postprocess.py --weights best.pt --image frame.jpg --output annotated.jpg --calib calibration.npz --hsv-calib hsv_calib.json` |
