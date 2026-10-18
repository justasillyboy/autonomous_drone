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

# 6. Run on pre-recorded video (--skip-takeoff since drone already airborne)
python test_video.py --weights best.pt \
       --source flight.mp4 --output result.mp4 \
       --calib calibration.npz --log run_001.csv \
       --skip-takeoff

# 7. Run on live camera (real competition — TAKEOFF phase active)
python test_video.py --weights best.pt \
       --source 0 --output result.mp4 \
       --calib calibration.npz --log run_001.csv \
       --takeoff-alt 3.3 --finals

# 8. Annotate a single frame
python postprocess.py --weights best.pt --image frame.jpg --output annotated.jpg
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
| `DROPPING` | `3.05 m ≤ Z ≤ 4.5 m` AND offset `< 25 px` | `drop_ball=True` |
| `RECOVERY` | Target lost 10 frames | Yaw sweep `+30°/s → −60°/s → +30°/s → 0°/s` |
| `FIND_HELIPAD` | All 5 balls dropped | Hover + yaw scan if helipad lost |
| `LANDING` | H_landing `d_ratio ≥ 0.70` | `vz=0.4` descend, proportional XY |
| `COMPLETE` | `signal_landed()` called | All zeros — awards landing bonus |

### Reading `FlightCommand` in your MAVLink loop

```python
from mission import MissionStateMachine, Phase

mission = MissionStateMachine(f_px=490.0, is_finals=False, takeoff_alt=3.3)

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

## Altitude-based drop control

Vision `Z_est = f_px × D_real / d_px` is the fallback when no sensor is connected.
**When a rangefinder/barometer is connected, it is used as the authoritative altitude source.**
This is critical: drone pitch during XY corrections projects the circle as an ellipse,
inflating `d_px` and underestimating `Z_est`, potentially triggering a drop below 3.0 m.

The HUD shows `[SNS]` (sensor active) or `[VIS]` (vision fallback) so you can verify
which source is being used during testing.

| Circle | D_real | Safe drop window |
|---|---|---|
| Red F1/F2 | 1.0 m | 3.05 – 4.50 m |
| Yellow F3/F4 | 0.6 m | 3.05 – 4.50 m |
| Blue F5 | 0.4 m | 3.05 – 4.50 m |

---

## False-positive suppression

| Filter | Constant | Rejects |
|---|---|---|
| Fill-ratio | `FILL_RATIO_MIN_CIRCLE = 0.40` | Hollow arcs, H_landing ring |
| Size cap | `MAX_FIRE_DIAMETER_RATIO = 0.55` | Court arc boundary lines |
| Cross-class NMS | `H_LANDING_OVERLAP_THRESH = 0.45` | Circles inside H_landing bbox |

---

## Temporal smoothing & feedforward (Finals F2)

- `smoothing_alpha=0.4` for Qualifier (static targets)
- Auto-lowered to `0.15` when `--finals` — reduces phase lag on 10 cm/s F2
- Feedforward velocity from 8-frame position history compensates for target motion
- **Feedforward is suppressed when `|cmd_vx| > 0.35 m/s` or `|cmd_vy| > 0.35 m/s`**
  (drone is actively pitching — camera tilt would corrupt pixel history)

---

## Recovery behaviour

Fire target lost → `RECOVERY` phase:
1. Yaw sweep: `+30°/s` × 20 frames → `−60°/s` × 20 frames → `+30°/s` → `0°/s`
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
`PX_TO_MS_FIRE = 0.005` was chosen so 100 px error → 0.5 m/s, which the FC tracks cleanly
with P=0.135 in the inner loop. See `mission.py` constants block for derivation.
