# Upgrade design: Autonomous Fire-Fighting UAV

- **Date:** 2026-10-18
- **Status:** Approved 2026-10-18
- **Repo:** `github.com/justasillyboy/autonomous_drone` (local clone: `~/dev/autonomous_drone`)
- **Legacy code:** imported into `legacy/` (binaries in the `legacy-assets-v0` release); the
  original `~/Documents/AI/UAV_Competition_NCKH/` folder was reduced to a README and private
  docs during M0

---

## 1. Goals

Turn the "UAV Cứu Hỏa Cứu Hộ 2026" competition entry into a personal, public portfolio project
that proves competence in **all three areas**: Robotics/Autonomy, Computer Vision/ML and Edge AI.
Each area gets **one claim backed by a real measurement**; no claim ships without evidence:

| Area | Target claim | Evidence |
|---|---|---|
| Robotics | The full autonomous mission (takeoff → 5 drops → landing) runs closed-loop in Gazebo + ArduPilot SITL with an X% success rate over N ≥ 50 randomised runs | Independent in-sim referee, per-run JSON results, demo video |
| CV / ML | A sim-to-real study and a loss ablation on a frozen real-image test set | mAP / Boundary F1 / Hausdorff tables with seeds and confidence intervals |
| Edge AI | An INT8 model on the Hailo-8L with an end-to-end loop ≥ 20 Hz and a per-stage latency table | p50/p95/p99 benchmark measured on the Pi 5, including HIL mode |

### Fixed constraints

- **Personal** project, **public** repo.
- Timeline **longer than 6 weeks** (8-week plan, section 13).
- Hardware: **Raspberry Pi 5 + Hailo-8L** and a Mac M1 Pro 16 GB. **No drone**, so flight is
  demonstrated with simulation and hardware-in-the-loop.
- **Approach A:** Gazebo Harmonic + ArduPilot SITL + ROS 2, with an HIL mode on the Pi 5.
- The organisers' rulebook PDF and its translation are **not** published. The README only
  summarises the rules.
- **Everything is written in English** — README, reports, ADRs, specs, plans, code comments,
  commit messages (owner decision 2026-10-18).

### Out of scope (YAGNI)

Real flight; PX4 support; RL-based control; multiple drones; web dashboard; an MLOps platform
(only configs in git + result CSVs; W&B is optional behind a flag).

---

## 2. Current state (surveyed 2026-10-18)

**Strengths to keep:**
- A complete CV pipeline: synthetic image generator (`dataset.py`), 286 labelled real images
  (Roboflow), YOLO11-seg with boundary/Dice loss, evaluation with Boundary F1 and Hausdorff,
  ONNX export.
- An 8-phase state machine: median-filtered ToF, IMU synchronised to the frame timestamp,
  feedforward for the moving F2 target, attitude gate on drop, dead-reckoning geofence.
- A five-node ROS 2 architecture: considered QoS, watchdog, e-stop, GUIDED-mode gate.

**Defects / technical debt to address:**

| # | Problem | Consequence |
|---|---|---|
| D1 | Never ran end to end: no `.hef`, `calibration.npz` or `hsv_calib.json` | No measurements at all |
| D2 | `uav_deploy` is an `ament_python` package but contains `msg/` | rosidl does not generate the messages, so `from uav_deploy.msg import …` fails. The fallback package `uav_interfaces` does not exist |
| D3 | `mission_control_node` re-draws fake masks (`_ShimResult`) to call `mission.update()` | Geometry is computed twice, the mission node has to receive images, the two nodes depend on each other backwards |
| D4 | `mission.py` is 3,095 lines: `_fire_phase` 547 lines, `update` 435 lines, every phase takes `frame` + the YOLO `result` | Perception is mixed with decision-making; nothing is unit-testable |
| D5 | Time is counted in frames (`fps_hint`) | Wrong when FPS fluctuates; broken when the sim runs slower than real time |
| D6 | Config is applied by overwriting module globals and must be loaded **before** construction | An ordering trap; `config.yaml`, README and code must be kept in sync by hand |
| D7 | **Suspected:** `action_mavros_node` pushes body-NED `vx, vy, vz, yaw_rate` straight into `/mavros/setpoint_velocity/cmd_vel_unstamped`. MAVROS expects ROS conventions (ENU/FLU), in the world frame by default | z/y/yaw signs may be flipped (a descend command climbs) and body/world frames confused. **Verify in SITL during M2** |
| D8 | `data.yaml` points at `valid/` and `test/`, but only `train/` exists | No independent real-image evaluation set |
| D9 | 3 copies of the code (whitespace-only differences) plus 1 older copy, no git; a 5.2 GB Linux/CUDA `.venv` inside iCloud-synced `~/Documents` | Clutter, wasted disk, hanging shell commands |
| D10 | *(found during the M0 inventory)* All 4 weight/ONNX files were **trained on synthetic images only**; the real dataset was exported on 2026-04-23, after the last training run | No model has ever learned from real images |
| D11 | *(found during the M0 inventory)* `best.onnx` (meant for deployment) is **YOLO11s**, while the `best.pt` next to it is YOLO11n | The ONNX is not an export of that `best.pt`, and it contradicts the legacy "only `n` on Hailo" rule |

**Source of truth for the legacy code:** `fixed_yolo_circle_seg/files/` (identical to the copies in
`deploy_files_…` and `UAV_node_ROS2/src/uav_deploy/main/` once whitespace is ignored), except
`deploy_guide.md`, taken from the newer copy in `deploy_files_drone_competition/`; plus the
`UAV_node_ROS2/src/uav_deploy/` package (the nodes, 2026-04-21). The `yolo_circle_seg/` copy
(2026-04-13) is superseded. The Google Drive folder "NodeRos2_TuHanh" is a copy of
`deploy_files_drone_competition` (checked file by file for the nodes).

---

## 3. Target repo layout

```
autonomous_drone/
├── README.md                 # intro, demo GIF, results table, architecture diagram
├── LICENSE                   # Apache-2.0 (matches the legacy package.xml)
├── pyproject.toml            # uv workspace: uav_core, ml, edge, tools
├── .pre-commit-config.yaml   # ruff, ruff-format, end-of-file, check-added-large-files (5 MB)
├── docs/
│   ├── architecture.md       # ROS graph, data flow, timing budget
│   ├── competition-rules.md  # rules summary, without the organisers' PDF
│   ├── config-reference.md   # GENERATED from the config schema (section 5); CI checks it stays in sync
│   ├── decisions/            # ADR-0001…, migrated from "Architecture Decisions" in the legacy CLAUDE.md
│   ├── report/               # technical report (section 12)
│   └── superpowers/{specs,plans}/
├── legacy/                   # legacy code, as-is; deleted step by step (section 4)
├── uav_core/                 # pure Python, NO ROS / cv2-GUI / torch imports
│   ├── src/uav_core/
│   │   ├── config.py         # pydantic schema: MissionConfig, RulebookConfig
│   │   ├── types.py          # Detection, SensorSnapshot, FlightCommand, MissionStatus
│   │   ├── frames.py         # body-FRD ↔ ROS-FLU / ENU conversions, in one place only
│   │   ├── perception/       # geometry.py, color.py, filters.py, altitude_vision.py
│   │   ├── estimation/       # altitude.py (priority ladder), tof_filter.py, imu_sync.py, target_motion.py
│   │   ├── control/          # pid.py, feedforward.py, drop_gate.py
│   │   ├── mission/          # phases/*.py, state_machine.py, scoring.py, geofence.py, timers.py
│   │   └── viz/hud.py        # HUD drawing (cv2), separate from decision-making
│   └── tests/
├── ml/
│   ├── src/uav_ml/           # synth/ (from dataset.py), train.py, loss.py, evaluate.py, export.py
│   ├── configs/              # one YAML file per experiment
│   ├── experiments/          # E1–E4 scripts + CSV results + plotting scripts
│   └── tests/
├── edge/
│   ├── hailo/                # Colab (x86) compile notebook, calib-frame extraction, alls/yaml
│   ├── runtime/              # backends: hailo.py, onnx.py, ultralytics.py (one shared interface)
│   └── benchmarks/           # measurement scripts + results
├── ros2_ws/src/
│   ├── uav_interfaces/       # ament_cmake: Detection.msg, Detections.msg, FlightCommand.msg, MissionStatus.msg
│   ├── uav_perception/       # perception_node (+ built-in v4l2 camera), camera_node, hud_node (both optional)
│   ├── uav_mission/          # mission_node: a thin shell around uav_core
│   ├── uav_hardware/         # bno080_node, action_node (MAVROS)
│   ├── uav_sim/              # sim_referee, ball_dropper, target_mover, scenario_runner
│   └── uav_bringup/          # real / sim / hil launch files + params
├── sim/
│   ├── worlds/competition_field.sdf.jinja   # template filled in by scenarios
│   ├── models/               # fire_circle_{red,yellow,blue}, tz_gate, helipad, ball, iris_downcam
│   └── scenarios/            # randomisation schema + seeds
├── tools/                    # camera_calibration.py, hsv_calibrator.py, bag_report.py (from logger.py)
├── docker/                   # sim.Dockerfile, pi.Dockerfile, compose.yaml, cyclonedds.xml
├── scripts/                  # fetch_assets.sh, publish_assets.sh, run_monte_carlo.sh
└── .github/workflows/        # ci.yml, sim-smoke.yml (manual / scheduled)
```

**Dependency rules** (enforced in CI with `import-linter`):
- `uav_core` depends on nothing in the repo.
- `ml`, `edge`, `tools` → `uav_core`.
- ROS packages → `uav_core` (+ `edge.runtime` for perception).
- Nothing may import from `legacy/`. `legacy/` is used only by characterisation tests.

---

## 4. Migrating the legacy code

1. **Import the legacy code as-is** *(done 2026-10-18)*. The old project had no git but 4
   snapshots (04-12, 04-13, 04-18/19, 04-21), so each snapshot became one `chore: import legacy …`
   commit, oldest first. Final state: `legacy/vision/` from `fixed_yolo_circle_seg/files/`,
   `legacy/ros2/uav_deploy/` from `src/uav_deploy` (without `build/ install/ log/ __pycache__`).
   Not a single byte changed (52/52 files compared), so the later refactor commits have clean
   diffs. The provenance of every snapshot is recorded in `legacy/README.md`.
2. **No binaries in git.** `.gitignore` blocks `*.pt *.onnx *.hef *.mp4 *.zip *.npz`, `data/`,
   `runs/`, `.venv/`, `docs/private/`. The assets are attached to the **GitHub Release
   `legacy-assets-v0`** (the owner confirmed everything, including the dataset, may be public);
   `scripts/fetch_assets.sh` downloads them and checks their sha256 against `assets/SHA256SUMS`
   (verified from a fresh clone: 18/18 OK). Asset names describe their actual content (full table
   in `legacy/README.md`):
   - weights: `yolo11s-seg_synth_2026-04-09.pt`, `yolo11n-seg_synth_2026-04-14.pt` and two
     YOLO11s ONNX exports (see D10, D11)
   - `dataset-real-286.zip` (286 labelled real images, class IDs remapped, no `__MACOSX`)
   - `real-frames-300.zip` (300 real frames: 286 of them are the labelled images above, only 14 are unlabelled)
   - `hailo-calib-frames.zip`, `debug-images.zip`
   - 10 distinct result videos: `output_{2,3,7,8}`, `final_flight_result{,_new}`, `ket_qua`,
     `result-v0`, `result_new-v0`, `result_new-v1`
3. **Local only:** the rulebook PDF, `Thể lệ_dịch.docx`, `node_feedback.docx`, `node_split_deploy.docx`
   → `docs/private/` (gitignored).
4. **Replace gradually:** every new `uav_core` or `ml` module replaces a part of `legacy/`. By the
   end of M1, `legacy/vision/mission.py` and `legacy/ros2/` are deleted; by the end of M3, the
   legacy ML scripts are deleted. The git history documents the upgrade.
5. **Clean up `~/Documents`** (M0; owner's choice on 2026-10-18): the repo is the single source of
   truth; the old folder keeps only `README.md` (what moved where) and `private/` (the 4 private
   documents, as an iCloud backup). Everything else goes to the **Trash** via Finder (never a
   permanent delete), **only after** the Release holds every asset and `fetch_assets.sh`
   downloads them with matching sha256. That includes:
   - the 5.5 GB `.venv`, the 2.9 GB `UAV_node_ROS2.zip` (the code snapshot inside it is in git)
   - duplicate zips/copies, `__MACOSX`, `build/install/log/__pycache__`

   This also frees disk space for the Docker images, since the drive has only ~30 GB free.

---

## 5. Config: a single source of truth

- `uav_core.config.MissionConfig`: a pydantic v2 model with `frozen=True`. Defaults are the
  current constants (`DROP_ALTITUDE_MIN=3.05`, `ALIGN_PIXEL_THRESH=25`, …).
- A separate, read-only `RulebookConfig`: qualifier/final scoring tables, field dimensions, circle
  diameters.
- Loaded from YAML **once at the boundary** (launch file or CLI), then **passed explicitly** to
  constructors. No global overrides, so the "load before construction" trap (D6) disappears.
- **Fail-fast validators:** `drop_altitude_min < 3.05` is a startup error, not a silent
  mid-match correction like the legacy REQ-NF-04. Reason: a wrong config must be caught on the
  ground.
- `docs/config-reference.md` is generated from the schema (name, default, unit, description, rule
  source). CI fails if the file drifts from the schema. This replaces the legacy CLAUDE.md manual
  rule "update both config.yaml and README".

---

## 6. `uav_core`: splitting `mission.py`

### Data contracts (`types.py`, all `frozen` dataclasses)

```python
Detection(cls: TargetClass, color: Color | None, cx: float, cy: float, radius: float,
          bbox: tuple[float, float, float, float], fill_ratio: float, conf: float,
          edge_clipped: bool, z_vision_m: float | None)
SensorSnapshot(stamp_s: float, fc_alt_m, tof_alt_m, vel_body_ms: tuple | None,
               rpy_deg: tuple | None, rates_dps: tuple | None)   # every field may be None
FlightCommand(vx, vy, vz, yaw_rate_dps, yaw_lock, drop_ball, phase, action)  # body FRD
MissionStatus(phase, score, penalties, balls_dropped, targets_done, alt_source, elapsed_s)
```

### Core API

```python
sm = MissionStateMachine(cfg: MissionConfig, rules: RulebookConfig, round: Round)
out = sm.step(t_s: float, detections: Sequence[Detection], sensors: SensorSnapshot,
              frame_size: tuple[int, int]) -> StepOutput(command, status)
sm.reset(); sm.skip_takeoff(); sm.signal_landed(); sm.start_match(t_s)
```

- `step()` **takes no image** and draws nothing. The HUD moves to `uav_core.viz.hud`, used by the
  (optional) `hud_node` or by offline tools.
- **Time is in seconds**, from the input timestamps (D5). `fps_hint` is gone. Every window
  (5 s altitude penalty, 2 s drop cooldown, recovery, 1.5 s tilt relaxation) is a timer on `t_s`.
  Works correctly with sim time too.
- Each phase is its own module (`phases/takeoff.py`, `search.py`, `targeting.py`, `drop.py`,
  `recovery.py`, `helipad.py`, `landing.py`): pure functions of the form
  `(ctx, inputs) -> (command, transition | None)`.
- Mutable internal state lives only inside `MissionStateMachine`; inputs and outputs are always
  immutable.
- Geometry (`_extract_geometry`, fill ratio, H_landing overlap suppression, HSV colour) moves to
  `uav_core.perception`. **Only perception_node** calls these functions, so nothing is computed
  twice (D3).

### Preserve behaviour: characterisation tests **before** refactoring

1. Write a harness that runs `legacy/vision/mission.py` on **scripted scenarios**: frames rendered
   with the `dataset.py` synthetic generator plus simulated sensor sequences (takeoff, alignment,
   drop, target loss, ToF spikes, moving F2, timeout). Record the `FlightCommand` sequence as a
   golden file.
2. The new code must reproduce the golden (small float tolerance), **except** for deliberate
   changes recorded in an ADR: time in seconds, fail-fast config. For those changes the golden is
   updated with the reason in the commit.
3. Unit tests for every module. `uav_core` coverage ≥ 80%. `hypothesis` for `frames.py`
   (round-trip conversions return the original value) and for geometry (synthetic circular masks
   of many sizes, including edge clipping).

---

## 7. ROS 2

- **Versions:** ROS 2 **Jazzy** + Gazebo **Harmonic** — the officially supported pairing, with
  stable binary `ros_gz`, `ardupilot_gazebo` and MAVROS.
- **`uav_interfaces` (ament_cmake)** fixes D2: `Detection.msg`, `Detections.msg` (`std_msgs/Header`
  carrying the frame capture stamp), `FlightCommand.msg`, `MissionStatus.msg`. Services use
  `std_srvs/Trigger`.
- **`perception_node`:**
  - Image source `input: topic | v4l2`. In `v4l2` mode it reads the camera on the Pi directly, to
    avoid copying images over DDS.
  - Undistortion with precomputed maps (`initUndistortRectifyMap`).
  - Backend `hailo | onnx | ultralytics` via `edge.runtime`.
  - Calls `uav_core.perception`, then publishes `/perception/detections` (Reliable, depth 1) and
    `/perception/debug_image` (rate-limited, only when someone subscribes).
  - Takes `f_px` from `CameraInfo`.
- **`mission_node`:**
  - Subscribes to `/perception/detections`, the IMU and MAVROS (rel_alt, rangefinder,
    velocity_body, state, extended_state).
  - Feeds sensors into `imu_sync` by stamp, calls `sm.step()`, publishes `/mission/command`
    (Reliable) and `/mission/status`.
  - **Does not subscribe to images.** Keeps the 0.5 s watchdog, `/mission/reset`, `/mission/e_stop`.
  - The match clock starts on the ARM rising edge.
  - `use_sim_time` works because time comes from stamps.
- **`action_node`:**
  - Converts `FlightCommand` (body FRD) into MAVROS commands with **a single function in
    `uav_core.frames`**. Sends via `/mavros/setpoint_raw/local` (`PositionTarget`, body frame,
    type_mask using velocity + yaw rate only), so the frame is declared explicitly instead of
    depending on the `mav_frame` parameter.
  - Keeps the GUIDED gate. The drop servo uses `MAV_CMD_DO_SET_SERVO`.
  - Handles D7: write tests for the frame conversion, then confirm in SITL. If SITL proves the
    legacy code had the wrong signs, the report records it as a real bug caught by simulation.
- **`bno080_node`:** as before, 50 Hz by default (per the old feedback about I2C load).
- **Launch:** `real.launch.py`, `sim.launch.py`, `hil.launch.py` — the same nodes, different params.
- **Tests:** `launch_testing` for fake detections → mission_node → command, plus frame-conversion
  tests for action_node. `colcon test` runs in CI.

---

## 8. Simulation (Gazebo Harmonic + ArduPilot SITL)

- **Environment:** `docker/sim.Dockerfile` (Ubuntu 24.04, multi-arch arm64/amd64): ROS 2 Jazzy,
  Gazebo Harmonic, ArduCopter SITL (pinned to a stable Copter-4.x tag at bootstrap),
  `ardupilot_gazebo`, MAVROS. `compose.yaml` has the services `sim`, `stack` and `novnc`
  (optional, for recording video). Runs in lockstep: slow rendering only lowers the real-time
  factor (RTF); the control loop stays correct.
- **World `competition_field.sdf.jinja`:**
  - 24 × 12 m floor, 12 × 3.5 m Safe Zone with the H helipad.
  - 5 target circles per the rules: red 1.0 m ×2, yellow 0.6 m ×2, blue 0.4 m ×1, on 0.1 m plinths.
  - TZ gate: 1.6 m inner ring, centre 1.7 m high, with a contact sensor.
  - Lighting and materials are parameters.
- **Drone:** `iris_downcam` = the `ardupilot_gazebo` iris plus a 640 × 480 downward camera whose
  FOV and distortion approximate the wide-angle IMX219, so the undistortion pipeline is exercised
  in sim too.
- **Nodes in `uav_sim`:**
  - `target_mover`: moves F2 at 0.1 m/s in finals mode, through the `set_pose` service.
  - `ball_dropper`: watches `/mavros/rc/out` on the servo channel; when the PWM crosses the
    threshold it spawns a physical ball at the drone's position. `action_node` therefore behaves
    exactly as in real flight.
  - `sim_referee`: scores **independently** of the mission's self-reported score, from Gazebo
    ground-truth poses:
    - takeoff ≥ 3.0 m
    - ball lands inside the circle radius
    - penalty for < 3.0 m altitude longer than 5 s
    - out of bounds longer than 5 s
    - TZ collisions
    - landing inside the helipad
    - 300 s limit

    Writes `results/<run_id>.json`.
  - `scenario_runner`: from a seed, generates non-overlapping target positions, lighting
    (direction/intensity), floor tint, wind (`SIM_WIND_SPD/DIR`), sensor noise (SITL params),
    qualifier or final round.
- **Monte Carlo:** `scripts/run_monte_carlo.sh` runs N headless runs. The report gives the success
  rate (all 5 drops and a landing within 300 s), the score distribution (max 200), a failure
  taxonomy (target lost, missed drop, altitude penalty, timeout, …) and referee vs. self-reported
  scores.

---

## 9. Hardware-in-the-loop (Pi 5 + Hailo-8L)

- **Mac:** Gazebo + SITL. **Pi 5:** MAVROS + perception (Hailo backend) + mission + action. This
  split is identical to real flight, where the Pi talks MAVLink to the FC. MAVROS on the Pi
  connects to SITL over UDP.
- **Images from Gazebo to the Pi:** compressed (JPEG) `image_transport` over a **direct Ethernet
  cable**, with Cyclone DDS configured for unicast peers. The extra latency of image transport is
  measured **separately** so it is not mixed into perception latency.
- **Pi:** Raspberry Pi OS Bookworm as the host (officially supported by Hailo) plus
  `docker/pi.Dockerfile` (`ros:jazzy` + HailoRT), with `--device /dev/hailo0` passed into the
  container. **The HailoRT version in the container must exactly match the PCIe driver on the
  host**; pin both.

---

## 10. CV / ML

### Data and evaluation protocol (fixes D8)

- **Three data domains:**
  1. procedural synthetic (`uav_ml.synth`, from `dataset.py`)
  2. **Gazebo renders, auto-labelled** from known poses (added to `uav_sim`)
  3. real images
- **Real images:** 286 labelled images exist today; of the 300 frames in `real-frames-300.zip`
  only **14** are unlabelled (M0 inventory). A meaningful increase in real data requires
  **recording new video**; the current model then proposes labels that are corrected by hand on
  Roboflow.
- **Split train/val/test by recording session / source video**, not randomly by frame, so that
  near-identical consecutive frames never land in both train and test. Note: the existing 300
  frames are numbered consecutively `real_frame_0000–0299`, probably from one or very few videos;
  if so, split into **contiguous time blocks**, not by frame. The test set is **frozen**: the file
  list and sha256 values are committed in `ml/configs/splits/`.
- **Metrics:** mask mAP50-95, Boundary F1 (2 px), Hausdorff, circularity, colour classification
  accuracy, all with **bootstrap 95% CIs**, since the real test set is small.

### Experiments

| ID | Question | Design |
|---|---|---|
| E1 | How many real labels are needed? (sim-to-real) | Train on {synth, gazebo, synth+gazebo} × {0, 25, 50, 100 %} of the real images, plus pretrain-synth → fine-tune-real. Plot against label count |
| E2 | Does the custom boundary/Dice loss help? | Baseline / +boundary / +Dice / +both, 3 seeds each, report mean ± std. Negative results are reported too |
| E3 | Which model gives the best trade-off on the Hailo-8L? | YOLO11n-seg, YOLO11s-seg, YOLOv8n-seg (officially supported by the Hailo Model Zoo), and Ultralytics' newest seg model at the time if it compiles. Plot the accuracy–latency Pareto front |
| E4 | Test the legacy "Danger Zones" | INT8 calibrated on real vs. synthetic frames; `mask_ratio` 4 vs 2 (FPS and PCIe bandwidth). Turn the legacy CLAUDE.md assumptions into measurements |

- **Compute:** smoke tests on the Mac's MPS; multi-seed training runs on Colab/Kaggle GPUs. Every
  run is reproducible from `ml/configs/*.yaml` + seed; results are CSV and figures are generated by
  scripts.

---

## 11. Edge (Hailo-8L)

- **Compile:** ONNX → Hailo DFC (parse → optimise with the real calibration set → compile `.hef`)
  in the notebook `edge/hailo/compile.ipynb`, run on **Colab x86**; DFC does not run on the M1 Mac.
  **The owner must download the DFC wheel** from the Hailo Developer Zone (account required).
  - Fallback: if YOLO11-seg hits an unsupported op, use YOLOv8n-seg (already part of E3).
- **Runtime:** `edge.runtime.hailo` uses the HailoRT async `InferModel`. Mask post-processing
  (proto × coeffs, mask_ratio 4) runs in numpy on the CPU. All three backends share one
  interface: `infer(frame) -> list[RawInstance]`.
- **Benchmarks on the Pi 5:**
  - Per-stage latency p50/p95/p99: capture, pre-processing, NPU, post-processing, geometry,
    `sm.step`, publish.
  - End-to-end FPS; CPU %.
  - Temperature and throttling over 10 minutes of continuous running.
  - Hailo INT8 vs. ONNX on the Pi 5's own CPU.
  - Power: optional, if a USB-C meter is available.
- **Check the legacy numbers:** the "~2 ms inference" in the legacy README was never measured;
  replace it with a measurement.

---

## 12. Documentation and presentation

- **README:**
  - a one-paragraph introduction
  - demo GIF (sim + HIL)
  - **results table** with the numbers from section 1
  - architecture diagram (mermaid)
  - `docker compose up sim` quickstart
  - repo map
  - an honest **Limitations** section (no real flight; small real test set)
- **Technical report** `docs/report/` (Markdown → PDF, about 8–10 pages): problem, system, E1–E4,
  closed-loop and HIL results, limitations. Can later become a blog post or an arXiv preprint.
- **ADRs:** migrate the 10 "Architecture Decisions" of the legacy CLAUDE.md into `docs/decisions/`,
  plus the new decisions (time in seconds, fail-fast config, frame handling in `frames.py`,
  Jazzy/Harmonic, MAVROS).
- Rewrite the repo's `CLAUDE.md` for the new structure.
- Demo video recorded via noVNC/Gazebo GUI, published on YouTube and linked.

---

## 13. Milestones (8 weeks, one person, overlapping)

This spec is too large for one plan. Each milestone gets **its own plan** in
`docs/superpowers/plans/`, written when that milestone starts. The first plan combines M0 + M1,
because every later milestone depends on `uav_core` and `uav_interfaces`.

| Milestone | Weeks | Deliverables | Done when |
|---|---|---|---|
| M0 Bootstrap | 1 | legacy import, `.gitignore`, Release assets + `fetch_assets.sh`, pre-commit, CI (ruff + pytest), `~/Documents` cleanup (after approval) | CI green on `main`, downloaded assets match sha256 |
| M1 Core | 1–3 | characterisation golden → `uav_core` + config + frames + tests; `uav_interfaces` + new nodes + launch_testing; delete `legacy/vision/mission.py` and `legacy/ros2` | golden matches; coverage ≥ 80%; `colcon build && colcon test` green |
| M2 Sim | 2–4 | Docker image, world, iris_downcam, ball_dropper, referee, first closed-loop run; resolve D7 | one qualifier run goes from takeoff to landing in sim, with referee JSON |
| M3 ML | 3–5 | labelling + splits, auto-labelled Gazebo images, E1–E3 | frozen test set; E1–E3 tables with CIs |
| M4 Edge | 4–6 | `.hef`, Hailo backend, benchmarks, E4 | p50/p95/p99 latency table on the Pi 5; ≥ 20 Hz or a documented reason why not |
| M5 HIL + MC | 6–7 | `hil.launch.py`, Ethernet + DDS, Monte Carlo N ≥ 50 (sim) and N ≥ 10 (HIL) | success rate + failure taxonomy in `results/` |
| M6 Presentation | 7–8 | README, report, video, ADRs, delete the remaining legacy parts | README with real numbers; PDF report; public video |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Gazebo camera rendering in Docker on the M1 is too slow (no GPU) | Lockstep stays correct; lower camera resolution/rate; fall back to running MC on a cloud x86 VM (shared with the Hailo DFC) |
| Hailo DFC needs x86 + a Hailo account | Colab notebook; the owner downloads the wheel |
| YOLO11-seg uses an op the Hailo does not support | YOLOv8n-seg as the fallback (already in E3) |
| HailoRT in the container mismatches the host driver | Pin both versions; a check script at container start |
| Small real test set (a few hundred images) | Bootstrap CIs; split by video; stated in Limitations |
| SITL/Gazebo/MAVROS versions change | Pin in the Dockerfile; tag images by date |
| ~30 GB free on the Mac, sim image about 8–10 GB | Clean `~/Documents` in M0; periodic `docker system prune` |
| Refactoring breaks existing behaviour | Characterisation golden tests first (section 6) |

---

## 15. Project-wide done criteria

1. CI green on `main`: ruff, import-linter, pytest (`uav_core` coverage ≥ 80%), `colcon test`, and
   the check that `config-reference.md` matches the schema.
2. Sim: N ≥ 50 randomised runs, reporting the success rate and mean score **as judged by the
   independent referee**.
3. HIL: at least 10 runs with perception on the Hailo-8L; closed-loop latency measured and reported.
4. ML: frozen real test set; E1–E4 results with seeds/CIs, negative results included.
5. README with a demo GIF and real numbers; a PDF technical report; no unmeasured claims left.
