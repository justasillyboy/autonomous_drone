# autonomous_drone

[![CI](https://github.com/justasillyboy/autonomous_drone/actions/workflows/ci.yml/badge.svg)](https://github.com/justasillyboy/autonomous_drone/actions/workflows/ci.yml)

Autonomous fire-fighting UAV built for the Vietnamese "UAV Cứu Hỏa Cứu Hộ 2026" competition:
a downward camera finds coloured fire circles, the drone drops five tennis balls on them and
lands on a helipad, all without a pilot. Target hardware: Raspberry Pi 5 + Hailo-8L, IMX219
camera, ArduPilot flight controller via MAVROS.

> **Status: being rebuilt.** The competition code is imported unchanged under `legacy/`.
> It is being refactored into tested packages with a closed-loop Gazebo + ArduPilot SITL
> simulation and on-device benchmarks. See the
> [design spec](docs/superpowers/specs/2026-10-18-uav-upgrade-design.md).

## Progress

| Milestone | State |
|---|---|
| M0 Bootstrap: legacy import, asset release, CI | Done |
| M1 Core: `uav_core` + ROS 2 packages | Done in code: `uav_core`, inference backends, tools, ROS 2 nodes with `colcon test` in CI. Needs hardware to go further |
| M2 Closed-loop simulation (Gazebo + ArduPilot SITL) | Field, independent referee, ball dropper, moving F2, direction check, Docker image and CI smoke test written; first runs in CI |
| M3 ML experiments on real images | Frozen time-block split of the real frames, metrics with bootstrap CIs, legacy baseline measured; training to do (needs a GPU) |
| M4 Edge: INT8 model on the Hailo-8L, benchmarks | Not started |
| M5 Hardware-in-the-loop + Monte Carlo | Not started |
| M6 Report and demo | Not started |

`uav_core` reproduces the legacy `mission.py` command sequence bit-exactly on 10 scripted
scenarios (15,350 steps at 100 Hz). Python CI: about 210 tests, 98 % line coverage; ROS CI:
`colcon build && colcon test` in `ros:jazzy-ros-base`, including a `launch_testing` run of
`mission_node`.

Measured so far, without hardware:

- **Colour thresholds:** on 85 held-out real frames, the legacy HSV ranges give 41.8 % of the
  fire circles the right colour; ranges fitted on the other 201 frames give 95.1 %
  ([details](config/README.md)).
- **Legacy models on real images** (frozen test split, 54 frames, 169 instances, 95 %
  bootstrap CIs): the better YOLO11s export reaches mask AP50 0.37 [0.29, 0.45] and AP50-95
  0.23 [0.19, 0.27]; circles are found (AP50 0.58) but helipads mostly are not (0.15). Both
  were trained on synthetic data only, so M3 retrains on real labels
  ([details](ml/experiments/baseline/README.md)).
- **Bugs found by porting** (none was ever observed, because the legacy stack never flew end
  to end): velocity commands sent to MAVROS with the wrong frame (descend would climb),
  mirrored body velocity and yaw from the IMU, masks scaled with letterbox padding, the
  vision altitude computed with the pre-undistortion focal length, and a drop servo that
  never returned to idle. See [ADR-0003](docs/decisions/0003-frame-conventions.md) and the
  commit history.

## Repository map

| Path | Contents |
|---|---|
| `uav_core/` | Perception, estimation, control, mission logic and HUD in pure Python (no ROS), with golden characterisation tests against the legacy code |
| `edge/` | Inference backends behind one interface: ONNX Runtime, Ultralytics (Hailo in M4) |
| `tools/` | `uav-calibrate-camera`, `uav-hsv` (fit or tune colour ranges), `uav-run-video` (replay a flight with the HUD) |
| `ros2_ws/src/` | ROS 2 Jazzy packages: `uav_interfaces`, `uav_perception`, `uav_mission`, `uav_hardware`, `uav_sim`, `uav_bringup` |
| `docker/` | Simulation image (Gazebo Harmonic + ArduCopter SITL + the stack) and compose file |
| `ml/` | Frozen data split, evaluation metrics with bootstrap CIs, baseline results |
| `config/` | Field configuration, e.g. HSV ranges fitted on the real frames |
| `legacy/` | Original competition code, byte-for-byte, with its snapshot history ([details](legacy/README.md)) |
| `assets/` | Binary assets (weights, datasets, videos), not in git; only `SHA256SUMS` is tracked |
| `scripts/fetch_assets.sh` | Downloads the assets from the GitHub Release and verifies their checksums |
| `scripts/publish_assets.sh` | Uploads `assets/` to a GitHub Release (maintainers) |
| `docs/config-reference.md` | Every mission config key with default, unit, range and rule source; generated from the schema, a test fails if it drifts |
| `docs/decisions/` | Architecture decision records |
| `docs/architecture.md` | Layers, runtime graph, data flow, timing budget |
| `docs/competition-rules.md` | The rules the autonomous round depends on |
| `docs/simulation.md` | How the closed-loop simulation and its referee work |
| `docs/hardware-checklist.md` | The steps that need the drone, in order |
| `docs/superpowers/` | Design spec and implementation plans |

## Development

Needs [uv](https://docs.astral.sh/uv/); it installs Python 3.12 itself.

```bash
uv sync                    # environment with uav_core and the dev tools
uv run pytest --cov        # golden + unit tests, fails below 80 % coverage
uv run lint-imports        # uav_core must not import ROS, torch or ultralytics
uv run pre-commit install  # ruff and file checks on every commit

# ROS 2 packages, in the same container as CI
docker run --rm -v "$PWD":/repo -w /repo ros:jazzy-ros-base scripts/ros_ci.sh

# replay a recorded flight through perception + mission
uv run uav-run-video --weights model.onnx --source flight.mp4 --output hud.mp4 \
    --hsv-calib config/hsv_calib.json
```

## Getting the assets

```bash
scripts/fetch_assets.sh                      # everything (~600 MB)
scripts/fetch_assets.sh dataset-real-286.zip # or just the files you need
```

## License

[Apache-2.0](LICENSE)
