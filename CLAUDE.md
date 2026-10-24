# CLAUDE.md

Autonomous fire-fighting UAV (Pi 5 + Hailo-8L, ArduCopter via MAVROS, ROS 2 Jazzy). The
README has the overview; `docs/architecture.md` the structure; `docs/decisions/` the ADRs.

## Commands

```bash
uv sync                                    # Python 3.12 env with every workspace package
uv run pytest --cov                        # all Python tests, coverage >= 80 %
uv run ruff check . && uv run ruff format --check .
uv run lint-imports                        # package dependency rules
docker run --rm -v "$PWD":/repo -w /repo ros:jazzy-ros-base scripts/ros_ci.sh  # colcon
uv run python -m uav_core.config_doc       # after any change to uav_core/config.py
```

## Layout and rules

| Package | Role | May import |
|---|---|---|
| `uav_core` | perception, estimation, control, mission, HUD | nothing in the repo; no ROS, torch, ultralytics, onnxruntime |
| `edge` (`uav_edge`) | ONNX / Ultralytics / Hailo backends, benchmark, `.hef` compiler | `uav_core` |
| `ml` (`uav_ml`) | frozen split, metrics, training | `uav_core`, `uav_edge` |
| `tools` (`uav_tools`) | calibration, HSV fit, video replay | all of the above |
| `ros2_ws/src/*` | ROS 2 nodes; logic in ROS-free modules tested by the main suite | `uav_core`, `uav_edge` |

- **Conventions change only in `uav_core.frames`** (ADR-0003): commands are body FRD
  (+vz down, +yaw clockwise) inside the stack; ROS/MAVROS sides are FLU/ENU.
- **Config is fail-fast and immutable** (ADR-0002): add a field to `MissionConfig` with unit,
  description and rule source, then regenerate `docs/config-reference.md` (a test fails
  otherwise). Never read module globals for tuning.
- **Timers are seconds, not frames** (ADR-0001); the mission clock is the frame's capture
  stamp.
- **Goldens** (`uav_core/tests/characterisation/golden/`) were recorded from the legacy code.
  A deliberate behaviour change re-records them with `record_golden.py` and says why in the
  commit; otherwise they must not change.
- ROS nodes stay thin: put logic in `bridge.py` / `setpoint.py` / `pipeline.py` /
  `referee.py`-style modules that read messages by attribute, and test those without ROS.
- In ROS containers install `uav_core` / `uav_edge` with `-c ros2_ws/constraints.txt`
  (numpy < 2: ROS's extensions are built against Debian's numpy).
- `legacy/` is a byte-for-byte record; never edit or import it.

## Writing

- Everything committed is in English: code, comments, docs, commit messages.
- Commits: conventional (`feat:`, `fix:`, `docs:`, ...), body explains why.
- Do not claim numbers that were not measured; put measurements next to the command that
  produced them (`ml/experiments/`, `edge/benchmarks/`, `config/README.md`).
- Binary assets live in the GitHub Release; `scripts/fetch_assets.sh` downloads them into
  `assets/` (gitignored).
