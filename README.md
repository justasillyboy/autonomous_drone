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
| M1 Core: `uav_core` + ROS 2 packages | `uav_core` mission logic ported and tested; HUD, tools and ROS 2 packages to do |
| M2 Closed-loop simulation (Gazebo + ArduPilot SITL) | Not started |
| M3 ML experiments on real images | Not started |
| M4 Edge: INT8 model on the Hailo-8L, benchmarks | Not started |
| M5 Hardware-in-the-loop + Monte Carlo | Not started |
| M6 Report and demo | Not started |

`uav_core` reproduces the legacy `mission.py` command sequence bit-exactly on 10 scripted
scenarios (15,350 steps at 100 Hz), with 118 tests and 99 % line coverage.

## Repository map

| Path | Contents |
|---|---|
| `uav_core/` | Perception, estimation, control and mission logic in pure Python (no ROS), with golden characterisation tests against the legacy code |
| `legacy/` | Original competition code, byte-for-byte, with its snapshot history ([details](legacy/README.md)) |
| `assets/` | Binary assets (weights, datasets, videos), not in git; only `SHA256SUMS` is tracked |
| `scripts/fetch_assets.sh` | Downloads the assets from the GitHub Release and verifies their checksums |
| `scripts/publish_assets.sh` | Uploads `assets/` to a GitHub Release (maintainers) |
| `docs/config-reference.md` | Every mission config key with default, unit, range and rule source; generated from the schema, a test fails if it drifts |
| `docs/superpowers/` | Design spec and implementation plans |

## Development

Needs [uv](https://docs.astral.sh/uv/); it installs Python 3.12 itself.

```bash
uv sync                    # environment with uav_core and the dev tools
uv run pytest --cov        # golden + unit tests, fails below 80 % coverage
uv run lint-imports        # uav_core must not import ROS, torch or ultralytics
uv run pre-commit install  # ruff and file checks on every commit
```

## Getting the assets

```bash
scripts/fetch_assets.sh                      # everything (~600 MB)
scripts/fetch_assets.sh dataset-real-286.zip # or just the files you need
```

## License

[Apache-2.0](LICENSE)
