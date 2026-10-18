# autonomous_drone

Autonomous fire-fighting UAV built for the Vietnamese "UAV Cứu Hỏa Cứu Hộ 2026" competition:
a downward camera finds coloured fire circles, the drone drops five tennis balls on them and
lands on a helipad, all without a pilot. Target hardware: Raspberry Pi 5 + Hailo-8L, IMX219
camera, ArduPilot flight controller via MAVROS.

> **Status: being rebuilt.** The competition code is imported unchanged under `legacy/`.
> It is being refactored into tested packages with a closed-loop Gazebo + ArduPilot SITL
> simulation and on-device benchmarks. See the
> [design spec](docs/superpowers/specs/2026-10-18-uav-upgrade-design.md) (Vietnamese).

## Repository map

| Path | Contents |
|---|---|
| `legacy/` | Original competition code, byte-for-byte, with its snapshot history ([details](legacy/README.md)) |
| `assets/` | Binary assets (weights, datasets, videos), not in git; only `SHA256SUMS` is tracked |
| `scripts/fetch_assets.sh` | Downloads the assets from the GitHub Release and verifies their checksums |
| `scripts/publish_assets.sh` | Uploads `assets/` to a GitHub Release (maintainers) |
| `docs/superpowers/` | Design spec and implementation plans |

## Getting the assets

```bash
scripts/fetch_assets.sh                      # everything (~600 MB)
scripts/fetch_assets.sh dataset-real-286.zip # or just the files you need
```

## License

[Apache-2.0](LICENSE)
