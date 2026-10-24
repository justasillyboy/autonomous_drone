# Hardware bring-up checklist

Everything in this repo up to here runs without the drone. These are the steps that need
the hardware, in order. Do them with the **propellers removed** until step 6.

## 1. Pi 5 software

- Raspberry Pi OS Bookworm (64-bit), ROS 2 Jazzy in the `docker/pi.Dockerfile` container
  (M5), or natively for a first test.
- `pip install ./uav_core ./edge` next to ROS's Python 3.12, then `colcon build` in
  `ros2_ws` (the same steps as `scripts/ros_ci.sh`).
- MAVROS: `ros-jazzy-mavros` plus the GeographicLib datasets
  (`ros2 run mavros install_geographiclib_datasets.sh`).

## 2. Camera

```bash
uv run uav-calibrate-camera --video board.mp4 --cols 9 --rows 6 --square-mm 25
```

- Record the checkerboard at many positions and tilts, including the frame corners.
- RMS should be below 1 px. Note the printed `f_px` after undistortion.

## 3. Colours at the venue

```bash
uv run uav-hsv fit --dataset <labelled frames from the venue> --output hsv_calib.json
# or by hand
uv run --with opencv-python uav-hsv tune --source 0 --input config/hsv_calib.json
```

`config/hsv_calib.json` was fitted on the old recordings; lighting at the competition field
will differ.

## 4. IMU mounting (`bno080_node`)

Run `ros2 topic echo /imu/data` and check, with the drone in your hands:

| Motion | Expected in `/imu/data` (ROS FLU) |
|---|---|
| Nose up | pitch decreases (negative) |
| Right side down | roll increases (positive) |
| Turn clockwise seen from above | yaw decreases |

If not, set `mount_rpy_deg` in `ros2_ws/src/uav_bringup/config/real.yaml`.

## 4b. Rangefinder offset

With the drone standing on the floor, read `ros2 topic echo /mavros/rangefinder/rangefinder`
and set `estimation.tof_ground_offset_m` to that reading in the mission YAML. The rulebook's
3.0 m is measured at the drone's lowest point; without the offset a 3.05 m ToF reading can
mean skids below 3.0 m.

## 5. Drop servo

- Bench test with `require_guided: false`; publish one `FlightCommand` with
  `drop_ball: true` on `/mission/command`.
- Check that the servo pulses to `servo_pwm_drop`, returns to `servo_pwm_idle` after
  `servo_pulse_s` and releases exactly one ball. Set `require_guided` back to `true`.

## 6. Command directions (ADR-0003)

First in SITL (M2), then on the real drone in GUIDED at a safe height with a pilot ready to
take over:

| `FlightCommand` | Expected motion |
|---|---|
| `vz = +0.3` | descends |
| `vy = +0.3` | moves right |
| `yaw_rate_dps = +20` | turns clockwise seen from above |

The legacy bridge would have failed all three; this test is a release gate.

## 7. Full system

```bash
ros2 launch uav_bringup real.launch.py weights:=model.onnx \
    hsv_calib:=hsv_calib.json calibration:=calibration.npz
```

- Before arming, `/mission/status` must show `alt_source: FC` or `TOF`, never `VIS`.
- `/mission/e_stop` (hover) and `/mission/reset` (retry) must work from the ground station.
