# ADR-0009: ROS 2 Jazzy, Gazebo Harmonic and MAVROS with explicit body-frame setpoints

- **Status:** Accepted
- **Date:** 2026-10-24
- **Code:** `ros2_ws/src/*`, `scripts/ros_ci.sh`, `.github/workflows/ci.yml`

## Context

The legacy nodes targeted an unspecified ROS 2 release, kept the messages inside an
`ament_python` package (which cannot generate them, defect D2) and published velocities on
`/mavros/setpoint_velocity/cmd_vel_unstamped`, whose frame depends on a plugin parameter.

## Decision

- **ROS 2 Jazzy + Gazebo Harmonic** on Ubuntu 24.04: the officially paired LTS releases, with
  binary `ros_gz`, `ardupilot_gazebo` and MAVROS packages. Python 3.12 matches `uav_core`.
- **MAVROS**, not a hand-written MAVLink bridge: GUIDED-mode handling, servo commands and
  the extended landed state come for free and are what ArduPilot documents.
- Messages in a separate **`uav_interfaces`** (`ament_cmake`) package.
- Velocities go to **`/mavros/setpoint_raw/local`** as `PositionTarget` with
  `FRAME_BODY_NED` declared in each message (ADR-0003).
- Each node is a thin shell; its logic lives in a ROS-free module (`bridge.py`,
  `setpoint.py`, `pipeline.py`) tested in the main Python CI. A second CI job builds the
  workspace in `ros:jazzy-ros-base` and runs `colcon test`, including a `launch_testing`
  test that drives `mission_node` through real topics.

## Consequences

- ROS's Python extensions are compiled against Debian's numpy 1.26, so `uav_core` and
  `uav_edge` are installed with `ros2_ws/constraints.txt` (`numpy<2`), which also selects
  OpenCV and ONNX Runtime builds that support it. The uv environment on the Mac is free to
  use numpy 2.
- No `cv_bridge`: images are converted with numpy in `uav_perception.pipeline`.
- The real-time behaviour on the Pi (DDS settings, image transport) is measured in M5, not
  assumed.
