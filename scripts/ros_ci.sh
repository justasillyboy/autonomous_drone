#!/usr/bin/env bash
# Build and test ros2_ws inside a ros:jazzy-ros-base container (CI and local use the same).
#   docker run --rm -v "$PWD":/repo -w /repo ros:jazzy-ros-base scripts/ros_ci.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3-pip python3-pytest ros-jazzy-mavros-msgs ros-jazzy-launch-testing-ros >/dev/null

# uav_core and uav_edge are plain Python packages; install them next to ROS's Python 3.12,
# keeping Debian's numpy 1.26 that the ROS extensions are compiled against.
pip3 install -q --break-system-packages -c ros2_ws/constraints.txt ./uav_core ./edge

# Build outside the source tree so the mounted repo stays clean.
WS=/tmp/ros_ws
mkdir -p "$WS"
cp -r ros2_ws/src "$WS/"
cd "$WS"
set +u
source /opt/ros/jazzy/setup.bash
set -u
colcon build --event-handlers console_direct-
colcon test --event-handlers console_direct- --return-code-on-test-failure
colcon test-result --verbose
