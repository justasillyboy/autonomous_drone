# Flight stack for the Raspberry Pi 5 (arm64): ROS 2 Jazzy, MAVROS, uav_core / uav_edge and
# the ROS workspace, plus HailoRT when its wheel is provided. The host runs Raspberry Pi OS
# Bookworm with the Hailo PCIe driver; the HailoRT wheel in wheels/ must match its version.
#   docker build -f docker/pi.Dockerfile -t uav-pi .
#   docker run --rm -it --network host --device /dev/hailo0 --device /dev/video0 \
#       --device /dev/i2c-1 --device /dev/ttyAMA0 uav-pi \
#       ros2 launch uav_bringup real.launch.py weights:=/models/best.hef
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ros-jazzy-mavros ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-image-transport ros-jazzy-compressed-image-transport geographiclib-tools \
    && geographiclib-get-geoids egm96-5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY uav_core uav_core
COPY edge edge
COPY ros2_ws ros2_ws
COPY config config
COPY docker/cyclonedds.xml /etc/cyclonedds.xml
# Optional: HailoRT wheel(s) from the Hailo Developer Zone, matching the host driver.
COPY docker/wheels/ /tmp/wheels/
RUN pip3 install --break-system-packages -c ros2_ws/constraints.txt ./uav_core ./edge \
        adafruit-circuitpython-bno08x \
    && (ls /tmp/wheels/*.whl >/dev/null 2>&1 && pip3 install --break-system-packages /tmp/wheels/*.whl || true) \
    && . /opt/ros/jazzy/setup.sh \
    && colcon --log-base /tmp/colcon-log build --base-paths ros2_ws/src \
        --build-base /opt/ws/build --install-base /opt/ws/install

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
RUN printf '%s\n' '#!/bin/bash' 'set -e' \
        'source /opt/ros/jazzy/setup.bash' 'source /opt/ws/install/setup.bash' 'exec "$@"' \
        > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
