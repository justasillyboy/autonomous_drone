# Closed-loop simulation image: ROS 2 Jazzy + Gazebo Harmonic + ArduCopter SITL + MAVROS +
# this repo's stack. Multi-arch (amd64, arm64). About 7-9 GB.
#   docker compose -f docker/compose.yaml run --rm sim
FROM ros:jazzy-ros-base

ARG ARDUPILOT_TAG=Copter-4.6.3
ARG ARDUPILOT_GAZEBO_REF=082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5
ENV DEBIAN_FRONTEND=noninteractive \
    GZ_VERSION=harmonic \
    ARDUPILOT_HOME=/opt/ardupilot \
    ARDUPILOT_GAZEBO=/opt/ardupilot_gazebo

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates ccache \
        python3-pip python3-pytest python3-dev \
        ros-jazzy-ros-gz ros-jazzy-mavros ros-jazzy-mavros-extras ros-jazzy-launch-testing-ros \
        libgz-sim8-dev rapidjson-dev geographiclib-tools \
        libegl1 libgl1-mesa-dri mesa-utils \
    && geographiclib-get-geoids egm96-5 \
    && rm -rf /var/lib/apt/lists/*

# ArduCopter SITL, built for the host architecture
RUN git clone --depth 1 --branch ${ARDUPILOT_TAG} --recurse-submodules --shallow-submodules \
        https://github.com/ArduPilot/ardupilot.git ${ARDUPILOT_HOME} \
    && pip3 install --break-system-packages empy==3.3.4 future pexpect pymavlink \
    && cd ${ARDUPILOT_HOME} && ./waf configure --board sitl && ./waf copter \
    && rm -rf ${ARDUPILOT_HOME}/build/sitl/ArduCopter ${ARDUPILOT_HOME}/build/sitl/libraries

# Gazebo <-> ArduPilot JSON plugin and the iris models
RUN git clone https://github.com/ArduPilot/ardupilot_gazebo.git ${ARDUPILOT_GAZEBO} \
    && cd ${ARDUPILOT_GAZEBO} && git checkout ${ARDUPILOT_GAZEBO_REF} \
    && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && make -j"$(nproc)"

# This repo: Python packages next to ROS's Python (numpy 1.x, see ros2_ws/constraints.txt),
# then the ROS workspace.
WORKDIR /repo
COPY uav_core uav_core
COPY edge edge
COPY ros2_ws ros2_ws
COPY config config
RUN pip3 install --break-system-packages -c ros2_ws/constraints.txt ./uav_core ./edge \
    && . /opt/ros/jazzy/setup.sh \
    && colcon --log-base /tmp/colcon-log build --base-paths ros2_ws/src \
        --build-base /opt/ws/build --install-base /opt/ws/install

RUN printf '%s\n' '#!/bin/bash' 'set -e' \
        'source /opt/ros/jazzy/setup.bash' 'source /opt/ws/install/setup.bash' 'exec "$@"' \
        > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
