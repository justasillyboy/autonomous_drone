"""Closed-loop simulation: Gazebo Harmonic + ArduCopter SITL + MAVROS + the real stack.

    ros2 launch uav_bringup sim.launch.py weights:=/models/model.onnx seed:=7 [round:=finals]

The scenario (world, drone model, SITL noise, camera calibration) is generated from the seed
into out_dir. The flight stack is the same as real.launch.py except that perception reads
the bridged Gazebo camera and every node runs on sim time. sim_operator plays the team (sets
GUIDED and arms); sim_referee scores from ground truth into results_dir.
mode:=direction_check replaces perception and mission with the ADR-0003 axis check.
Expects ARDUPILOT_HOME and ARDUPILOT_GAZEBO (set in docker/sim.Dockerfile).
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from uav_core.types import Round
from uav_sim.scenario import make_scenario
from uav_sim.world import WORLD_NAME, write_scenario

SITL_TCP = "tcp://127.0.0.1:5760"


def _setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    seed, round_ = int(get("seed")), Round(get("round"))
    random_layout = get("random_layout").lower() == "true"
    scenario = make_scenario(seed, round_, fixed_layout=not random_layout)
    out = Path(get("out_dir")) / scenario.name
    world = write_scenario(scenario, out)
    ap_home, ap_gz = os.environ["ARDUPILOT_HOME"], os.environ["ARDUPILOT_GAZEBO"]
    resource = os.pathsep.join(
        [
            str(out / "models"),
            f"{ap_gz}/models",
            f"{ap_gz}/worlds",
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ]
    )
    sim_time = {"use_sim_time": True}
    common = {"seed": seed, "round": round_.value, "random_layout": random_layout}

    gz = ["gz", "sim", "-r", "-v", "2", str(world)]
    if get("gui").lower() != "true":
        gz[2:2] = ["-s", "--headless-rendering"]
    defaults = ",".join(
        [
            f"{ap_home}/Tools/autotest/default_params/copter.parm",
            f"{ap_gz}/config/gazebo-iris-gimbal.parm",
            str(out / "sitl.parm"),
        ]
    )
    sitl = [
        f"{ap_home}/build/sitl/bin/arducopter", "-w", "--model", "JSON", "--speedup", "1",
        "--defaults", defaults, "-I0", "--home", "10.85,106.77,10,180",
    ]  # fmt: skip
    bridge_topics = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/downcam@sensor_msgs/msg/Image[gz.msgs.Image",
        "/downcam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        f"/world/{WORLD_NAME}/create@ros_gz_interfaces/srv/SpawnEntity",
        f"/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
    ]
    actions = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource),
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", f"{ap_gz}/build"),
        ExecuteProcess(cmd=gz, output="screen"),
        ExecuteProcess(cmd=sitl, cwd=str(out), output="screen"),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=bridge_topics,
            remappings=[
                ("/downcam", "/camera/image_raw"),
                ("/downcam/camera_info", "/camera/camera_info"),
            ],
            parameters=[sim_time],
        ),
        Node(
            package="mavros", executable="mavros_node", parameters=[{"fcu_url": SITL_TCP}, sim_time]
        ),
        Node(
            package="uav_hardware", executable="action_node", parameters=[sim_time], output="screen"
        ),
        Node(package="uav_sim", executable="ball_dropper", parameters=[sim_time]),
        Node(package="uav_sim", executable="sim_operator", parameters=[sim_time], output="screen"),
    ]
    if get("mode") == "direction_check":  # ADR-0003 gate: no perception, mission or referee
        actions.append(
            Node(
                package="uav_sim",
                executable="direction_check",
                parameters=[{"results_dir": get("results_dir")}, sim_time],
                output="screen",
            )
        )
        return actions
    actions += [
        Node(
            package="uav_perception",
            executable="perception_node",
            parameters=[
                {
                    "weights": get("weights"),
                    "input": "topic",
                    "calibration": str(out / "sim_calibration.npz"),
                    "hsv_calib": get("hsv_calib"),
                },
                sim_time,
            ],
            output="screen",
        ),
        Node(
            package="uav_mission",
            executable="mission_node",
            parameters=[{"finals": round_ is Round.FINALS}, sim_time],
            output="screen",
        ),
        Node(
            package="uav_sim",
            executable="sim_referee",
            parameters=[{**common, "results_dir": get("results_dir")}, sim_time],
            output="screen",
        ),
    ]
    if round_ is Round.FINALS:
        actions.append(
            Node(package="uav_sim", executable="target_mover", parameters=[common, sim_time])
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("weights", description=".onnx or .pt"),
            DeclareLaunchArgument("seed", default_value="0"),
            DeclareLaunchArgument("round", default_value="qualifier"),
            DeclareLaunchArgument("random_layout", default_value="false"),
            DeclareLaunchArgument(
                "hsv_calib", default_value="", description="default ranges in sim"
            ),
            DeclareLaunchArgument("out_dir", default_value="/tmp/uav_sim"),
            DeclareLaunchArgument("results_dir", default_value="results"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument(
                "mode", default_value="mission", description="or direction_check"
            ),
            OpaqueFunction(function=_setup),
        ]
    )
