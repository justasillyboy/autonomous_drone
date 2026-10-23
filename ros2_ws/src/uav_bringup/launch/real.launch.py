"""Real drone: MAVROS + perception (camera read directly) + mission + action + IMU.

    ros2 launch uav_bringup real.launch.py weights:=/opt/uav/model.onnx \
        hsv_calib:=/opt/uav/hsv_calib.json calibration:=/opt/uav/calibration.npz [finals:=true]
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arg = LaunchConfiguration
    params = PathJoinSubstitution([FindPackageShare("uav_bringup"), "config", "real.yaml"])
    args = [
        DeclareLaunchArgument("weights", description=".onnx, .pt (later .hef)"),
        DeclareLaunchArgument("hsv_calib", default_value=""),
        DeclareLaunchArgument("calibration", default_value=""),
        DeclareLaunchArgument("config_yaml", default_value="", description="mission overrides"),
        DeclareLaunchArgument("finals", default_value="false"),
        DeclareLaunchArgument("skip_takeoff", default_value="false"),
        DeclareLaunchArgument("fcu_url", default_value="/dev/ttyAMA0:921600"),
    ]
    flag = lambda name: ParameterValue(arg(name), value_type=bool)  # noqa: E731
    nodes = [
        Node(
            package="mavros",
            executable="mavros_node",
            parameters=[{"fcu_url": arg("fcu_url")}],
            output="screen",
        ),
        Node(
            package="uav_perception",
            executable="perception_node",
            parameters=[
                params,
                {
                    "weights": arg("weights"),
                    "hsv_calib": arg("hsv_calib"),
                    "calibration": arg("calibration"),
                    "config_yaml": arg("config_yaml"),
                },
            ],
            output="screen",
        ),
        Node(
            package="uav_mission",
            executable="mission_node",
            parameters=[
                params,
                {
                    "config_yaml": arg("config_yaml"),
                    "finals": flag("finals"),
                    "skip_takeoff": flag("skip_takeoff"),
                },
            ],
            output="screen",
        ),
        Node(
            package="uav_hardware", executable="action_node", parameters=[params], output="screen"
        ),
        Node(
            package="uav_hardware", executable="bno080_node", parameters=[params], output="screen"
        ),
    ]
    return LaunchDescription(args + nodes)
