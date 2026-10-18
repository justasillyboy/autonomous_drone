"""uav.launch.py — 5-node deploy launch, per node_split_deploy.md §4c.

Launch order (MAVROS is launched separately by the operator):
    camera_node → sensor_bridge_node → perception_node
                                     → mission_control_node
                                     → action_mavros_node

Required CLI arguments (with sensible defaults pulled from the installed
package share dir):
    weights:=<path to best.pt>
    calib_npz:=<path to calibration.npz>
    hsv_calib:=<path to hsv_calib.json>
    config_yaml:=<path to config.yaml>
    takeoff_alt:=<m>                (default 3.3)
    is_finals:=<true|false>         (default false)
    fps_hint:=<Hz>                  (default 20.0)
    skip_takeoff:=<true|false>      (default false — set true when testing indoors
                                     with the drone already at altitude)
    video_device:=/dev/video0
    camera_fps:=30.0
    require_guided:=true            (suppress setpoints/drop if FC not in GUIDED)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('uav_deploy')
    default_config = os.path.join(pkg_share, 'config', 'config.yaml')
    default_calib = os.path.join(pkg_share, 'config', 'calibration.npz')
    default_hsv = os.path.join(pkg_share, 'config', 'hsv_calib.json')

    # ── Launch arguments ────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('weights', default_value='best.pt',
                              description='YOLO weights path (.pt or .hef)'),
        DeclareLaunchArgument('calib_npz', default_value=default_calib,
                              description='Camera calibration.npz path'),
        DeclareLaunchArgument('hsv_calib', default_value=default_hsv,
                              description='HSV calibration JSON path'),
        DeclareLaunchArgument('config_yaml', default_value=default_config,
                              description='Mission constants YAML (overrides mission.py defaults)'),
        DeclareLaunchArgument('takeoff_alt', default_value='3.3',
                              description='Takeoff target altitude [m] (≥ 3.05)'),
        DeclareLaunchArgument('is_finals', default_value='false',
                              description='Enable Finals scoring + F2 feedforward'),
        DeclareLaunchArgument('fps_hint', default_value='20.0',
                              description='Expected mission loop rate [Hz]'),
        DeclareLaunchArgument('skip_takeoff', default_value='false',
                              description='Skip takeoff phase (drone already airborne)'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_fps', default_value='30.0'),
        DeclareLaunchArgument('conf_threshold', default_value='0.5'),
        DeclareLaunchArgument('imu_rate_hz', default_value='100.0'),
        DeclareLaunchArgument('i2c_address', default_value='74',  # 0x4A = 74
                              description='BNO080 I2C address (decimal)'),
        DeclareLaunchArgument('servo_channel', default_value='9',
                              description='MAV AUX channel for drop servo'),
        DeclareLaunchArgument('servo_pwm_drop', default_value='1900'),
        DeclareLaunchArgument('servo_pwm_idle', default_value='1100'),
        DeclareLaunchArgument('require_guided', default_value='true',
                              description='Suppress setpoints/drop if FC not in GUIDED'),
    ]

    weights = LaunchConfiguration('weights')
    calib_npz = LaunchConfiguration('calib_npz')
    hsv_calib = LaunchConfiguration('hsv_calib')
    config_yaml = LaunchConfiguration('config_yaml')
    takeoff_alt = LaunchConfiguration('takeoff_alt')
    is_finals = LaunchConfiguration('is_finals')
    fps_hint = LaunchConfiguration('fps_hint')
    skip_takeoff = LaunchConfiguration('skip_takeoff')
    video_device = LaunchConfiguration('video_device')
    camera_fps = LaunchConfiguration('camera_fps')
    conf_threshold = LaunchConfiguration('conf_threshold')
    imu_rate_hz = LaunchConfiguration('imu_rate_hz')
    i2c_address = LaunchConfiguration('i2c_address')
    servo_channel = LaunchConfiguration('servo_channel')
    servo_pwm_drop = LaunchConfiguration('servo_pwm_drop')
    servo_pwm_idle = LaunchConfiguration('servo_pwm_idle')
    require_guided = LaunchConfiguration('require_guided')

    # ── Nodes ───────────────────────────────────────────────────────────────
    # Node 1 — Camera (publishes /camera/image_raw + /camera/camera_info).
    camera_node = Node(
        package='uav_deploy',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'calib_path': calib_npz,
            'video_device': video_device,
            'fps': camera_fps,
            'lock_awb_aec': True,
            'frame_id': 'camera',
        }],
    )

    # Node 3 — Sensor bridge (BNO080 IMU @ /bno080/imu).
    sensor_bridge_node = Node(
        package='uav_deploy',
        executable='sensor_node',
        name='sensor_bridge_node',
        output='screen',
        parameters=[{
            'rate_hz': imu_rate_hz,
            'i2c_address': i2c_address,
            'frame_id': 'bno080',
        }],
    )

    # Node 2 — Perception (YOLO → /perception/targets).
    perception_node = Node(
        package='uav_deploy',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{
            'weights': weights,
            'conf_threshold': conf_threshold,
            'hsv_calib': hsv_calib,
            'debug_image_rate_hz': 5.0,
        }],
    )

    # Node 4 — Mission control (state machine; drives /mission/flight_command).
    mission_control_node = Node(
        package='uav_deploy',
        executable='mission_node',
        name='mission_control_node',
        output='screen',
        parameters=[{
            'takeoff_alt': takeoff_alt,
            'is_finals': is_finals,
            'config_yaml': config_yaml,
            'fps_hint': fps_hint,
            'skip_takeoff': skip_takeoff,
            'watchdog_timeout_s': 0.5,
        }],
    )

    # Node 5 — Action/MAVROS bridge (velocity + drop servo).
    action_mavros_node = Node(
        package='uav_deploy',
        executable='action_node',
        name='action_mavros_node',
        output='screen',
        parameters=[{
            'servo_channel': servo_channel,
            'servo_pwm_drop': servo_pwm_drop,
            'servo_pwm_idle': servo_pwm_idle,
            'require_guided': require_guided,
        }],
    )

    return LaunchDescription(args + [
        camera_node,
        sensor_bridge_node,
        perception_node,
        mission_control_node,
        action_mavros_node,
    ])
