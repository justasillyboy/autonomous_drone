from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='uav_deploy',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),

        Node(
            package='uav_deploy',
            executable='perception_node',
            name='perception_node',
            output='screen'
        ),

        Node(
            package='uav_deploy',
            executable='mission_node',  # Updated executable name
            name='mission_control_node',
            output='screen'
        ),

        Node(
            package='uav_deploy',
            executable='sensor_node',
            name='sensor_node',
            output='screen'
        ),

        Node(
            package='uav_deploy',
            executable='action_node',
            name='action_node',
            output='screen'
        ),

    ])