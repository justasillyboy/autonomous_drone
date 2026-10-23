"""Launch mission_node and drive it through ROS topics (runs under colcon test only)."""

import time
import unittest

import pytest

launch_testing = pytest.importorskip("launch_testing")

import launch  # noqa: E402
import launch_ros.actions  # noqa: E402
import rclpy  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402
from uav_interfaces.msg import Detection, Detections, FlightCommand  # noqa: E402

RELIABLE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(
        package="uav_mission",
        executable="mission_node",
        parameters=[{"skip_takeoff": True, "watchdog_timeout_s": 0.5}],
    )
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()])


def _centred_red(node):
    msg = Detections()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.width, msg.height = 640, 480
    msg.detections = [
        Detection(
            cls="circle", color="red", cx=320, cy=240, radius=60,
            bbox=[260, 180, 380, 300], fill_ratio=0.95, conf=0.9, edge_clipped=False,
        )
    ]  # fmt: skip
    return msg


class TestMissionNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("mission_probe")
        self.commands = []
        self.node.create_subscription(
            FlightCommand, "/mission/command", self.commands.append, RELIABLE
        )
        self.pub_det = self.node.create_publisher(Detections, "/perception/detections", RELIABLE)
        self.pub_alt = self.node.create_publisher(
            Float64, "/mavros/global_position/rel_alt", qos_profile_sensor_data
        )

    def tearDown(self):
        self.node.destroy_node()

    def _spin_until(self, done, timeout_s, feed=True):
        end = time.time() + timeout_s
        while time.time() < end and not done():
            if feed:
                self.pub_alt.publish(Float64(data=3.4))
                self.pub_det.publish(_centred_red(self.node))
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return done()

    def test_centred_target_at_drop_altitude_releases_a_ball_then_watchdog_hovers(self):
        assert self._spin_until(lambda: any(c.drop_ball for c in self.commands), 15.0)
        n = len(self.commands)
        hovered = self._spin_until(
            lambda: any(c.action == "WATCHDOG" for c in self.commands[n:]), 3.0, feed=False
        )
        assert hovered, [c.action for c in self.commands[n:]]
        last = self.commands[-1]
        assert (last.vx, last.vy, last.vz, last.drop_ball) == (0.0, 0.0, 0.0, False)
