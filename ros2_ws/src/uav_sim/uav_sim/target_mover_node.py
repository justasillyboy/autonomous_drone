"""target_mover (finals): moves F2 back and forth at 0.1 m/s and publishes where it is."""

from __future__ import annotations

import contextlib

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from uav_core.types import Round

from uav_sim.field import f2_position
from uav_sim.scenario import make_scenario
from uav_sim.world import WORLD_NAME

PERIOD_S = 0.05


class TargetMoverNode(Node):
    def __init__(self) -> None:
        super().__init__("target_mover")
        seed = int(self.declare_parameter("seed", 0).value)
        scenario = make_scenario(seed, Round.FINALS)
        self.f2 = scenario.layout.target("F2")
        self.t0: float | None = None
        self.client = self.create_client(SetEntityPose, f"/world/{WORLD_NAME}/set_pose")
        self.pub = self.create_publisher(PointStamped, "/sim/f2", 10)
        self.create_timer(PERIOD_S, self.on_timer)

    def on_timer(self) -> None:
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        self.t0 = t if self.t0 is None else self.t0
        x, y = f2_position(self.f2.x, self.f2.y, t - self.t0)
        if self.client.service_is_ready():
            req = SetEntityPose.Request()
            req.entity.name = "target_F2"
            req.pose.position.x, req.pose.position.y = x, y
            req.pose.orientation.w = 1.0
            self.client.call_async(req)
        msg = PointStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y = x, y
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetMoverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):  # a second Ctrl-C during teardown
            node.destroy_node()
        rclpy.try_shutdown()
