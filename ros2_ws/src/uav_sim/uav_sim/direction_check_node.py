"""direction_check: drives /mission/command through the ADR-0003 axis checks (instead of
mission_node) and writes results/direction_check.json. Launch with sim.launch.py
mode:=direction_check."""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from uav_core.messages import fill_command_msg
from uav_interfaces.msg import FlightCommand

from uav_sim.direction_check import DirectionCheck, enu_yaw_deg
from uav_sim.world import DRONE_MODEL, WORLD_NAME

RELIABLE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


class DirectionCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("direction_check")
        self.results_dir = Path(self.declare_parameter("results_dir", "results").value)
        self.check = DirectionCheck(float(self.declare_parameter("step_s", 3.0).value))
        self.pub = self.create_publisher(FlightCommand, "/mission/command", RELIABLE)
        self.create_subscription(
            TFMessage, f"/world/{WORLD_NAME}/dynamic_pose/info", self.on_poses, 10
        )
        self.pose: tuple | None = None
        self.written = False
        self.create_timer(0.1, self.on_timer)

    def on_poses(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            if tf.child_frame_id == DRONE_MODEL:
                p, q = tf.transform.translation, tf.transform.rotation
                self.pose = (p.x, p.y, p.z, enu_yaw_deg(q.x, q.y, q.z, q.w))

    def on_timer(self) -> None:
        if self.pose is None:
            return
        now = self.get_clock().now()
        cmd = self.check.update(now.nanoseconds * 1e-9, self.pose)
        out = fill_command_msg(FlightCommand(), cmd)
        out.header.stamp = now.to_msg()
        self.pub.publish(out)
        if self.check.done and not self.written:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            result = {"passed": self.check.passed, "steps": self.check.results}
            (self.results_dir / "direction_check.json").write_text(json.dumps(result, indent=2))
            self.written = True
            log = self.get_logger().info if self.check.passed else self.get_logger().error
            log(f"direction check {'PASSED' if self.check.passed else 'FAILED'}: {result}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DirectionCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
