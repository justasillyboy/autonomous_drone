"""ball_dropper: when the FC drives the drop servo high, spawn a tennis ball under the drone.

It watches /mavros/rc/out, so action_node behaves exactly as in real flight (it only sends
MAV_CMD_DO_SET_SERVO); the simulated ball then falls, bounces and rolls under physics.
"""

from __future__ import annotations

import rclpy
from mavros_msgs.msg import RCOut
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity
from tf2_msgs.msg import TFMessage

from uav_sim.balls import DropTrigger, ball_sdf
from uav_sim.world import DRONE_MODEL, WORLD_NAME

RELEASE_BELOW_M = 0.15  # spawn point under the drone's origin, clear of the frame


class BallDropperNode(Node):
    def __init__(self) -> None:
        super().__init__("ball_dropper")
        self.channel = int(self.declare_parameter("servo_channel", 9).value)
        self.max_balls = int(self.declare_parameter("max_balls", 5).value)
        self.trigger = DropTrigger(int(self.declare_parameter("threshold_pwm", 1500).value))
        self.drone_xyz: tuple[float, float, float] | None = None
        self.count = 0
        self.spawn = self.create_client(SpawnEntity, f"/world/{WORLD_NAME}/create")
        self.create_subscription(RCOut, "/mavros/rc/out", self.on_rc, 10)
        self.create_subscription(
            TFMessage, f"/world/{WORLD_NAME}/dynamic_pose/info", self.on_poses, 10
        )

    def on_poses(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            if tf.child_frame_id == DRONE_MODEL:
                p = tf.transform.translation
                self.drone_xyz = (p.x, p.y, p.z)

    def on_rc(self, msg: RCOut) -> None:
        idx = self.channel - 1
        if idx >= len(msg.channels) or not self.trigger.update(msg.channels[idx]):
            return
        if self.drone_xyz is None or self.count >= self.max_balls:
            self.get_logger().warn("drop ignored (no drone pose yet, or no balls left)")
            return
        x, y, z = self.drone_xyz
        req = SpawnEntity.Request()
        req.entity_factory.name = f"ball_{self.count}"
        req.entity_factory.sdf = ball_sdf(req.entity_factory.name)
        req.entity_factory.pose.position.x = x
        req.entity_factory.pose.position.y = y
        req.entity_factory.pose.position.z = z - RELEASE_BELOW_M
        self.count += 1
        self.spawn.call_async(req)
        self.get_logger().info(f"ball {self.count} released at ({x:.2f}, {y:.2f}, {z:.2f})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallDropperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
