"""sim_operator: plays the team at the start signal. Once MAVROS is connected and the EKF has
settled (`settle_s`), it sets GUIDED and arms, retrying every 2 s until armed. Arming starts
the match clock in mission_node and the referee; mission_node's TAKEOFF phase does the rest.
"""

from __future__ import annotations

import rclpy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node

RETRY_S = 2.0


class OperatorNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_operator")
        self.settle_s = float(self.declare_parameter("settle_s", 20.0).value)
        self.state: State | None = None
        self.connected_at: float | None = None
        self.done = False
        self.mode = self.create_client(SetMode, "/mavros/set_mode")
        self.arm = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.create_timer(RETRY_S, self.on_timer)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg: State) -> None:
        if msg.connected and self.connected_at is None:
            self.connected_at = self._now()
        self.state = msg

    def on_timer(self) -> None:
        s = self.state
        if self.done or s is None or self.connected_at is None:
            return
        if self._now() - self.connected_at < self.settle_s:
            return
        if s.armed:
            self.done = True
            self.get_logger().info("armed in GUIDED: match on")
        elif s.mode != "GUIDED" and self.mode.service_is_ready():
            self.mode.call_async(SetMode.Request(custom_mode="GUIDED"))
        elif s.mode == "GUIDED" and self.arm.service_is_ready():
            self.arm.call_async(CommandBool.Request(value=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OperatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
