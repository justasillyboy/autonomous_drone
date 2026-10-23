"""action_node: /mission/command (body FRD) -> MAVROS velocity setpoints and the drop servo."""

from __future__ import annotations

import rclpy
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandLong
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from uav_core.messages import command_from_msg
from uav_interfaces.msg import FlightCommand

from uav_hardware.setpoint import ActionGate, fill_position_target, servo_command

RELIABLE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
TICK_S = 0.05


class ActionNode(Node):
    def __init__(self) -> None:
        super().__init__("action_node")
        self.channel = int(self.declare_parameter("servo_channel", 9).value)
        self.gate = ActionGate(
            require_guided=bool(self.declare_parameter("require_guided", True).value),
            drop_pwm=int(self.declare_parameter("servo_pwm_drop", 1900).value),
            idle_pwm=int(self.declare_parameter("servo_pwm_idle", 1100).value),
            pulse_s=float(self.declare_parameter("servo_pulse_s", 0.5).value),
        )
        self.pub = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)
        self.cmd_client = self.create_client(CommandLong, "/mavros/cmd/command")
        self.create_subscription(FlightCommand, "/mission/command", self.on_command, RELIABLE)
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.create_timer(TICK_S, self.on_tick)
        self.get_logger().info(f"action_node ready (servo channel {self.channel})")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg: State) -> None:
        if msg.mode != self.gate.mode:
            self.get_logger().info(f"FC mode {msg.mode}")
        self.gate.mode = msg.mode

    def on_command(self, msg: FlightCommand) -> None:
        cmd = command_from_msg(msg)
        setpoint, pwm = self.gate.on_command(cmd, self._now_s())
        if setpoint is None:
            self.get_logger().warn(
                f"FC mode {self.gate.mode!r} is not GUIDED: setpoint and drop suppressed",
                throttle_duration_sec=2.0,
            )
            return
        out = fill_position_target(PositionTarget(), setpoint)
        out.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(out)
        if pwm is not None:
            self.set_servo(pwm)

    def on_tick(self) -> None:
        pwm = self.gate.on_tick(self._now_s())
        if pwm is not None:
            self.set_servo(pwm)

    def set_servo(self, pwm: int) -> None:
        if not self.cmd_client.service_is_ready():
            self.get_logger().error("/mavros/cmd/command unavailable: servo not moved")
            return
        req = CommandLong.Request()
        for key, value in servo_command(self.channel, pwm).items():
            setattr(req, key, value)
        self.cmd_client.call_async(req).add_done_callback(
            lambda fut, pwm=pwm: self._servo_done(fut, pwm)
        )

    def _servo_done(self, future, pwm: int) -> None:
        try:
            ok = future.result().success
        except Exception as exc:  # the service call itself failed
            self.get_logger().error(f"servo {pwm} call failed: {exc}")
            return
        log = self.get_logger().info if ok else self.get_logger().error
        log(f"servo {pwm} {'accepted' if ok else 'rejected'} by the FC")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
