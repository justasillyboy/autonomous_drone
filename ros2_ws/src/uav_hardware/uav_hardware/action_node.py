"""action_node: /mission/command (body FRD) -> MAVROS velocity setpoints and the drop servo."""

from __future__ import annotations

import rclpy
from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import CommandLong, CommandTOL, SetMode
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from uav_core.messages import command_from_msg
from uav_interfaces.msg import FlightCommand

from uav_hardware.setpoint import ActionGate, fill_position_target, servo_command

MAV_LANDED_STATE_ON_GROUND = 1

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
            takeoff_alt_m=float(self.declare_parameter("takeoff_alt_m", 3.3).value),
        )
        self.pub = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)
        self.cmd_client = self.create_client(CommandLong, "/mavros/cmd/command")
        self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.create_subscription(FlightCommand, "/mission/command", self.on_command, RELIABLE)
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.create_subscription(ExtendedState, "/mavros/extended_state", self.on_extended, 10)
        self.create_timer(TICK_S, self.on_tick)
        self.get_logger().info(f"action_node ready (servo channel {self.channel})")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg: State) -> None:
        if msg.mode != self.gate.mode:
            self.get_logger().info(f"FC mode {msg.mode}")
        self.gate.mode = msg.mode
        if self.gate.armed and not msg.armed:
            self.gate.on_disarmed()
        self.gate.armed = msg.armed

    def on_extended(self, msg: ExtendedState) -> None:
        self.gate.on_ground = msg.landed_state == MAV_LANDED_STATE_ON_GROUND

    def on_command(self, msg: FlightCommand) -> None:
        cmd = command_from_msg(msg)
        act = self.gate.on_command(cmd, self._now_s())
        if act.setpoint is None:
            self.get_logger().warn(
                f"FC mode {self.gate.mode!r} is not GUIDED: setpoint and drop suppressed",
                throttle_duration_sec=2.0,
            )
            return
        if act.takeoff_alt_m is not None:
            self.call(
                self.takeoff_client, CommandTOL.Request(altitude=act.takeoff_alt_m), "takeoff"
            )
        out = fill_position_target(PositionTarget(), act.setpoint)
        out.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(out)
        if act.servo_pwm is not None:
            self.set_servo(act.servo_pwm)
        if act.set_mode is not None:
            self.call(self.mode_client, SetMode.Request(custom_mode=act.set_mode), act.set_mode)

    def on_tick(self) -> None:
        pwm = self.gate.on_tick(self._now_s())
        if pwm is not None:
            self.set_servo(pwm)

    def set_servo(self, pwm: int) -> None:
        req = CommandLong.Request()
        for key, value in servo_command(self.channel, pwm).items():
            setattr(req, key, value)
        self.call(self.cmd_client, req, f"servo {pwm}")

    def call(self, client, request, what: str) -> None:
        if not client.service_is_ready():
            self.get_logger().error(f"{client.srv_name} unavailable: {what} not sent")
            return
        client.call_async(request).add_done_callback(lambda fut: self._done(fut, what))

    def _done(self, future, what: str) -> None:
        try:
            res = future.result()
        except Exception as exc:  # the service call itself failed
            self.get_logger().error(f"{what} call failed: {exc}")
            return
        ok = getattr(res, "success", getattr(res, "mode_sent", False))
        log = self.get_logger().info if ok else self.get_logger().error
        log(f"{what} {'accepted' if ok else 'rejected'} by the FC")


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
