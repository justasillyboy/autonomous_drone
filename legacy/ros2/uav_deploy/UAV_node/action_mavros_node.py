"""action_mavros_node — hardware effector layer, per node_split_deploy.md §2 Node 5.

Translates /mission/flight_command (FlightCommandMsg) into MAVROS primitives:
  • geometry_msgs/Twist → /mavros/setpoint_velocity/cmd_vel_unstamped
    (yaw_rate is converted deg/s → rad/s, per MAVROS convention)
  • mavros_msgs/srv/CommandLong (MAV_CMD_DO_SET_SERVO) → drop servo on
    drop_ball=True

Safety:
  • GUIDED mode gate — velocity setpoints are suppressed when FC is not in
    GUIDED (STABILIZE/ALT_HOLD ignore setpoints but the servo would still
    fire, risking a ball drop outside the target).
  • Relays LAND_COMPLETE via /action/land_complete so mission_control_node
    can advance to Phase.COMPLETE.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

try:
    from mavros_msgs.msg import ExtendedState, State as MavrosState
    from mavros_msgs.srv import CommandLong
except ImportError:  # pragma: no cover
    MavrosState = None
    ExtendedState = None
    CommandLong = None

try:
    from uav_deploy.msg import FlightCommandMsg
except ImportError:  # pragma: no cover
    from uav_interfaces.msg import FlightCommandMsg


class ActionMavrosNode(Node):
    def __init__(self):
        super().__init__('action_mavros_node')

        self.declare_parameter('servo_channel', 9)   # AUX9 by default
        self.declare_parameter('servo_pwm_drop', 1900)
        self.declare_parameter('servo_pwm_idle', 1100)
        self.declare_parameter('require_guided', True)

        self._servo_channel = int(self.get_parameter('servo_channel').value)
        self._servo_pwm_drop = int(self.get_parameter('servo_pwm_drop').value)
        self._servo_pwm_idle = int(self.get_parameter('servo_pwm_idle').value)
        self._require_guided = bool(self.get_parameter('require_guided').value)

        qos_rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )
        qos_best = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )

        self.pub_cmd_vel = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', qos_best)
        self.pub_land_complete = self.create_publisher(
            Bool, '/action/land_complete', qos_rel)

        self.create_subscription(
            FlightCommandMsg, '/mission/flight_command',
            self.flight_command_callback, qos_rel)
        if MavrosState is not None:
            self.create_subscription(
                MavrosState, '/mavros/state', self.mavros_state_cb, qos_rel)
            self.create_subscription(
                ExtendedState, '/mavros/extended_state',
                self.extended_state_cb, qos_rel)
        else:
            self.get_logger().warn(
                "mavros_msgs missing — GUIDED gate DISABLED and LAND_COMPLETE "
                "relay inactive")

        if CommandLong is not None:
            self.command_client = self.create_client(
                CommandLong, '/mavros/cmd/command')
        else:
            self.command_client = None

        self._mode = None
        self._armed = False
        self._last_land_state = None
        self._last_drop_pub_s = 0.0

        self.get_logger().info(
            f"action_mavros_node started (servo_channel={self._servo_channel}, "
            f"drop_pwm={self._servo_pwm_drop}, require_guided={self._require_guided})")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def mavros_state_cb(self, msg):
        self._mode = msg.mode
        self._armed = bool(msg.armed)

    def extended_state_cb(self, msg):
        # MAV_LANDED_STATE_ON_GROUND == 1.  Relay one edge-triggered Bool.
        on_ground = (msg.landed_state == 1)
        if on_ground and self._last_land_state is False:
            self.pub_land_complete.publish(Bool(data=True))
            self.get_logger().info("LAND_COMPLETE relayed → /action/land_complete")
        self._last_land_state = on_ground

    def flight_command_callback(self, msg: FlightCommandMsg):
        # GUIDED mode gate — if FC is not accepting velocity setpoints, do not
        # send them; critically, ALSO suppress the drop so we never release a
        # ball outside the target window.
        if self._require_guided and self._mode is not None and self._mode != "GUIDED":
            self.get_logger().warn(
                f"FC mode '{self._mode}' (expected GUIDED) — suppressing setpoint"
                f" and drop_ball",
                throttle_duration_sec=2.0)
            return

        twist = Twist()
        twist.linear.x = float(msg.vx)
        twist.linear.y = float(msg.vy)
        twist.linear.z = float(msg.vz)
        # FlightCommandMsg.yaw_rate is deg/s (clockwise positive).  MAVROS
        # cmd_vel expects rad/s — convert here, not in mission_control_node.
        twist.angular.z = math.radians(float(msg.yaw_rate))
        self.pub_cmd_vel.publish(twist)

        if msg.drop_ball:
            self.trigger_drop_ball()

    # ── Servo ────────────────────────────────────────────────────────────────

    def trigger_drop_ball(self):
        if self.command_client is None:
            self.get_logger().error("MAVROS CommandLong client unavailable — cannot drop")
            return
        if not self.command_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error("MAVROS /cmd/command service not available")
            return

        request = CommandLong.Request()
        request.broadcast = False
        request.command = 183        # MAV_CMD_DO_SET_SERVO
        request.param1 = float(self._servo_channel)
        request.param2 = float(self._servo_pwm_drop)
        request.param3 = 0.0
        request.param4 = 0.0
        future = self.command_client.call_async(request)
        future.add_done_callback(self._drop_response_cb)

    def _drop_response_cb(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo drop command accepted by FC")
            else:
                self.get_logger().error(
                    f"Servo drop rejected: result={response.result}")
        except Exception as e:
            self.get_logger().error(f"Servo drop service call failed: {e}")


def main():
    rclpy.init()
    node = ActionMavrosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
