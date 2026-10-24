"""mission_node: detections + sensors -> uav_core mission -> FlightCommand.

Subscribes /perception/detections and the sensors; never subscribes to images. The mission
clock is the detections' capture stamp, so use_sim_time works. A 0.5 s watchdog publishes a
hover command when perception goes silent, without touching the mission state.
"""

from __future__ import annotations

import contextlib

import rclpy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import ExtendedState, State
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from uav_core.config import MissionConfig, load_mission_config
from uav_core.messages import (
    detection_from_msg,
    fill_command_msg,
    fill_status_msg,
    stamp_to_s,
)
from uav_core.types import FlightCommand, Round
from uav_interfaces.msg import Detections, MissionStatus
from uav_interfaces.msg import FlightCommand as FlightCommandMsg

from uav_mission.bridge import MissionRunner, Watchdog

RELIABLE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
WATCHDOG_PERIOD_S = 0.1


class MissionNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_node")
        config_yaml = self.declare_parameter("config_yaml", "").value
        finals = self.declare_parameter("finals", False).value
        skip_takeoff = self.declare_parameter("skip_takeoff", False).value
        timeout = self.declare_parameter("watchdog_timeout_s", 0.5).value

        cfg = load_mission_config(config_yaml) if config_yaml else MissionConfig()
        round_ = Round.FINALS if finals else Round.QUALIFIER
        self.runner = MissionRunner(cfg, round_, skip_takeoff)
        self.watchdog = Watchdog(timeout)

        self.pub_cmd = self.create_publisher(FlightCommandMsg, "/mission/command", RELIABLE)
        self.pub_status = self.create_publisher(MissionStatus, "/mission/status", 10)

        sensor = qos_profile_sensor_data
        self.create_subscription(Detections, "/perception/detections", self.on_detections, RELIABLE)
        self.create_subscription(Imu, "/imu/data", self.on_imu, sensor)
        self.create_subscription(Float64, "/mavros/global_position/rel_alt", self.on_alt, sensor)
        self.create_subscription(Range, "/mavros/rangefinder/rangefinder", self.on_range, sensor)
        self.create_subscription(
            TwistStamped, "/mavros/local_position/velocity_body", self.on_velocity, sensor
        )
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.create_subscription(ExtendedState, "/mavros/extended_state", self.on_extended, 10)
        self.create_service(Trigger, "/mission/reset", self.srv_reset)
        self.create_service(Trigger, "/mission/e_stop", self.srv_e_stop)
        self.create_timer(WATCHDOG_PERIOD_S, self.on_watchdog)
        self.get_logger().info(f"mission_node ready ({round_.value}, skip_takeoff={skip_takeoff})")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- inputs ----------------------------------------------------------------------------

    def on_detections(self, msg: Detections) -> None:
        self.watchdog.feed(self._now_s())
        dets = [detection_from_msg(d) for d in msg.detections]
        out = self.runner.step(stamp_to_s(msg.header.stamp), dets, (msg.width, msg.height))
        if out is None:  # e-stopped: the watchdog timer keeps publishing hover
            return
        self.publish(out.command, msg.header.stamp)

    def on_imu(self, msg: Imu) -> None:
        self.runner.sensors.on_imu(
            stamp_to_s(msg.header.stamp), msg.orientation, msg.angular_velocity
        )

    def on_alt(self, msg: Float64) -> None:
        self.runner.sensors.on_rel_alt(msg.data)

    def on_range(self, msg: Range) -> None:
        self.runner.sensors.on_range(msg.range, msg.min_range, msg.max_range)

    def on_velocity(self, msg: TwistStamped) -> None:
        self.runner.sensors.on_velocity_body_flu(msg.twist.linear.x, msg.twist.linear.y)

    def on_state(self, msg: State) -> None:
        if self.runner.on_armed(msg.armed, self._now_s()):
            self.get_logger().info("ARM: match clock started")

    def on_extended(self, msg: ExtendedState) -> None:
        if self.runner.on_landed_state(msg.landed_state):
            self.get_logger().info("touchdown on the helipad: landing scored")

    # -- outputs ---------------------------------------------------------------------------

    def publish(self, cmd: FlightCommand, stamp) -> None:
        out = fill_command_msg(FlightCommandMsg(), cmd)
        out.header.stamp = stamp
        self.pub_cmd.publish(out)
        status = fill_status_msg(MissionStatus(), self.runner.sm.status(), self.runner.e_stopped)
        status.header.stamp = stamp
        self.pub_status.publish(status)

    def on_watchdog(self) -> None:
        now = self.get_clock().now()
        if self.runner.e_stopped:
            self.publish(self.runner.hover("E_STOP"), now.to_msg())
        elif self.watchdog.expired(now.nanoseconds * 1e-9):
            self.publish(self.runner.hover("WATCHDOG"), now.to_msg())

    def srv_reset(self, _request, response):
        self.runner.reset()
        response.success = True
        response.message = "mission reset: score, phase and balls cleared"
        self.get_logger().info("/mission/reset")
        return response

    def srv_e_stop(self, _request, response):
        self.runner.e_stop()
        self.publish(self.runner.hover("E_STOP"), self.get_clock().now().to_msg())
        response.success = True
        response.message = "e-stop engaged: hovering until /mission/reset"
        self.get_logger().warn("/mission/e_stop")
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):  # a second Ctrl-C during teardown
            node.destroy_node()
        rclpy.try_shutdown()
