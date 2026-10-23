"""bno080_node: BNO080 over I2C -> sensor_msgs/Imu on /imu/data (body FLU, REP 103).

Uses the game rotation vector (no magnetometer: indoor calibration is unreliable), so the
yaw origin is arbitrary; the mission only uses yaw differences. `mount_rpy_deg` is the
sensor's rotation relative to base_link: check it on the bench (tilt the nose up, pitch in
/imu/data must go negative; turn clockwise seen from above, yaw must decrease).
50 Hz by default, following the old feedback about I2C load.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from uav_hardware.imu_math import from_rpy_deg, normalise, rotate_vector, sensor_to_body


class Bno080Node(Node):
    def __init__(self) -> None:
        super().__init__("bno080_node")
        rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        address = int(self.declare_parameter("i2c_address", 0x4A).value)
        mount = list(self.declare_parameter("mount_rpy_deg", [0.0, 0.0, 0.0]).value)
        self.frame_id = str(self.declare_parameter("frame_id", "base_link").value)
        self.q_mount = from_rpy_deg(*mount)

        import board  # hardware-only dependencies, imported on the Pi
        import busio
        from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_GYROSCOPE
        from adafruit_bno08x.i2c import BNO08X_I2C

        self.bno = BNO08X_I2C(busio.I2C(board.SCL, board.SDA, frequency=400_000), address=address)
        self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
        self.pub = self.create_publisher(Imu, "/imu/data", qos_profile_sensor_data)
        self.create_timer(1.0 / max(rate_hz, 1.0), self.on_timer)
        self.get_logger().info(f"BNO080 at 0x{address:02X}, {rate_hz:.0f} Hz, mount {mount}")

    def on_timer(self) -> None:
        try:
            quat, gyro = self.bno.game_quaternion, self.bno.gyro
        except (OSError, RuntimeError) as exc:  # transient I2C errors are common
            self.get_logger().warn(f"BNO080 read failed: {exc}", throttle_duration_sec=5.0)
            return
        if quat is None or gyro is None:
            return
        q = sensor_to_body(normalise(tuple(float(c) for c in quat)), self.q_mount)
        w = rotate_vector(self.q_mount, tuple(float(c) for c in gyro))
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = w
        msg.linear_acceleration_covariance[0] = -1.0  # REP 145: not provided
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Bno080Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
