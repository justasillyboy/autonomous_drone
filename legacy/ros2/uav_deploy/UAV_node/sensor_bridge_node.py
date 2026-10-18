"""sensor_bridge_node — BNO080 IMU reader, per node_split_deploy.md §2 Node 3.

Publishes sensor_msgs/Imu on /bno080/imu at ~100 Hz.  Includes:
  • orientation quaternion — mission_control_node converts to (pitch, roll, yaw)
    via tf_transformations.euler_from_quaternion.  yaw_deg is REQUIRED for
    MissionStateMachine._world_vel_to_body and geofence rotations.
  • angular_velocity (rad/s) — converted to deg/s by mission_control_node to
    feed SensorData.pitch_rate_dps / roll_rate_dps (feedforward guard).

MTF-01 ToF + optical flow come via MAVROS, not through this node.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Imu


class SensorBridgeNode(Node):
    def __init__(self):
        super().__init__('sensor_bridge_node')

        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('i2c_address', 0x4A)
        self.declare_parameter('frame_id', 'bno080')
        rate_hz = float(self.get_parameter('rate_hz').value)
        self._frame_id = self.get_parameter('frame_id').value

        # /bno080/imu uses Best Effort QoS (lossy is acceptable for 100 Hz IMU).
        qos_imu = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_imu = self.create_publisher(Imu, '/bno080/imu', qos_imu)

        # Initialise BNO080.  adafruit_bno08x.i2c.BNO08X_I2C is the correct
        # class (not BNO08X); the previous FEATURE_* attribute access was
        # invalid — reports are enabled via BNO_REPORT_* constants.
        try:
            import board
            import busio
            from adafruit_bno08x import (
                BNO_REPORT_GAME_ROTATION_VECTOR,
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_ROTATION_VECTOR,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
        except ImportError as e:
            self.get_logger().error(f"adafruit_bno08x import failed: {e}")
            raise

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
        addr = int(self.get_parameter('i2c_address').value)
        self.bno = BNO08X_I2C(i2c, address=addr)

        # GAME_ROTATION_VECTOR ignores the magnetometer (BNO080 mag calibration is
        # unreliable indoors).  If it is not available on the firmware we fall
        # back to the full ROTATION_VECTOR feature.
        try:
            self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
            self._quat_attr = 'game_quaternion'
        except Exception:
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self._quat_attr = 'quaternion'
        self.bno.enable_feature(BNO_REPORT_GYROSCOPE)

        period = 1.0 / max(rate_hz, 1.0)
        self.timer = self.create_timer(period, self.publish_imu_data)
        self.get_logger().info(
            f"sensor_bridge_node started (BNO080 @ 0x{addr:02X}, {rate_hz:.0f} Hz, "
            f"rotation={self._quat_attr})")

    def publish_imu_data(self):
        try:
            quat = getattr(self.bno, self._quat_attr)
            gyro = self.bno.gyro
        except Exception as e:
            # Transient I2C hiccups are common; warn occasionally rather than
            # crashing the node.
            self.get_logger().warn(f"BNO080 read failed: {e}", throttle_duration_sec=5.0)
            return
        if quat is None or gyro is None:
            self.get_logger().warn("BNO080 returned None", throttle_duration_sec=5.0)
            return

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self._frame_id

        # adafruit_bno08x returns quaternion as (i, j, k, real) = (x, y, z, w).
        qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        # Normalise to guard against firmware returning slightly denormalised quats.
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
        imu_msg.orientation.x = qx / n
        imu_msg.orientation.y = qy / n
        imu_msg.orientation.z = qz / n
        imu_msg.orientation.w = qw / n

        # Angular velocity is rad/s (body frame).
        imu_msg.angular_velocity.x = float(gyro[0])
        imu_msg.angular_velocity.y = float(gyro[1])
        imu_msg.angular_velocity.z = float(gyro[2])

        # Covariance: negative-first-element convention in REP-145 means "unknown".
        imu_msg.orientation_covariance[0] = -1.0
        imu_msg.angular_velocity_covariance[0] = -1.0
        imu_msg.linear_acceleration_covariance[0] = -1.0

        self.pub_imu.publish(imu_msg)


def main():
    rclpy.init()
    node = SensorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
