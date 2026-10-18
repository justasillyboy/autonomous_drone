
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from adafruit_bno08x import BNO08X
import board
import busio
from collections import deque
import time
from main.mission import SensorData


class SensorBridgeNode(Node):
    def __init__(self):
        super().__init__('sensor_bridge_node')

        # Publisher for BNO080 IMU data
        self.pub_imu = self.create_publisher(Imu, '/bno080/imu', 10)

        # Initialize I2C for BNO080 using adafruit_bno08x
        i2c = busio.I2C(board.SCL, board.SDA)
        self.bno = BNO08X(i2c)

        # Enable quaternion and gyro
        self.bno.enable_feature(self.bno.FEATURE_ROTATION_VECTOR)
        self.bno.enable_feature(self.bno.FEATURE_GYRO)

        # IMU buffer for time synchronization
        self._imu_buffer = deque(maxlen=60)  # ~1 second at 60 Hz

        # Timer for IMU data publishing (50Hz)
        self.timer = self.create_timer(0.02, self.publish_imu_data)

        self.get_logger().info("SensorBridgeNode started, publishing IMU data at 50Hz")


    def publish_imu_data(self):
        try:
            quat = self.bno.quaternion
            gyro = self.bno.gyro
            if quat is None or gyro is None:
                self.get_logger().warn("No IMU data available from BNO080")
                return

            # Create and populate Imu message
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'bno080'

            # Orientation (quaternion)
            imu_msg.orientation.x = quat[1]
            imu_msg.orientation.y = quat[2]
            imu_msg.orientation.z = quat[3]
            imu_msg.orientation.w = quat[0]

            # Angular velocity (rad/s)
            imu_msg.angular_velocity.x = gyro[0]
            imu_msg.angular_velocity.y = gyro[1]
            imu_msg.angular_velocity.z = gyro[2]

            # Publish IMU data
            self.pub_imu.publish(imu_msg)

            # Buffer the IMU data for synchronization (optional)
            sensor_data = SensorData(
                pitch_deg=None,
                roll_deg=None,
                yaw_deg=None,
                of_vx_ms=None,
                of_vy_ms=None,
            )
            self._imu_buffer.append((time.perf_counter(), sensor_data))

        except Exception as e:
            self.get_logger().error(f"Failed to read BNO080 data: {e}")


    # No need for manual I2C register read with adafruit_bno08x


    def destroy_node(self):
        # No explicit close needed for adafruit_bno08x
        super().destroy_node()

def main():
    rclpy.init()
    node = SensorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()