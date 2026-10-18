import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from main.mission import MissionStateMachine, SensorData, FlightCommand
from collections import deque
import time
import json

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')

        # Publishers
        # Publisher: dùng String (JSON) cho prototype
        self.pub_flight_command = self.create_publisher(
            String, '/mission/flight_command', 10
        )

        # Subscribers
        self.sub_targets = self.create_subscription(
            String, '/perception/targets', self.target_callback, 10
        )
        self.sub_imu = self.create_subscription(
            Imu, '/bno080/imu', self.imu_callback, 10
        )
        self.sub_tof = self.create_subscription(
            String, '/mavros/distance_sensor/rangefinder_sub', self.tof_callback, 10
        )
        self.sub_velocity = self.create_subscription(
            Twist, '/mavros/local_position/velocity_body', self.velocity_callback, 10
        )
        self.sub_altitude = self.create_subscription(
            String, '/mavros/global_position/rel_alt', self.altitude_callback, 10
        )

        # Mission state machine (khởi tạo đúng args)
        # TODO: lấy f_px, is_finals, takeoff_alt từ param hoặc config
        self.mission = MissionStateMachine(f_px=1.0, is_finals=False, takeoff_alt=1.0)

        # Time-sync buffer for IMU data
        self._imu_buffer = deque(maxlen=60)  # ~1 second at 60 Hz

        # Timer cho publish_flight_command (20Hz)
        self.timer = self.create_timer(0.05, self.publish_flight_command)

        self.get_logger().info("MissionControlNode started")

    def target_callback(self, msg):
        # Nhận JSON string, parse thành list
        try:
            targets = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Target JSON parse error: {e}")
            targets = []
        self.mission.update_target(targets)

    def imu_callback(self, msg):
        # Parse IMU data and push to buffer
        sensor_data = SensorData(
            pitch_deg=msg.orientation.x,
            roll_deg=msg.orientation.y,
            yaw_deg=msg.orientation.z,
            of_vx_ms=None,  # Placeholder for Optical Flow
            of_vy_ms=None   # Placeholder for Optical Flow
        )
        self._imu_buffer.append((time.perf_counter(), sensor_data))

    def tof_callback(self, msg):
        self.get_logger().info(f"Received ToF altitude: {msg.data}")
        self.mission.update_tof_altitude(float(msg.data))

    def velocity_callback(self, msg):
        self.get_logger().info(f"Received velocity: {msg.linear.x}, {msg.linear.y}")
        self.mission.update_velocity(msg.linear.x, msg.linear.y)

    def altitude_callback(self, msg):
        self.get_logger().info(f"Received FC altitude: {msg.data}")
        self.mission.update_fc_altitude(float(msg.data))

    def publish_flight_command(self):
        # Generate flight command from mission state
        flight_command = self.mission.generate_flight_command()
        # Serialize to JSON for prototype
        msg_cmd = String()
        try:
            msg_cmd.data = json.dumps(flight_command.__dict__)
        except Exception as e:
            self.get_logger().error(f"FlightCommand serialize error: {e}")
            return
        self.pub_flight_command.publish(msg_cmd)

    def destroy_node(self):
        super().destroy_node()

def main():
    rclpy.init()
    node = MissionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()