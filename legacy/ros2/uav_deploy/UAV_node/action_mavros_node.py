import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from mavros_msgs.srv import CommandLong
from std_msgs.msg import String

class ActionMavrosNode(Node):
    def __init__(self):
        super().__init__('action_mavros_node')

        # Publisher for MAVROS velocity commands
        self.pub_cmd_vel = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10
        )

        # Subscriber for flight commands
        self.sub_flight_command = self.create_subscription(
            String, '/mission/flight_command', self.flight_command_callback, 10
        )

        # MAVROS CommandLong service client
        self.command_service = self.create_client(CommandLong, '/mavros/cmd/command')

        self.get_logger().info("ActionMavrosNode started")

    def flight_command_callback(self, msg):
        # Parse JSON string
        import json
        try:
            cmd = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"FlightCommand JSON parse error: {e}")
            return
        twist_msg = Twist()
        twist_msg.linear.x = cmd.get('vx', 0.0)
        twist_msg.linear.y = cmd.get('vy', 0.0)
        twist_msg.linear.z = cmd.get('vz', 0.0)
        twist_msg.angular.z = cmd.get('yaw_rate', 0.0)
        self.pub_cmd_vel.publish(twist_msg)
        self.get_logger().debug(f"Published velocity command: {twist_msg}")
        if cmd.get('drop_ball', False):
            self.trigger_drop_ball()

    def trigger_drop_ball(self):
        # Call MAVROS CommandLong service to trigger servo
        if not self.command_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("CommandLong service not available")
            return

        request = CommandLong.Request()
        request.broadcast = False
        request.command = 183  # MAV_CMD_DO_SET_SERVO
        request.param1 = 9    # Servo channel (example: AUX 9)
        request.param2 = 1900  # PWM value to trigger drop

        future = self.command_service.call_async(request)
        future.add_done_callback(self.drop_ball_response_callback)

    def drop_ball_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Ball drop command executed successfully")
            else:
                self.get_logger().error("Ball drop command failed")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main():
    rclpy.init()
    node = ActionMavrosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()