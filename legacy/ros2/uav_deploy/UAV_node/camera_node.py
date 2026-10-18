import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
from UAV_node.undistort_utils import SimpleUndistorter

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # QoS Profile: SensorData (Best Effort)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.pub = self.create_publisher(Image, '/camera/image_raw', qos_profile)
        self.bridge = CvBridge()

        # Open camera device
        self.cap = cv2.VideoCapture('/dev/video0')
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera device")
            return

        # Lock AWB and AEC using v4l2
        subprocess.run(['v4l2-ctl', '--set-ctrl=white_balance_auto_preset=0'], check=True)
        subprocess.run(['v4l2-ctl', '--set-ctrl=exposure_auto=1'], check=True)

        # Load undistortion matrix using SimpleUndistorter
        from ament_index_python.packages import get_package_share_directory
        import os
        calib_path = self.declare_parameter('calib_path', os.path.join(get_package_share_directory('uav_deploy'), 'config', 'calibration.npz')).get_parameter_value().string_value
        self.undistorter = SimpleUndistorter(calib_path)

        self.frame_count = 0
        self.timer = self.create_timer(0.03, self.publish_image)  # ~30 FPS

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Camera frame read failed")
            return
        # Apply undistortion
        frame = self.undistorter.undistort(frame)
        # Publish image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().debug(f"Camera: {self.frame_count} frames")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()