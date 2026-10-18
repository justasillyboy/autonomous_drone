import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import json
from main.postprocess import _extract_geometry, color_of_mask
from main.mission import load_hsv_calibration, MAX_FIRE_DIAMETER_RATIO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        # QoS Profile for debug image
        qos_debug = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # QoS Profile cho camera (BEST_EFFORT, depth=1)
        qos_camera = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_camera
        )

        # Publish detected targets (JSON string)
        self.pub_targets = self.create_publisher(
            String,
            '/perception/targets',
            10
        )

        # Optional debug image publisher
        self.pub_debug = self.create_publisher(
            Image,
            '/perception/debug_image',
            qos_debug
        )

        # Load YOLO11n-seg model
        self.model = YOLO("yolov11n-seg.pt")

        # Load HSV calibration config
        load_hsv_calibration('hsv_calib.json')

        self.get_logger().info("🔥 Perception Node + YOLO11n-seg Started")

    def image_callback(self, msg):
        # Convert ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO inference
        results = self.model(frame)

        # Process YOLO results
        detected_targets = []
        has_debug_sub = self.pub_debug.get_subscription_count() > 0
        debug_frame = frame.copy() if has_debug_sub else None
        for result in results:
            # Ultralytics: result.masks (số nhiều)
            if not hasattr(result, 'masks') or result.masks is None:
                continue
            for i, mask in enumerate(result.masks.data):
                geo = _extract_geometry(mask, frame.shape[1], frame.shape[0])
                if geo is None:
                    continue
                cx, cy, radius, fill_ratio, _, edge_clipped = geo
                if radius / frame.shape[1] > MAX_FIRE_DIAMETER_RATIO:
                    continue
                color = color_of_mask(mask, frame)
                detected_targets.append({
                    'class': int(result.boxes.cls[i]) if hasattr(result, 'boxes') else 0,
                    'color': color,
                    'cx': float(cx),
                    'cy': float(cy),
                    'radius': float(radius),
                    'z_est_raw': 0.0,
                    'edge_clipped': bool(edge_clipped)
                })
                if has_debug_sub and debug_frame is not None:
                    cv2.circle(debug_frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"{color}", (int(cx), int(cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Publish targets as JSON string
        msg_targets = String()
        msg_targets.data = json.dumps(detected_targets)
        self.pub_targets.publish(msg_targets)

        # Publish debug image nếu có subscriber
        if has_debug_sub and debug_frame is not None:
            msg_debug = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.pub_debug.publish(msg_debug)

        # Log debug mỗi 30 frame
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().debug(f"Perception: {self.frame_count} frames")
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        self.pub_debug.publish(debug_msg)

        self.get_logger().info(f"🎯 Published {len(detected_targets)} targets")

    def destroy_node(self):
        super().destroy_node()

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()