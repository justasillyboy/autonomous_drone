"""camera_node — IMX219 frame source + CameraInfo publisher.

Responsibilities (see node_split_deploy.md §2 Node 1):
  • Grab frames from /dev/video0 and apply calibration-based undistortion.
  • Lock AWB/AEC via v4l2-ctl so colour thresholds stay stable under match
    lighting (same locking done by hsv_calibrator.py).
  • Publish /camera/image_raw (sensor_msgs/Image, Best Effort) **and**
    /camera/camera_info (sensor_msgs/CameraInfo, Reliable + transient-local)
    so perception_node / mission_control_node can read the real f_px from
    CameraInfo.K[0].  Without this, downstream silently falls back to
    DEFAULT_F_PX = 500 — a bug flagged in CLAUDE.md Danger Zones.
  • header.stamp is set once per frame at grab time so the pipeline's
    IMU time-sync (MissionStateMachine._get_synced_imu) stays accurate.
"""
import os
import subprocess

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image

from UAV_node.undistort_utils import SimpleUndistorter


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('calib_path', '')
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('lock_awb_aec', True)
        self.declare_parameter('frame_id', 'camera')

        calib_path = self.get_parameter('calib_path').value
        if not calib_path:
            # Fallback to packaged calibration if param not provided.
            try:
                from ament_index_python.packages import get_package_share_directory
                calib_path = os.path.join(
                    get_package_share_directory('uav_deploy'),
                    'config', 'calibration.npz')
            except Exception:
                calib_path = ''
        video_device = self.get_parameter('video_device').value
        fps = float(self.get_parameter('fps').value)
        self._frame_id = self.get_parameter('frame_id').value

        # QoS: image stream is Best Effort (skip stale frames under load);
        # CameraInfo is Reliable + transient-local so late subscribers still
        # receive the latched focal length.
        qos_image = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_info = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', qos_image)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', qos_info)
        self.bridge = CvBridge()

        # Lock auto-white-balance and auto-exposure.  Use check=False so a
        # missing control (e.g. different kernel driver) warns rather than
        # crashing the node.
        if self.get_parameter('lock_awb_aec').value:
            for args in (
                ['v4l2-ctl', '-d', video_device, '--set-ctrl=white_balance_auto_preset=0'],
                ['v4l2-ctl', '-d', video_device, '--set-ctrl=exposure_auto=1'],
            ):
                r = subprocess.run(args, capture_output=True)
                if r.returncode != 0:
                    self.get_logger().warn(
                        f"v4l2 control failed ({' '.join(args)}): "
                        f"{r.stderr.decode(errors='ignore').strip()}")

        # Open camera.
        self.cap = cv2.VideoCapture(video_device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera device {video_device}")
            raise RuntimeError(f"Camera open failed: {video_device}")

        # Load calibration (required — fx/fy populate CameraInfo).
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(
                f"calibration.npz not found at '{calib_path}'. "
                "Run camera_calibration.py before flight; set ROS param 'calib_path'.")
        self.undistorter = SimpleUndistorter(calib_path)
        self.get_logger().info(
            f"Loaded calibration from {calib_path} — fx={self.undistorter.fx:.1f}px, "
            f"fy={self.undistorter.fy:.1f}px")

        # Cache immutable CameraInfo — only header.stamp and width/height are
        # refreshed per frame.
        self._info_template = CameraInfo()
        self._info_template.distortion_model = 'plumb_bob'
        self._info_template.d = [float(v) for v in self.undistorter.dist_coeffs.flatten()]
        self._info_template.k = [float(v) for v in self.undistorter.camera_matrix.flatten()]
        self._info_template.p = self._info_template.k[:3] + [0.0] + \
                                self._info_template.k[3:6] + [0.0] + \
                                self._info_template.k[6:9] + [0.0]
        self._info_template.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        self.frame_count = 0
        period = 1.0 / max(fps, 1.0)
        self.timer = self.create_timer(period, self.publish_image)
        self.get_logger().info(f"camera_node started at {fps:.1f} Hz")

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Camera frame read failed")
            return
        # Single stamp shared by Image + CameraInfo — downstream correlates on it.
        stamp = self.get_clock().now().to_msg()

        frame = self.undistorter.undistort(frame)
        h, w = frame.shape[:2]

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self._frame_id
        self.pub_image.publish(img_msg)

        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = self._frame_id
        info_msg.height = h
        info_msg.width = w
        info_msg.distortion_model = self._info_template.distortion_model
        info_msg.d = list(self._info_template.d)
        info_msg.k = list(self._info_template.k)
        info_msg.r = list(self._info_template.r)
        info_msg.p = list(self._info_template.p)
        self.pub_info.publish(info_msg)

        self.frame_count += 1
        if self.frame_count % 150 == 0:
            self.get_logger().debug(f"camera_node: {self.frame_count} frames published")

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
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
