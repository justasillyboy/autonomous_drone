"""perception_node: frames -> backend -> uav_core.perception -> /perception/detections.

input:=topic subscribes /camera/image_raw (sim), input:=compressed its JPEG version (HIL,
over Ethernet), input:=v4l2 reads the camera directly (on the Pi), so full frames never cross
DDS. Detections carry the frame's capture stamp.
/perception/debug_image (detections drawn) is published only while someone subscribes, at
most debug_rate_hz.
"""

from __future__ import annotations

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from uav_core.config import MissionConfig, load_mission_config, with_hsv_ranges
from uav_core.messages import fill_detection_msg
from uav_core.perception.camera import Calibration, Undistorter
from uav_core.perception.color import load_hsv_calibration
from uav_core.viz import draw_detections
from uav_edge.runtime import load_backend
from uav_interfaces.msg import Detection, Detections

from uav_perception.pipeline import (
    PerceptionPipeline,
    compressed_to_bgr,
    fill_image_msg,
    image_to_bgr,
)

RELIABLE = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        p = self.declare_parameter
        weights = p("weights", "").value
        conf = float(p("conf", 0.5).value)
        config_yaml = p("config_yaml", "").value
        hsv_calib = p("hsv_calib", "").value
        calib = p("calibration", "").value
        source = p("input", "topic").value
        self.debug_period_s = 1.0 / max(float(p("debug_rate_hz", 5.0).value), 0.1)
        if not weights:
            raise ValueError("parameter 'weights' (.onnx or .pt) is required")

        cfg = load_mission_config(config_yaml) if config_yaml else MissionConfig()
        if hsv_calib:
            cfg = with_hsv_ranges(cfg, load_hsv_calibration(hsv_calib))
        undistort = Undistorter(Calibration.load(calib)) if calib else None
        self.pipeline = PerceptionPipeline(load_backend(weights, conf), cfg.perception, undistort)
        self.pub = self.create_publisher(Detections, "/perception/detections", RELIABLE)
        self.pub_debug = self.create_publisher(Image, "/perception/debug_image", 1)
        self._last_debug_s = 0.0

        if source == "topic":
            self.create_subscription(
                Image, "/camera/image_raw", self.on_image, qos_profile_sensor_data
            )
        elif source == "compressed":  # HIL: JPEG over the Ethernet link
            self.create_subscription(
                CompressedImage,
                "/camera/image_raw/compressed",
                self.on_compressed,
                qos_profile_sensor_data,
            )
        elif source == "v4l2":
            self.cap = cv2.VideoCapture(p("device", "/dev/video0").value, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(p("width", 640).value))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(p("height", 480).value))
            if not self.cap.isOpened():
                raise RuntimeError("cannot open the v4l2 camera")
            self.create_timer(1.0 / float(p("fps", 30.0).value), self.on_grab)
        else:
            raise ValueError(f"input must be 'topic', 'compressed' or 'v4l2', not {source!r}")
        self.get_logger().info(f"perception_node ready ({source}, {weights})")

    def on_image(self, msg: Image) -> None:
        self.handle(image_to_bgr(msg), msg.header)

    def on_compressed(self, msg: CompressedImage) -> None:
        self.handle(compressed_to_bgr(msg), msg.header)

    def on_grab(self) -> None:
        stamp = self.get_clock().now()
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("camera read failed", throttle_duration_sec=2.0)
            return
        header = Image().header
        header.stamp = stamp.to_msg()
        header.frame_id = "camera"
        self.handle(frame, header)

    def handle(self, frame, header) -> None:
        used, dets = self.pipeline.process(frame)
        out = Detections()
        out.header = header
        out.height, out.width = used.shape[:2]
        out.detections = [fill_detection_msg(Detection(), d) for d in dets]
        self.pub.publish(out)
        self.publish_debug(used, dets, header)

    def publish_debug(self, frame, dets, header) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pub_debug.get_subscription_count() == 0:
            return
        if now - self._last_debug_s < self.debug_period_s:
            return
        self._last_debug_s = now
        img = frame.copy()
        draw_detections(img, dets)
        msg = fill_image_msg(Image(), img)
        msg.header = header
        self.pub_debug.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
