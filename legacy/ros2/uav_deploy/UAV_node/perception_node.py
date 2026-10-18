"""perception_node — YOLO11n-seg inference + filtering, per node_split_deploy.md §2 Node 2.

All inference runs here (never split YOLO across nodes).  For every frame:
  1. Run YOLO / Hailo inference.
  2. Collect H_landing bboxes (for cross-class suppression).
  3. For every mask: _extract_geometry → apply three filters (fill_ratio,
     MAX_FIRE_DIAMETER_RATIO, H_LANDING_OVERLAP) → classify colour → compute
     vision-only altitude estimate z_est_raw = f_px·D_real / (2·radius).
  4. Publish VisionTargetsMsg on /perception/targets (Reliable, depth=1) with
     header.stamp copied from /camera/image_raw so mission_control_node can
     time-sync IMU samples against frame-grab time.

f_px is taken from /camera/camera_info.K[0] (latched).  Falls back to the
ROS param f_px_fallback (default 500 — DEFAULT_F_PX from mission.py) with a
warning log until CameraInfo arrives.
"""
import json

import cv2
import numpy as np
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

# mission.py is the single source of truth for constants and helpers.
from main import mission
from main.mission import (
    DEFAULT_F_PX,
    FILL_RATIO_MIN_CIRCLE,
    FIRE_REAL_DIAMETERS,
    MAX_FIRE_DIAMETER_RATIO,
    _extract_geometry,
    collect_h_landing_bboxes,
    color_of_mask,
    load_hsv_calibration,
)

try:
    from uav_deploy.msg import VisionTarget, VisionTargetsMsg
except ImportError:  # pragma: no cover — fallback when built under a different interfaces pkg
    from uav_interfaces.msg import VisionTarget, VisionTargetsMsg


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Parameters
        self.declare_parameter('weights', 'best.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('hsv_calib', '')
        self.declare_parameter('f_px_fallback', float(DEFAULT_F_PX))
        self.declare_parameter('debug_image_rate_hz', 5.0)

        self._weights_path = self.get_parameter('weights').value
        self._conf = float(self.get_parameter('conf_threshold').value)
        self._f_px = float(self.get_parameter('f_px_fallback').value)
        self._f_px_from_calib = False
        self._debug_min_period = 1.0 / max(
            float(self.get_parameter('debug_image_rate_hz').value), 0.1)
        self._last_debug_pub = 0.0

        self.bridge = CvBridge()

        # QoS — see node_split_deploy.md §3e.
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
        qos_targets = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_image)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, qos_info)

        self.pub_targets = self.create_publisher(
            VisionTargetsMsg, '/perception/targets', qos_targets)
        self.pub_debug = self.create_publisher(
            Image, '/perception/debug_image', qos_image)

        # Load YOLO model.
        from ultralytics import YOLO
        self.model = YOLO(self._weights_path)
        self._class_names = dict(self.model.names)

        # HSV calibration — loaded here (not in mission_control_node) because
        # color_of_mask runs in this node.  Fall back to built-in COLOR_HSV
        # on any error so startup is never blocked by a missing file.
        hsv_calib = self.get_parameter('hsv_calib').value
        if hsv_calib:
            try:
                load_hsv_calibration(hsv_calib)
                self.get_logger().info(f"Loaded HSV calibration from {hsv_calib}")
            except (FileNotFoundError, ValueError, KeyError) as e:
                self.get_logger().warn(
                    f"HSV calib load failed ({e}) — using built-in defaults")
        else:
            self.get_logger().warn(
                "hsv_calib parameter empty — using built-in COLOR_HSV defaults")

        self.frame_count = 0
        self.get_logger().info(
            f"perception_node started (weights={self._weights_path}, "
            f"f_px_fallback={self._f_px:.1f})")

    def camera_info_callback(self, msg: CameraInfo):
        # K is a row-major 3x3 intrinsic matrix; K[0]=fx, K[4]=fy.
        fx = float(msg.k[0])
        if fx > 0.0:
            if not self._f_px_from_calib or abs(fx - self._f_px) > 0.5:
                self.get_logger().info(f"CameraInfo received → f_px = {fx:.1f}")
            self._f_px = fx
            self._f_px_from_calib = True

    def image_callback(self, msg: Image):
        # Carry the camera-grab stamp downstream so mission_control_node can
        # time-sync IMU data to the instant this frame was captured.
        frame_stamp = msg.header.stamp

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_h, frame_w = frame.shape[:2]

        results = self.model(frame, conf=self._conf, verbose=False)
        if not results:
            self._publish_empty(frame_stamp)
            return
        result = results[0]

        has_debug_sub = self.pub_debug.get_subscription_count() > 0
        debug_frame = frame.copy() if has_debug_sub else None

        # Step 1: collect H_landing bounding boxes for cross-class suppression.
        h_bboxes = collect_h_landing_bboxes(result, self._class_names)

        # Step 2: per-mask processing.
        targets_msg = VisionTargetsMsg()
        targets_msg.header.stamp = frame_stamp
        targets_msg.header.frame_id = msg.header.frame_id or 'camera'

        if result.masks is not None and result.boxes is not None:
            for mask_obj, box in zip(result.masks.data, result.boxes):
                cls_id = int(box.cls[0].item())
                class_name = self._class_names.get(cls_id, "")
                conf_score = float(box.conf[0].item())

                geo = _extract_geometry(mask_obj, frame_w, frame_h)
                if geo is None:
                    continue
                cx, cy, radius, fr, cnt_frame, edge_clipped = geo

                color = ""
                z_est_raw = 0.0

                if class_name == "circle":
                    # Filter 1 — fill ratio (rejects hollow arcs, H_landing rings).
                    if fr < FILL_RATIO_MIN_CIRCLE:
                        continue
                    # Filter 2 — absolute diameter cap (rejects court boundary arcs).
                    if (2.0 * radius) / max(1, frame_w) > MAX_FIRE_DIAMETER_RATIO:
                        continue
                    # Filter 3 — cross-class NMS against H_landing.
                    if h_bboxes and mission.overlaps_h_landing(
                            cx, cy, radius, h_bboxes):
                        continue

                    # Colour classification — color_of_mask needs a binary mask
                    # with the SAME shape as the bgr frame it is given.
                    if radius > 0:
                        color = self._classify_color(
                            cnt_frame, cx, cy, radius, frame, frame_w, frame_h) or ""

                    # Vision altitude estimate (fallback only).  Trust the
                    # radius only when the circle is fully inside the frame.
                    if radius > 0 and not edge_clipped:
                        D_real = FIRE_REAL_DIAMETERS.get(color, 1.0)
                        z_est_raw = self._f_px * D_real / (2.0 * float(radius))
                    else:
                        z_est_raw = 0.0

                elif class_name == "H_landing":
                    # Not filtered — mission_control_node needs the bbox to
                    # drive the FIND_HELIPAD approach.  No colour / z_est.
                    pass

                elif class_name == "semicircle":
                    # Passed through for diagnostic overlay.  No filtering
                    # because semicircle is a TZ-gate indicator, not a drop target.
                    pass
                else:
                    continue

                vt = VisionTarget()
                vt.cls_id = cls_id
                vt.color = color
                vt.cx = float(cx)
                vt.cy = float(cy)
                vt.radius = float(radius)
                vt.fill_ratio = float(fr)
                vt.z_est_raw = float(z_est_raw)
                vt.conf = conf_score
                vt.edge_clipped = bool(edge_clipped)
                targets_msg.targets.append(vt)

                if debug_frame is not None:
                    overlay_color = (0, 255, 0) if class_name == "circle" else \
                                    (255, 128, 0) if class_name == "H_landing" else \
                                    (0, 0, 255)
                    cv2.circle(debug_frame, (int(cx), int(cy)),
                               int(radius), overlay_color, 2)
                    label = f"{class_name}/{color or '?'} fr={fr:.2f}"
                    cv2.putText(debug_frame, label, (int(cx), max(15, int(cy) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 1)

        self.pub_targets.publish(targets_msg)

        # Debug image is rate-limited to keep it off the critical path under load.
        if debug_frame is not None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_debug_pub >= self._debug_min_period:
                dbg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
                dbg.header.stamp = frame_stamp
                dbg.header.frame_id = msg.header.frame_id or 'camera'
                self.pub_debug.publish(dbg)
                self._last_debug_pub = now

        self.frame_count += 1
        if self.frame_count % 150 == 0:
            self.get_logger().debug(
                f"perception: {self.frame_count} frames | "
                f"{len(targets_msg.targets)} targets last frame | "
                f"f_px={self._f_px:.1f}{' (calib)' if self._f_px_from_calib else ' (fallback)'}")

    def _classify_color(self, cnt_frame, cx, cy, radius, frame,
                        frame_w, frame_h):
        """Build a small ROI mask matching the circle bbox, then call color_of_mask."""
        x1 = max(0, int(cx - radius))
        y1 = max(0, int(cy - radius))
        x2 = min(frame_w, int(cx + radius))
        y2 = min(frame_h, int(cy + radius))
        if x2 <= x1 or y2 <= y1:
            return None
        roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cnt_roi = (cnt_frame.reshape(-1, 2) - np.array([x1, y1], dtype=np.int32))
        cnt_roi = np.clip(cnt_roi, 0, [roi_mask.shape[1] - 1, roi_mask.shape[0] - 1])
        cv2.fillPoly(roi_mask, [cnt_roi.reshape(-1, 1, 2)], 255)
        roi_bgr = frame[y1:y2, x1:x2]
        return color_of_mask(roi_mask, roi_bgr)

    def _publish_empty(self, stamp):
        msg = VisionTargetsMsg()
        msg.header.stamp = stamp
        self.pub_targets.publish(msg)


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
