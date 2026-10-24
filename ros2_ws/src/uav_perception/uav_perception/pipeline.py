"""Frame -> Detections, plus sensor_msgs/Image <-> numpy without cv_bridge. No ROS imports.

cv_bridge is avoided on purpose: it is compiled against the system numpy, while uav_core and
uav_edge are pip-installed and may bring a different one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from uav_core.config import PerceptionConfig
from uav_core.perception import detect
from uav_core.perception.camera import Undistorter
from uav_core.types import Detection
from uav_edge.runtime import Backend

_CHANNELS = {"bgr8": 3, "rgb8": 3, "mono8": 1}


def image_to_bgr(msg: Any) -> np.ndarray:
    """sensor_msgs/Image (bgr8, rgb8 or mono8) -> HxWx3 uint8 BGR."""
    if msg.encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    ch = _CHANNELS[msg.encoding]
    rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = rows[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if msg.encoding == "rgb8":
        return np.ascontiguousarray(img[:, :, ::-1])
    if msg.encoding == "mono8":
        return np.repeat(img, 3, axis=2)
    return img


def compressed_to_bgr(msg: Any) -> np.ndarray:
    """sensor_msgs/CompressedImage (JPEG or PNG) -> BGR. Used for HIL, where Gazebo images
    cross the Ethernet link compressed."""
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode a {msg.format!r} image")
    return img


def fill_image_msg(msg: Any, bgr: np.ndarray) -> Any:
    h, w = bgr.shape[:2]
    msg.height, msg.width, msg.encoding = h, w, "bgr8"
    msg.is_bigendian, msg.step = 0, w * 3
    msg.data = np.ascontiguousarray(bgr).tobytes()
    return msg


@dataclass
class PerceptionPipeline:
    backend: Backend
    cfg: PerceptionConfig
    undistort: Undistorter | None = None

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """(the frame the detections refer to, detections)."""
        frame = self.undistort(frame_bgr) if self.undistort is not None else frame_bgr
        return frame, detect(self.backend.infer(frame), frame, self.cfg)
