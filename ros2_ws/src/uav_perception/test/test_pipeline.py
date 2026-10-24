"""perception_node logic without ROS: image decoding and the frame -> detections pipeline."""

from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from uav_core.config import PerceptionConfig
from uav_core.perception import RawInstance
from uav_core.perception.camera import Calibration, Undistorter
from uav_core.types import Color, TargetClass
from uav_perception.pipeline import (
    PerceptionPipeline,
    compressed_to_bgr,
    fill_image_msg,
    image_to_bgr,
)


def _img_msg(arr, encoding, pad=0):
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    rows = np.zeros((h, w * ch + pad), np.uint8)
    rows[:, : w * ch] = arr.reshape(h, -1)
    return NS(height=h, width=w, encoding=encoding, step=w * ch + pad, data=rows.tobytes())


def test_bgr_rgb_and_mono_decode_with_row_padding():
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 2] = 200  # red
    assert (image_to_bgr(_img_msg(bgr, "bgr8", pad=3)) == bgr).all()
    assert (image_to_bgr(_img_msg(bgr[:, :, ::-1].copy(), "rgb8")) == bgr).all()
    gray = np.full((4, 5), 7, np.uint8)
    assert image_to_bgr(_img_msg(gray, "mono8")).shape == (4, 5, 3)


def test_unsupported_encoding():
    with pytest.raises(ValueError, match="16UC1"):
        image_to_bgr(NS(encoding="16UC1"))


def test_fill_image_round_trips():
    bgr = np.random.default_rng(0).integers(0, 255, (6, 7, 3), np.uint8)
    msg = fill_image_msg(NS(), bgr)
    assert (image_to_bgr(msg) == bgr).all() and msg.step == 21


class _Disc:
    def infer(self, frame):
        h, w = frame.shape[:2]
        yy, xx = np.mgrid[: h // 4, : w // 4]
        mask = ((xx - w // 8) ** 2 + (yy - h // 8) ** 2 <= 10**2).astype(np.float32)
        return [
            RawInstance(
                TargetClass.CIRCLE, mask, (w / 2 - 40, h / 2 - 40, w / 2 + 40, h / 2 + 40), 0.9
            )
        ]


def test_pipeline_undistorts_then_detects():
    frame = np.full((480, 640, 3), (0, 0, 200), np.uint8)
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    und = Undistorter(Calibration(k, np.zeros(5), (640, 480), 0.1))
    used, dets = PerceptionPipeline(_Disc(), PerceptionConfig(), und).process(frame)
    assert used.shape == frame.shape
    assert len(dets) == 1 and dets[0].color is Color.RED


def test_compressed_jpeg_decodes():
    bgr = np.zeros((48, 64, 3), np.uint8)
    bgr[..., 2] = 220
    _, buf = cv2.imencode(".jpg", bgr)
    out = compressed_to_bgr(NS(format="jpeg", data=buf.tobytes()))
    assert out.shape == (48, 64, 3) and abs(int(out[24, 32, 2]) - 220) < 6
    with pytest.raises(ValueError, match="jpeg"):
        compressed_to_bgr(NS(format="jpeg", data=b"not an image"))
