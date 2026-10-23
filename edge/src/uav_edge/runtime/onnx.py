"""ONNX Runtime backend: runs an Ultralytics YOLO-seg export (no built-in NMS) on the CPU.

Used on the Mac for offline videos and as the Pi 5 CPU baseline against the Hailo-8L.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from uav_core.perception import RawInstance

from uav_edge.runtime.base import class_map
from uav_edge.runtime.letterbox import Letterbox
from uav_edge.runtime.yolo_seg import decode


def to_blob(img_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR HWC -> float32 RGB NCHW in [0, 1]."""
    rgb = img_bgr[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


class OnnxBackend:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.5,
        iou: float = 0.7,
        providers: list[str] | None = None,
    ):
        self._session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        meta = self._session.get_modelmeta().custom_metadata_map
        if "names" not in meta:
            raise ValueError(f"{model_path}: no 'names' metadata; export it with Ultralytics")
        self.classes = class_map(meta["names"])
        self.conf = conf
        self.iou = iou

    def infer(self, frame_bgr: np.ndarray) -> list[RawInstance]:
        h, w = frame_bgr.shape[:2]
        lb = Letterbox(w, h, self.in_w, self.in_h)
        pred, proto = self._session.run(None, {self._input_name: to_blob(lb.apply(frame_bgr))})
        return decode(pred, proto, lb, self.classes, self.conf, self.iou)
