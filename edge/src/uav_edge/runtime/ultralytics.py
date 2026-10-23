"""Ultralytics backend for .pt weights. Needs the optional extra:
`uv sync --extra ultralytics` (pulls in torch)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from uav_core.perception import RawInstance
from uav_core.types import TargetClass

from uav_edge.runtime.base import class_map
from uav_edge.runtime.letterbox import Letterbox


def _np(tensor: Any) -> np.ndarray:
    return tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)


def from_result(result: Any, classes: Mapping[int, TargetClass]) -> list[RawInstance]:
    """One Ultralytics `Results` -> RawInstances.

    `result.masks.data` lies on the padded network canvas (e.g. 384x640 for a 720p frame);
    the padding is cropped so uav_core can scale the mask straight to the frame.
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    masks = _np(result.masks.data)
    frame_h, frame_w = result.orig_shape
    masks = Letterbox(frame_w, frame_h, masks.shape[2], masks.shape[1]).crop_masks(masks)
    boxes = _np(result.boxes.xyxy)
    cls = _np(result.boxes.cls).astype(int)
    conf = _np(result.boxes.conf)
    return [
        RawInstance(
            cls=classes[int(cls[k])],
            mask=masks[k].astype(np.float32),
            xyxy=tuple(float(v) for v in boxes[k]),
            conf=float(conf[k]),
        )
        for k in range(len(cls))
    ]


class UltralyticsBackend:
    def __init__(self, weights: str | Path, conf: float = 0.5, imgsz: int = 640):
        from ultralytics import YOLO  # optional dependency, imported only when used

        self._model = YOLO(str(weights))
        self.classes = class_map(self._model.names)
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, frame_bgr: np.ndarray) -> list[RawInstance]:
        result = self._model(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return from_result(result, self.classes)
