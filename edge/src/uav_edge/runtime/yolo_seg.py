"""Decode raw YOLO-seg outputs in numpy (shared by the ONNX and Hailo backends).

Outputs of an Ultralytics export without built-in NMS:
- pred  (1, 4 + nc + nm, N): box centre/size in network-input pixels, nc class scores
  (already sigmoid), nm mask coefficients
- proto (1, nm, mh, mw): mask prototypes at 1/4 of the input resolution

Masks stay at prototype resolution (mask_ratio 4): uav_core finds contours on the small
mask and scales the contour, which is the fast path on the Pi (legacy decision 4).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from uav_core.perception import RawInstance
from uav_core.types import TargetClass

from uav_edge.runtime.letterbox import Letterbox

_CLASS_OFFSET_PX = 7680.0  # shifts boxes per class so one NMS pass is class-aware


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xy, half = xywh[:, :2], xywh[:, 2:4] / 2
    return np.concatenate([xy - half, xy + half], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, best score first."""
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod((rb - lt).clip(min=0), axis=1)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)


def crop_to_boxes(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zero each mask outside its own box (boxes in mask pixels), as Ultralytics does."""
    _, mh, mw = masks.shape
    cols = np.arange(mw)[None, None, :]
    rows = np.arange(mh)[None, :, None]
    x1, y1, x2, y2 = (boxes[:, k, None, None] for k in range(4))
    inside = (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)
    return masks * inside


def decode(
    pred: np.ndarray,
    proto: np.ndarray,
    letterbox: Letterbox,
    classes: Mapping[int, TargetClass],
    conf_thres: float,
    iou_thres: float = 0.7,
    max_det: int = 50,
) -> list[RawInstance]:
    nm, mh, mw = proto.shape[1:]
    p = pred[0].T
    nc = p.shape[1] - 4 - nm
    scores = p[:, 4 : 4 + nc]
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    keep = conf >= conf_thres
    if not keep.any():
        return []
    p, cls, conf = p[keep], cls[keep], conf[keep]
    boxes = xywh_to_xyxy(p[:, :4])
    idx = nms(boxes + (cls * _CLASS_OFFSET_PX)[:, None], conf, iou_thres)[:max_det]
    boxes, cls, conf, coeffs = boxes[idx], cls[idx], conf[idx], p[idx, 4 + nc :]

    logits = coeffs @ proto[0].reshape(nm, -1)
    masks = (1.0 / (1.0 + np.exp(-logits))).reshape(-1, mh, mw)
    scale = np.array([mw / letterbox.in_w, mh / letterbox.in_h] * 2)
    masks = letterbox.crop_masks(crop_to_boxes(masks, boxes * scale))
    frame_boxes = letterbox.box_to_frame(boxes)
    return [
        RawInstance(
            cls=classes[int(c)],
            mask=masks[k].astype(np.float32),
            xyxy=tuple(float(v) for v in frame_boxes[k]),
            conf=float(conf[k]),
        )
        for k, c in enumerate(cls)
    ]
