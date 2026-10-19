"""Segmentation instances -> Detections. Only perception_node and offline tools call this,
so geometry is computed once per frame (the mission never sees images)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uav_core.config import PerceptionConfig
from uav_core.perception.color import color_in_contour
from uav_core.perception.filters import overlaps_h_landing
from uav_core.perception.geometry import extract_geometry
from uav_core.types import Detection, TargetClass


@dataclass(frozen=True, slots=True)
class RawInstance:
    """One instance from a segmentation backend, before any geometry."""

    cls: TargetClass
    mask: np.ndarray  # float or bool, model-output resolution
    xyxy: tuple[float, float, float, float]  # frame pixels
    conf: float


def detect(
    instances: Sequence[RawInstance], frame_bgr: np.ndarray, cfg: PerceptionConfig
) -> list[Detection]:
    """Circles that pass the fill-ratio and H_landing filters (with their colour) plus every
    H_landing box, in the backend's order. Semicircles are dropped: nothing consumes them."""
    frame_h, frame_w = frame_bgr.shape[:2]
    h_boxes = [tuple(int(v) for v in i.xyxy) for i in instances if i.cls is TargetClass.H_LANDING]
    detections: list[Detection] = []
    for inst in instances:
        if inst.cls is TargetClass.H_LANDING:
            x1, y1, x2, y2 = (int(v) for v in inst.xyxy)
            detections.append(
                Detection(
                    cls=inst.cls,
                    color=None,
                    cx=(x1 + x2) // 2,
                    cy=(y1 + y2) // 2,
                    radius=(x2 - x1) // 2,
                    bbox=(x1, y1, x2, y2),
                    fill_ratio=0.0,
                    conf=inst.conf,
                    edge_clipped=False,
                )
            )
            continue
        if inst.cls is not TargetClass.CIRCLE:
            continue
        geo = extract_geometry(inst.mask, frame_w, frame_h, cfg.edge_clip_margin_px)
        if geo is None or geo.fill_ratio < cfg.fill_ratio_min_circle:
            continue
        if h_boxes and overlaps_h_landing(
            geo.cx, geo.cy, geo.radius, h_boxes, cfg.h_landing_overlap_thresh
        ):
            continue
        color = color_in_contour(
            frame_bgr,
            geo.cx,
            geo.cy,
            geo.radius,
            geo.contour,
            cfg.hsv_ranges,
            cfg.color_match_threshold,
        )
        r = geo.radius
        detections.append(
            Detection(
                cls=inst.cls,
                color=color,
                cx=geo.cx,
                cy=geo.cy,
                radius=r,
                bbox=(geo.cx - r, geo.cy - r, geo.cx + r, geo.cy + r),
                fill_ratio=geo.fill_ratio,
                conf=inst.conf,
                edge_clipped=geo.edge_clipped,
            )
        )
    return detections
