"""Circle geometry from a low-resolution segmentation mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CircleGeometry:
    cx: int
    cy: int
    radius: int
    fill_ratio: float
    contour: np.ndarray  # int32 (N, 1, 2), frame pixels
    edge_clipped: bool


def extract_geometry(
    mask: np.ndarray, frame_w: int, frame_h: int, edge_clip_margin_px: int
) -> CircleGeometry | None:
    """Fit the enclosing circle of the largest blob in `mask` and scale it to the frame.

    Contours are found on the small mask and scaled, never by resizing the mask to the
    frame (that is the slow path on the Pi). The radius comes from a second fit on the
    float contour in frame space, so non-square scaling does not bias it; the fill ratio
    uses the mask-space fit. Returns None when the mask is empty.
    """
    h_m, w_m = mask.shape
    mask_small = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt_small = max(contours, key=cv2.contourArea)
    sx = frame_w / w_m
    sy = frame_h / h_m
    _, radius_small = cv2.minEnclosingCircle(cnt_small)
    cnt_frame_float = (
        cnt_small.reshape(-1, 2).astype(np.float32) * np.array([sx, sy], dtype=np.float32)
    ).reshape(-1, 1, 2)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt_frame_float)
    contour = cnt_frame_float.astype(np.int32)
    bx, by, bw, bh = cv2.boundingRect(contour)
    m = edge_clip_margin_px
    edge_clipped = bx <= m or by <= m or bx + bw >= frame_w - m or by + bh >= frame_h - m
    full_area = max(1.0, np.pi * radius_small**2)
    fill = min(int(mask_small.sum()) / full_area, 1.0)
    return CircleGeometry(int(cx), int(cy), int(radius), float(fill), contour, bool(edge_clipped))


def diameter_ratio(radius: float, frame_w: int) -> float:
    """Circle diameter as a fraction of the frame width."""
    return (2.0 * radius) / float(frame_w)
