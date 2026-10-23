"""Decode the raw per-scale YOLO-seg heads a compiled Hailo model outputs.

Hailo's DFC cuts YOLOv8/YOLO11-seg before the DFL decode, so the device returns, for each
stride s in (8, 16, 32): box logits (H, W, 4 * reg_max), class logits (H, W, nc) and mask
coefficients (H, W, nm), plus the prototypes (H/4 scale, nm). `decode_heads` rebuilds the
exact tensor an Ultralytics ONNX export emits, (1, 4 + nc + nm, N), so `yolo_seg.decode`
does the rest for every backend. tests/test_heads.py checks it against the ONNX graph cut
at the same nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REG_MAX = 16
STRIDES = (8, 16, 32)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def dfl_distances(box_logits: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    """(H, W, 4 * reg_max) -> (H, W, 4) expected distances (left, top, right, bottom) in
    grid cells."""
    h, w, _ = box_logits.shape
    probs = _softmax(box_logits.reshape(h, w, 4, reg_max), axis=-1)
    return probs @ np.arange(reg_max, dtype=np.float32)


@dataclass(frozen=True)
class ScaleOutputs:
    stride: int
    box: np.ndarray  # (H, W, 4 * reg_max)
    cls: np.ndarray  # (H, W, nc) logits
    coeffs: np.ndarray  # (H, W, nm)


def decode_heads(scales: Sequence[ScaleOutputs], reg_max: int = REG_MAX) -> np.ndarray:
    """Raw heads -> (1, 4 + nc + nm, N): cx, cy, w, h in input pixels, class probabilities,
    mask coefficients; anchors ordered by stride, then row-major, as Ultralytics does."""
    columns = []
    for sc in sorted(scales, key=lambda s: s.stride):
        h, w, _ = sc.box.shape
        ys, xs = np.mgrid[:h, :w].astype(np.float32)
        ax, ay = xs + 0.5, ys + 0.5
        lt_rb = dfl_distances(sc.box, reg_max)
        x1, y1 = ax - lt_rb[..., 0], ay - lt_rb[..., 1]
        x2, y2 = ax + lt_rb[..., 2], ay + lt_rb[..., 3]
        box = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1) * sc.stride
        probs = 1.0 / (1.0 + np.exp(-sc.cls))
        columns.append(np.concatenate([box, probs, sc.coeffs], axis=-1).reshape(h * w, -1))
    return np.concatenate(columns, axis=0).T[None].astype(np.float32)


def nchw_to_hwc(t: np.ndarray) -> np.ndarray:
    """(1, C, H, W) or (C, H, W) -> (H, W, C)."""
    t = t[0] if t.ndim == 4 else t
    return np.transpose(t, (1, 2, 0))


def group_outputs(
    tensors: dict[str, np.ndarray], in_hw: tuple[int, int], nc: int, nm: int = 32
) -> tuple[list[ScaleOutputs], np.ndarray]:
    """Sort a device's named HWC outputs into per-scale heads and the prototypes, by shape:
    channels tell box / class / coefficients apart, spatial size gives the stride, and the
    prototypes are the nm-channel tensor at 1/4 of the input."""
    in_h, _ = in_hw
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    proto = None
    for t in tensors.values():
        t = t[0] if t.ndim == 4 else t
        h, _, c = t.shape
        stride = in_h // h
        if c == nm and stride == 4:
            proto = t
            continue
        kind = {4 * REG_MAX: "box", nc: "cls", nm: "coeffs"}.get(c)
        if kind is None or stride not in STRIDES:
            raise ValueError(f"unexpected output of shape {t.shape}")
        by_stride.setdefault(stride, {})[kind] = t
    complete = sorted(s for s, kinds in by_stride.items() if len(kinds) == 3)
    if proto is None or complete != list(STRIDES):
        raise ValueError(f"need box, cls and coeffs at strides {STRIDES} plus prototypes")
    scales = [ScaleOutputs(s, **by_stride[s]) for s in STRIDES]
    return scales, proto
