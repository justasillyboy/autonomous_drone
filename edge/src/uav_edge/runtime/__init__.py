"""Backends share one interface: `infer(frame_bgr) -> list[RawInstance]`."""

from __future__ import annotations

from pathlib import Path

from uav_edge.runtime.base import Backend, class_map
from uav_edge.runtime.letterbox import Letterbox

__all__ = ["Backend", "Letterbox", "class_map", "load_backend"]

HEF_CLASSES = ("circle", "semicircle", "H_landing")


def load_backend(weights: str | Path, conf: float = 0.5) -> Backend:
    """Pick the backend from the file type: .onnx -> ONNX Runtime, .pt -> Ultralytics,
    .hef -> Hailo-8L."""
    suffix = Path(weights).suffix.lower()
    if suffix == ".onnx":
        from uav_edge.runtime.onnx import OnnxBackend

        return OnnxBackend(weights, conf=conf)
    if suffix == ".pt":
        from uav_edge.runtime.ultralytics import UltralyticsBackend

        return UltralyticsBackend(weights, conf=conf)
    if suffix == ".hef":  # a HEF carries no class names: the training data's order
        from uav_edge.runtime.hailo import HailoBackend

        return HailoBackend(weights, class_map(dict(enumerate(HEF_CLASSES))), conf=conf)
    raise ValueError(f"no backend for {weights} (expected .onnx, .pt or .hef)")
