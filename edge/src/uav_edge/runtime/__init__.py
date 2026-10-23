"""Backends share one interface: `infer(frame_bgr) -> list[RawInstance]`."""

from __future__ import annotations

from pathlib import Path

from uav_edge.runtime.base import Backend, class_map
from uav_edge.runtime.letterbox import Letterbox

__all__ = ["Backend", "Letterbox", "class_map", "load_backend"]


def load_backend(weights: str | Path, conf: float = 0.5) -> Backend:
    """Pick the backend from the file type: .onnx -> ONNX Runtime, .pt -> Ultralytics."""
    suffix = Path(weights).suffix.lower()
    if suffix == ".onnx":
        from uav_edge.runtime.onnx import OnnxBackend

        return OnnxBackend(weights, conf=conf)
    if suffix == ".pt":
        from uav_edge.runtime.ultralytics import UltralyticsBackend

        return UltralyticsBackend(weights, conf=conf)
    raise ValueError(f"no backend for {weights} (expected .onnx or .pt; .hef comes with M4)")
