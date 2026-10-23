"""The interface every backend implements, and the model-class mapping."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

import numpy as np
from uav_core.perception import RawInstance
from uav_core.types import TargetClass


class Backend(Protocol):
    """frame (BGR, full resolution) -> instances in frame pixels, masks at any resolution
    whose aspect ratio matches the frame (uav_core scales them)."""

    def infer(self, frame_bgr: np.ndarray) -> list[RawInstance]: ...


def class_map(names: Mapping[int, str] | str) -> dict[int, TargetClass]:
    """Model class id -> TargetClass, matched by name, never by position.

    `names` is the model's own table (Ultralytics `model.names`, or the `names` metadata
    string of an exported ONNX). A model trained with a different class order still maps
    correctly; an unknown class name is an error rather than a silent mislabel.
    """
    table = ast.literal_eval(names) if isinstance(names, str) else dict(names)
    known = {c.value: c for c in TargetClass}
    unknown = sorted(set(table.values()) - set(known))
    if unknown:
        raise ValueError(f"model has classes uav_core does not know: {unknown}")
    return {int(idx): known[name] for idx, name in table.items()}
