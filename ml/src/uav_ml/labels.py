"""Read YOLO-seg polygon labels (Roboflow export) as per-instance masks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class LabelledInstance:
    class_id: int
    mask: np.ndarray  # bool, image resolution


def read_label_file(path: Path, width: int, height: int) -> list[LabelledInstance]:
    """One line per instance: `class x1 y1 x2 y2 ...`, coordinates normalised to [0, 1]."""
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least three points
            continue
        xy = np.array(parts[1:], float).reshape(-1, 2) * (width, height)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(xy).astype(np.int32)], 1)
        out.append(LabelledInstance(int(parts[0]), mask.astype(bool)))
    return out


def iter_dataset(root: Path) -> Iterator[tuple[Path, np.ndarray, list[LabelledInstance]]]:
    """(image path, BGR image, instances) for every image under root/**/images with a label
    file of the same stem under the sibling labels/ folder."""
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES or img_path.parent.name != "images":
            continue
        label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not label.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        yield img_path, img, read_label_file(label, img.shape[1], img.shape[0])
