"""uav-ml-build: lay out a YOLO-seg training folder from the frozen split.

    uv run uav-ml-build --dataset <unzipped dataset-real-286> \
        --split ml/configs/splits/real-286.json --out data/real-286 [--train-fraction 0.5]

Writes images/{train,val,test}, labels/{train,val,test} (hard links when possible) and a
data.yaml with the class names. --train-fraction keeps a contiguous-block subset of the
training frames for the label-budget experiment E1 (25 / 50 / 100 %), always the same frames
for a given fraction so runs are comparable.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml

from uav_ml.splits import SUBSETS, load_split, resolve

CLASS_NAMES = ("circle", "semicircle", "H_landing")


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:  # another filesystem
        shutil.copy2(src, dst)


def subsample(pairs: list[tuple[Path, Path]], fraction: float) -> list[tuple[Path, Path]]:
    """Every k-th frame in time order, so the subset still spans the whole recording."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    keep = max(1, round(len(pairs) * fraction))
    step = len(pairs) / keep
    return [pairs[int(i * step)] for i in range(keep)]


def build(dataset: Path, split_path: Path, out: Path, train_fraction: float = 1.0) -> dict:
    split = load_split(split_path)
    counts = {}
    for subset in SUBSETS:
        pairs = resolve(split, dataset, subset)
        if subset == "train":
            pairs = subsample(pairs, train_fraction)
        for img, label in pairs:
            _place(img, out / "images" / subset / img.name)
            _place(label, out / "labels" / subset / label.name)
        counts[subset] = len(pairs)
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-ml-build", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)
    counts = build(args.dataset, args.split, args.out, args.train_fraction)
    print(f"{counts} -> {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
