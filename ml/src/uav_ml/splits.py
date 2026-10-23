"""Frozen train/val/test split of the real frames, by time blocks with buffers.

The 300 real frames are consecutive (`real_frame_0000..0299`, one recording, no scene cuts:
the mean absolute difference between neighbours is uniform). A random split would put
near-identical neighbours in train and test, so the timeline is cut into chunks; within each
chunk the first frames train, then val, then test, separated by unused buffer frames. Test
frames therefore come from the whole recording but never sit next to a training frame.

    uv run uav-ml-split --dataset <unzipped dataset-real-286> \
        --out ml/configs/splits/real-286.json
    uv run uav-ml-split --verify ml/configs/splits/real-286.json --dataset <...>

The JSON stores each file's sha256, so a changed image or label fails verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FRAME_RE = re.compile(r"real_frame_(\d+)")
SUBSETS = ("train", "val", "test")


@dataclass(frozen=True)
class BlockPlan:
    chunk: int = 50
    train: int = 32
    val: int = 5
    test: int = 9
    buffer: int = 2

    def __post_init__(self) -> None:
        used = self.train + self.val + self.test + 2 * self.buffer
        if used != self.chunk:
            raise ValueError(f"train+val+test+2*buffer = {used}, chunk = {self.chunk}")

    def subset_of(self, frame: int) -> str | None:
        """'train' | 'val' | 'test', or None for a buffer frame."""
        k = frame % self.chunk
        edges = (
            ("train", self.train),
            (None, self.buffer),
            ("val", self.val),
            (None, self.buffer),
            ("test", self.test),
        )
        for name, width in edges:
            if k < width:
                return name
            k -= width
        raise AssertionError("unreachable: widths sum to the chunk")


def frame_index(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"{path.name}: not a real_frame_NNNN image")
    return int(m.group(1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labelled_images(dataset: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(dataset.rglob("*")):
        if img.parent.name != "images" or img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def make_split(dataset: Path, plan: BlockPlan | None = None) -> dict:
    plan = plan or BlockPlan()
    out: dict = {"plan": plan.__dict__, "subsets": {s: [] for s in SUBSETS}, "buffer": []}
    for img, label in labelled_images(dataset):
        entry = {
            "frame": frame_index(img),
            "image": img.name,
            "image_sha256": sha256(img),
            "label_sha256": sha256(label),
        }
        subset = plan.subset_of(entry["frame"])
        (out["subsets"][subset] if subset else out["buffer"]).append(entry)
    for entries in (*out["subsets"].values(), out["buffer"]):
        entries.sort(key=lambda e: e["frame"])
    out["counts"] = {s: len(v) for s, v in out["subsets"].items()} | {"buffer": len(out["buffer"])}
    return out


def load_split(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve(split: dict, dataset: Path, subset: str) -> list[tuple[Path, Path]]:
    """(image, label) paths of one subset, found by file name under the dataset."""
    by_name = {img.name: (img, lbl) for img, lbl in labelled_images(dataset)}
    missing = [e["image"] for e in split["subsets"][subset] if e["image"] not in by_name]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {subset} images missing, e.g. {missing[0]}")
    return [by_name[e["image"]] for e in split["subsets"][subset]]


def verify(split: dict, dataset: Path) -> list[str]:
    """Mismatches between the frozen split and the files on disk (empty when all match)."""
    by_name = {img.name: (img, lbl) for img, lbl in labelled_images(dataset)}
    problems = []
    for subset, entries in split["subsets"].items():
        for e in entries:
            if e["image"] not in by_name:
                problems.append(f"{subset}: {e['image']} missing")
                continue
            img, lbl = by_name[e["image"]]
            if sha256(img) != e["image_sha256"] or sha256(lbl) != e["label_sha256"]:
                problems.append(f"{subset}: {e['image']} changed")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-ml-split", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write a new frozen split")
    parser.add_argument("--verify", type=Path, help="check an existing split against the files")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify(load_split(args.verify), args.dataset)
        print("\n".join(problems) or f"{args.verify}: all files match")
        return 1 if problems else 0
    if not args.out:
        parser.error("give --out or --verify")
    split = make_split(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=1) + "\n")
    print(f"{split['counts']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
