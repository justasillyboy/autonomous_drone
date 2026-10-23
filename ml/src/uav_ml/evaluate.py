"""uav-ml-eval: evaluate a model on one subset of the frozen split, with 95 % bootstrap CIs.

    uv run uav-ml-eval --weights model.onnx --dataset <unzipped dataset-real-286> \
        --split ml/configs/splits/real-286.json --subset test --out results/eval.json

Reports mask AP50 and AP50-95 per class and overall, and on matched instances (IoU >= 0.5)
Boundary F1 at 2 px, the 95 % Hausdorff distance and, for circles, the predicted mask's
circularity plus centre and radius errors against the labelled circle (the radius error
becomes the vision-altitude error: Z = f * D / (2 r)).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from uav_core.types import TargetClass
from uav_edge.runtime import Backend, load_backend

from uav_ml.labels import read_label_file
from uav_ml.metrics import (
    IOU_THRESHOLDS,
    Gt,
    Image,
    Pred,
    average_precision,
    bootstrap_ci,
    boundary_f1,
    circularity,
    evaluate_image,
    hausdorff95,
    match,
    mean_ap,
)
from uav_ml.splits import load_split, resolve

DATASET_CLASSES = ("circle", "semicircle", "H_landing")  # data.yaml order of the real set


def predict(backend: Backend, frame: np.ndarray) -> tuple[Pred, ...]:
    h, w = frame.shape[:2]
    out = []
    for inst in backend.infer(frame):
        full = cv2.resize(inst.mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(Pred(inst.cls.value, inst.conf, full > 0.5))
    return tuple(out)


def ground_truth(label: Path, w: int, h: int) -> tuple[Gt, ...]:
    return tuple(Gt(DATASET_CLASSES[i.class_id], i.mask) for i in read_label_file(label, w, h))


def _circle(mask: np.ndarray) -> tuple[float, float, float] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return cx, cy, r


def image_stats(im: Image) -> dict[str, list[float]]:
    """Per-match quality on IoU >= 0.5 matches of one image."""
    stats: dict[str, list[float]] = {
        "boundary_f1": [], "hausdorff95_px": [], "circularity": [],
        "centre_err_px": [], "radius_rel_err": [],
    }  # fmt: skip
    for pi, gi, _ in match(im.preds, im.gts, 0.5):
        p, g = im.preds[pi], im.gts[gi]
        stats["boundary_f1"].append(boundary_f1(p.mask, g.mask))
        stats["hausdorff95_px"].append(hausdorff95(p.mask, g.mask))
        if p.cls == TargetClass.CIRCLE.value:
            stats["circularity"].append(circularity(p.mask))
            pc, gc = _circle(p.mask), _circle(g.mask)
            if pc and gc and gc[2] > 0:
                stats["centre_err_px"].append(float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])))
                stats["radius_rel_err"].append((pc[2] - gc[2]) / gc[2])
    return stats


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _pooled(per_image: Sequence[dict[str, list[float]]], key: str) -> list[float]:
    return [v for s in per_image for v in s[key]]


def summarise(images: Sequence[Image], n_boot: int = 1000) -> dict:
    classes = [c for c in DATASET_CLASSES if any(g.cls == c for im in images for g in im.gts)]
    evals = [evaluate_image(im, classes) for im in images]
    report: dict = {"images": len(images), "instances": sum(len(im.gts) for im in images)}
    for name, thresholds in (("mask_ap50", (0.5,)), ("mask_ap50_95", IOU_THRESHOLDS)):
        stat = lambda ev, th=thresholds: mean_ap(ev, classes, th)  # noqa: E731
        value, lo, hi = bootstrap_ci(evals, stat, n_boot)
        report[name] = {"value": round(value, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    report["per_class_ap50"] = {c: average_precision(evals, c, 0.5) for c in classes}
    per_image = [image_stats(im) for im in images]
    for key in per_image[0] if per_image else ():
        values = _pooled(per_image, key)
        report[key] = {"mean": _mean(values), "n": len(values)}
        if key in ("boundary_f1", "radius_rel_err") and values:
            stat = lambda recs, key=key: _mean(_pooled(recs, key))  # noqa: E731
            _, lo, hi = bootstrap_ci(per_image, stat, n_boot)
            report[key]["ci95"] = [lo, hi]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-ml-eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=("train", "val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    backend = load_backend(args.weights, conf=args.conf)
    images = []
    for img_path, label in resolve(load_split(args.split), args.dataset, args.subset):
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        images.append(Image(predict(backend, frame), ground_truth(label, w, h)))
    report = {"weights": args.weights.name, "subset": args.subset, "conf": args.conf}
    report |= summarise(images, args.bootstrap)
    text = json.dumps(report, indent=2, default=float)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
