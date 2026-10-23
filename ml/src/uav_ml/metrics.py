"""Segmentation metrics for the frozen real test set, with bootstrap confidence intervals.

Everything works on full-resolution boolean masks. An image is (predictions, ground truth):
predictions are (class, score, mask), ground truth is (class, mask).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class Pred:
    cls: str
    score: float
    mask: np.ndarray


@dataclass(frozen=True)
class Gt:
    cls: str
    mask: np.ndarray


@dataclass(frozen=True)
class Image:
    preds: tuple[Pred, ...]
    gts: tuple[Gt, ...]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def _distance_to(boundary: np.ndarray) -> np.ndarray:
    """Distance of every pixel to the nearest boundary pixel."""
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol_px: float = 2.0) -> float:
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() or not bg.any():
        return 0.0
    precision = float((_distance_to(bg)[bp] <= tol_px).mean())
    recall = float((_distance_to(bp)[bg] <= tol_px).mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance between the two boundaries, in pixels."""
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() or not bg.any():
        return float("inf")
    return float(
        max(np.percentile(_distance_to(bg)[bp], 95), np.percentile(_distance_to(bp)[bg], 95))
    )


def circularity(mask: np.ndarray) -> float:
    """4 pi A / P^2 of the largest blob, capped at 1 (a disc).

    A is the pixel count and P the length of the contour simplified to 1 px. The legacy
    evaluate.py used the raw staircase contour for both, which scores a perfect digital disc
    of radius 30 px at 0.88 (so its >= 0.90 target was out of reach) and a 2:1 ellipse at 0.76
    instead of 0.84; this estimator gives 1.0 and 0.85.
    """
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    blob = np.zeros_like(m)
    cv2.drawContours(blob, [c], -1, 1, thickness=-1)
    perimeter = cv2.arcLength(cv2.approxPolyDP(c, 1.0, closed=True), closed=True)
    if perimeter == 0:
        return 0.0
    return float(min(1.0, 4 * np.pi * blob.sum() / perimeter**2))


def _iou_matrix(preds: Sequence[Pred], gts: Sequence[Gt]) -> np.ndarray:
    return np.array([[mask_iou(p.mask, g.mask) for g in gts] for p in preds]).reshape(
        len(preds), len(gts)
    )


def _greedy(
    scores: Sequence[float], ious: np.ndarray, iou_thr: float
) -> list[tuple[int, int, float]]:
    """Greedy matching, best score first, on a precomputed IoU matrix (preds x gts)."""
    taken: set[int] = set()
    out = []
    for pi in sorted(range(len(scores)), key=lambda i: -scores[i]):
        best, best_iou = None, iou_thr
        for gi in range(ious.shape[1]):
            if gi not in taken and ious[pi, gi] >= best_iou:
                best, best_iou = gi, ious[pi, gi]
        if best is not None:
            taken.add(best)
            out.append((pi, best, float(best_iou)))
    return out


def match(preds: Sequence[Pred], gts: Sequence[Gt], iou_thr: float) -> list[tuple[int, int, float]]:
    """Greedy same-class matching, best score first: (pred index, gt index, IoU)."""
    out = []
    for cls in {p.cls for p in preds}:
        pi = [i for i, p in enumerate(preds) if p.cls == cls]
        gi = [i for i, g in enumerate(gts) if g.cls == cls]
        ious = _iou_matrix([preds[i] for i in pi], [gts[i] for i in gi])
        out += [
            (pi[a], gi[b], iou)
            for a, b, iou in _greedy([preds[i].score for i in pi], ious, iou_thr)
        ]
    return out


@dataclass(frozen=True)
class ImageEval:
    """One image's matching, computed once so the bootstrap only aggregates."""

    scored: dict[tuple[str, float], tuple[tuple[float, bool], ...]]  # (class, IoU thr) -> hits
    n_gt: dict[str, int]


def evaluate_image(im: Image, classes: Sequence[str], thresholds=IOU_THRESHOLDS) -> ImageEval:
    scored: dict[tuple[str, float], tuple[tuple[float, bool], ...]] = {}
    for cls in classes:
        preds = [p for p in im.preds if p.cls == cls]
        gts = [g for g in im.gts if g.cls == cls]
        ious = _iou_matrix(preds, gts)
        scores = [p.score for p in preds]
        for thr in thresholds:
            hit = {pi for pi, _, _ in _greedy(scores, ious, thr)}
            scored[(cls, float(thr))] = tuple((sc, i in hit) for i, sc in enumerate(scores))
    return ImageEval(scored, {c: sum(g.cls == c for g in im.gts) for c in classes})


def average_precision(evals: Sequence[ImageEval], cls: str, iou_thr: float) -> float | None:
    """COCO-style 101-point AP for one class; None when the class has no ground truth."""
    n_gt = sum(e.n_gt.get(cls, 0) for e in evals)
    if n_gt == 0:
        return None
    scored = [s for e in evals for s in e.scored.get((cls, float(iou_thr)), ())]
    if not scored:
        return 0.0
    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([hit for _, hit in scored])
    precision = tp / np.arange(1, len(scored) + 1)
    recall = tp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))


def mean_ap(evals: Sequence[ImageEval], classes: Sequence[str], thresholds=IOU_THRESHOLDS) -> float:
    """Mean over classes (with ground truth) and IoU thresholds."""
    values = [average_precision(evals, c, t) for c in classes for t in thresholds]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(
    images: Sequence,
    stat: Callable[[Sequence], float],
    n: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(statistic, low, high): percentile bootstrap over per-image items (ImageEval or any
    precomputed per-image record)."""
    rng = np.random.default_rng(seed)
    k = len(images)
    samples = [stat([images[i] for i in rng.integers(0, k, k)]) for _ in range(n)]
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return stat(images), float(lo), float(hi)
