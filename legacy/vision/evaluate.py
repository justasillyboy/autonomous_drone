"""
evaluate.py — Evaluation metrics for circular instance segmentation.

Goes beyond standard mAP to measure what matters for circles:
  - Boundary F1 with pixel tolerance
  - Hausdorff distance (worst-case edge error)
  - Circularity score (geometric quality of predicted shapes)
  - Per-class breakdown (circle vs semi-circle)

Usage:
    python evaluate.py --weights best.pt --data circle_seg.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Pixel-level boundary metrics
# ---------------------------------------------------------------------------

def boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: int = 2,
) -> dict[str, float]:
    """
    Compute boundary precision, recall, and F1 with a pixel tolerance.

    A predicted boundary pixel counts as TP if it falls within `tolerance_px`
    of any GT boundary pixel (via dilation). This forgives sub-pixel jitter
    without hiding genuinely poor circular edge fitting.

    Args:
        pred_mask    : Binary uint8 mask (H, W).
        gt_mask      : Binary uint8 mask (H, W).
        tolerance_px : Dilation radius in pixels (default 2).

    Returns:
        Dict with keys: boundary_f1, boundary_precision, boundary_recall.
    """
    pred_boundary = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
    gt_boundary   = cv2.Canny((gt_mask   * 255).astype(np.uint8), 50, 150)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    gt_dilated   = cv2.dilate(gt_boundary, kernel)
    pred_dilated = cv2.dilate(pred_boundary, kernel)

    tp_pred = np.logical_and(pred_boundary > 0, gt_dilated > 0).sum()
    tp_gt   = np.logical_and(gt_boundary   > 0, pred_dilated > 0).sum()

    precision = float(tp_pred) / (pred_boundary.sum() + 1e-6)
    recall    = float(tp_gt)   / (gt_boundary.sum()   + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "boundary_f1":        round(f1, 4),
        "boundary_precision": round(precision, 4),
        "boundary_recall":    round(recall, 4),
    }


def hausdorff_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Symmetric Hausdorff distance between predicted and GT boundaries.

    Measures the worst-case edge error in pixels — the maximum distance
    from any predicted boundary pixel to the nearest GT boundary pixel
    (and vice versa). A circle with one flat section will have high HD.

    Returns float('inf') if either mask has no boundary pixels.
    """
    pred_edge = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
    gt_edge   = cv2.Canny((gt_mask   * 255).astype(np.uint8), 50, 150)

    pred_pts = np.argwhere(pred_edge > 0)
    gt_pts   = np.argwhere(gt_edge   > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return round(float(max(d1, d2)), 3)


def _load_gt_masks_for_result(r, iou_threshold: float = 0.5) -> list[tuple[np.ndarray, int]]:
    """
    Load ground-truth masks from YOLO label files for a given result object.

    For each GT instance, we rasterize the polygon from the .txt label file
    and then greedily match it to the predicted mask with the highest IoU.

    Returns list of (gt_mask_uint8, matched_pred_idx) where matched_pred_idx
    is -1 if no prediction was matched above iou_threshold.
    """
    import os
    orig_path = r.path if hasattr(r, "path") else None
    if orig_path is None:
        return []

    # Derive label file path: images/train/xxx.jpg → labels/train/xxx.txt
    orig_path = str(orig_path)
    label_path = orig_path.replace("/images/", "/labels/").replace(
        ".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")

    if not os.path.isfile(label_path):
        return []

    h, w = r.orig_shape
    pairs = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(parts[0])
            # YOLO seg polygon: alternating normalised x,y pairs
            coords = [float(v) for v in parts[1:]]
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32)

            gt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [pts], 1)

            # Match to best overlapping prediction
            best_idx, best_iou = -1, iou_threshold
            if r.masks is not None:
                pred_masks_np = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
                for pi, pm in enumerate(pred_masks_np):
                    # Resize pred mask from model output size to original frame size
                    pm_resized = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                    iou = mask_iou(pm_resized, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = pi

            pairs.append((gt_mask, cls_id, best_idx))

    return pairs
    """Standard intersection-over-union for a single mask pair."""
    pred_b = pred_mask > 0
    gt_b   = gt_mask   > 0
    inter  = np.logical_and(pred_b, gt_b).sum()
    union  = np.logical_or(pred_b, gt_b).sum()
    return float(inter) / (float(union) + 1e-6)


def circularity_score(pred_mask: np.ndarray) -> float:
    """
    Measure how circular the predicted mask is, independent of GT.

    Uses the isoperimetric quotient: 4π·Area / Perimeter²
    Perfect circle → 1.0. Elongated shapes → lower values.
    Useful for catching degenerate predictions (spiky masks, rectangles).
    """
    contours, _ = cv2.findContours(
        (pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0
    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, closed=True)
    if peri < 1e-6:
        return 0.0
    return float(4 * np.pi * area / (peri ** 2))


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    weights: str,
    data_yaml: str,
    split: str = "val",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    tolerance_px: int = 2,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Run the full evaluation suite on a validation split.

    Reports:
      - Ultralytics native mask mAP50 / mAP50-95
      - Mean boundary F1 (at tolerance_px)
      - Mean Hausdorff distance
      - Mean circularity score
      - Per-class breakdown

    Args:
        weights        : Path to trained .pt weights.
        data_yaml      : Dataset YAML path.
        split          : 'val' or 'test'.
        conf_threshold : Detection confidence cutoff.
        iou_threshold  : IoU threshold for TP/FP.
        tolerance_px   : Pixel tolerance for boundary F1.
        verbose        : Print results table to stdout.

    Returns:
        Dict of all computed metrics.
    """
    model = YOLO(weights)

    # ── Native Ultralytics val ───────────────────────────────────────────────
    native = model.val(data=data_yaml, split=split, conf=conf_threshold, iou=iou_threshold)
    mask_map50    = float(native.seg.map50)
    mask_map5095  = float(native.seg.map)

    # ── Custom boundary metrics ──────────────────────────────────────────────
    results = model.predict(
        source=data_yaml,
        conf=conf_threshold,
        stream=True,
        verbose=False,
    )

    all_bnd_f1, all_hd, all_circ, all_iou = [], [], [], []
    per_class: dict[int, dict[str, list]] = {}

    for r in results:
        if r.masks is None:
            continue

        pred_masks_np = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
        h_orig, w_orig = r.orig_shape

        # Load GT masks from label files and match to predictions
        gt_pairs = _load_gt_masks_for_result(r, iou_threshold=iou_threshold)

        if not gt_pairs:
            # No GT available — skip this image for boundary metrics
            continue

        for gt_mask, cls_id, pred_idx in gt_pairs:
            if cls_id not in per_class:
                per_class[cls_id] = {"bnd_f1": [], "hd": [], "circ": [], "iou": []}

            if pred_idx >= 0:
                # Resize pred mask to original resolution for fair comparison
                pred = cv2.resize(pred_masks_np[pred_idx], (w_orig, h_orig),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                # False negative — no matching prediction; treat as all-zero mask
                pred = np.zeros((h_orig, w_orig), dtype=np.uint8)

            bnd  = boundary_f1(pred, gt_mask, tolerance_px=tolerance_px)
            hd   = hausdorff_distance(pred, gt_mask)
            circ = circularity_score(pred)
            iou  = mask_iou(pred, gt_mask)

            all_bnd_f1.append(bnd["boundary_f1"])
            all_hd.append(hd if hd != float("inf") else 0.0)
            all_circ.append(circ)
            all_iou.append(iou)

            per_class[cls_id]["bnd_f1"].append(bnd["boundary_f1"])
            per_class[cls_id]["hd"].append(hd if hd != float("inf") else 0.0)
            per_class[cls_id]["circ"].append(circ)
            per_class[cls_id]["iou"].append(iou)

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    summary = {
        "mask_mAP50":      round(mask_map50, 4),
        "mask_mAP50-95":   round(mask_map5095, 4),
        "mean_boundary_f1": safe_mean(all_bnd_f1),
        "mean_hausdorff":   safe_mean(all_hd),
        "mean_circularity": safe_mean(all_circ),
        "mean_iou":         safe_mean(all_iou),
    }

    class_names = {0: "circle", 1: "semicircle", 2: "H_landing"}
    for cls_id, vals in per_class.items():
        cname = class_names.get(cls_id, f"class_{cls_id}")
        summary[f"{cname}_boundary_f1"] = safe_mean(vals["bnd_f1"])
        summary[f"{cname}_hausdorff"]   = safe_mean(vals["hd"])
        summary[f"{cname}_circularity"] = safe_mean(vals["circ"])

    if verbose:
        _print_results(summary)

    return summary


def _print_results(summary: dict) -> None:
    print("\n" + "═" * 52)
    print("  Evaluation Results")
    print("═" * 52)

    sections = [
        ("YOLO native",    ["mask_mAP50", "mask_mAP50-95"]),
        ("Edge quality",   ["mean_boundary_f1", "mean_hausdorff", "mean_circularity"]),
        ("Circle class",   ["circle_boundary_f1", "circle_hausdorff", "circle_circularity"]),
        ("Semi-circle",    ["semicircle_boundary_f1", "semicircle_hausdorff", "semicircle_circularity"]),
        ("H_landing",      ["H_landing_boundary_f1", "H_landing_hausdorff", "H_landing_circularity"]),
    ]
    for section, keys in sections:
        print(f"\n  {section}")
        for k in keys:
            if k in summary:
                print(f"    {k:<30} {summary[k]:.4f}")

    print("\n" + "═" * 52)
    _print_targets(summary)


def _print_targets(summary: dict) -> None:
    """Print pass/fail against research targets."""
    targets = {
        "mask_mAP50-95":    (0.75, ">="),
        "mean_boundary_f1": (0.85, ">="),
        "mean_hausdorff":   (3.0,  "<="),
        "mean_circularity": (0.90, ">="),
    }
    print("  Target check")
    for metric, (target, op) in targets.items():
        val = summary.get(metric, None)
        if val is None:
            continue
        if op == ">=" :
            passed = val >= target
        else:
            passed = val <= target
        icon = "✓" if passed else "✗"
        print(f"    {icon}  {metric:<30} {val:.4f}  (target {op} {target})")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO11-seg circle segmentation")
    parser.add_argument("--weights",   required=True, help="Path to .pt weights")
    parser.add_argument("--data",      required=True, help="Path to dataset YAML")
    parser.add_argument("--split",     default="val", choices=["val", "test"])
    parser.add_argument("--conf",      type=float, default=None,
                        help="Confidence threshold — defaults to value in --config")
    parser.add_argument("--tolerance", type=int,   default=2,
                        help="Boundary F1 pixel tolerance (default 2)")
    parser.add_argument("--config",    default=None,
                        help="Path to config.yaml — syncs conf threshold with training")
    args = parser.parse_args()

    # Sync confidence threshold from training config so metrics are not misleading
    conf = args.conf
    if conf is None and args.config is not None:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f) or {}
            conf = float(cfg.get("conf", cfg.get("conf_threshold", 0.5)))
            print(f"[Evaluate] Using conf={conf:.2f} from {args.config}")
        except Exception as e:
            print(f"[Evaluate] Could not read config ({e}), defaulting conf=0.5")
            conf = 0.5
    elif conf is None:
        conf = 0.5

    metrics = evaluate_model(
        weights=args.weights,
        data_yaml=args.data,
        split=args.split,
        conf_threshold=conf,
        tolerance_px=args.tolerance,
    )
