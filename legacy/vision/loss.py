"""
loss.py — Boundary-aware loss functions for circular instance segmentation.

Three composable components:
  1. boundary_loss  — up-weights edge pixels using a Laplacian boundary map.
  2. dice_loss      — shape-overlap loss; robust for circular masks.
  3. circle_seg_loss — weighted combination of all three (BCE + boundary + Dice).

Usage (monkey-patch or subclass the Ultralytics trainer):
    from loss import circle_seg_loss
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Laplacian kernel (shared, allocated once)
# ---------------------------------------------------------------------------

_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _get_laplacian(device: torch.device) -> torch.Tensor:
    return _LAPLACIAN.to(device)


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def boundary_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    boundary_weight: float = 5.0,
) -> torch.Tensor:
    """
    BCE loss with higher weight on boundary pixels.

    Boundary pixels are detected via a Laplacian of the GT mask — this
    produces a thin 1-pixel ring along every circular edge, exactly
    where standard BCE under-penalises errors.

    Args:
        pred_mask       : Float tensor [B, H, W] in [0, 1] (sigmoid output).
        gt_mask         : Binary float tensor [B, H, W].
        boundary_weight : Additional weight multiplier on boundary pixels (default 5×).

    Returns:
        Scalar loss tensor.
    """
    lap = _get_laplacian(pred_mask.device)

    # Extract boundary as |Laplacian(gt)|, clamped to [0, 1]
    gt_4d = gt_mask.unsqueeze(1).float()
    boundary_map = F.conv2d(gt_4d, lap, padding=1).abs().clamp(0, 1).squeeze(1)

    # Pixel-wise weight: 1 everywhere, boundary_weight on boundary pixels
    weight = 1.0 + (boundary_weight - 1.0) * boundary_map

    bce = F.binary_cross_entropy(pred_mask, gt_mask, weight=weight, reduction="mean")
    return bce


def dice_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss — particularly effective for circular regions because
    it optimises the overlap ratio directly rather than per-pixel accuracy.

    Handles class imbalance naturally (useful when small circles occupy
    few pixels relative to background).

    Args:
        pred_mask : Float tensor [B, H, W] in [0, 1].
        gt_mask   : Binary float tensor [B, H, W].
        smooth    : Laplace smoothing term to avoid division by zero.

    Returns:
        Scalar loss in [0, 1].
    """
    pred_flat   = pred_mask.reshape(-1)
    target_flat = gt_mask.reshape(-1).float()

    intersection = (pred_flat * target_flat).sum()
    denominator  = pred_flat.sum() + target_flat.sum()
    dice_score   = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice_score


def focal_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss — optional replacement for standard BCE.

    Down-weights easy background pixels and focuses training on hard
    boundary examples. Useful when the dataset has many easy cases
    (large solid circles) and few hard ones (partial occlusions, semi-circles).

    Args:
        pred_mask : Float tensor [B, H, W] in [0, 1].
        gt_mask   : Binary float tensor [B, H, W].
        alpha     : Weighting factor for positive class.
        gamma     : Focusing parameter (higher = more focus on hard examples).

    Returns:
        Scalar loss tensor.
    """
    bce = F.binary_cross_entropy(pred_mask, gt_mask, reduction="none")
    pt  = torch.exp(-bce)  # probability of correct class
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def circle_seg_loss(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    lambda_boundary: float = 0.4,
    lambda_dice: float = 0.3,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined segmentation loss optimised for circular objects.

        L = L_bce + λ_b · L_boundary + λ_d · L_dice

    Args:
        pred_mask       : Float [B, H, W] sigmoid mask predictions.
        gt_mask         : Binary float [B, H, W] ground-truth masks.
        lambda_boundary : Weight for boundary component (tuned to ~0.4).
        lambda_dice     : Weight for Dice component (tuned to ~0.3).
        use_focal       : Replace base BCE with focal loss.
        focal_gamma     : Gamma for focal loss (ignored if use_focal=False).

    Returns:
        total_loss : Scalar tensor (backprop-ready).
        components : Dict with individual loss values for logging.
    """
    gt_mask = gt_mask.float()

    base   = (focal_loss(pred_mask, gt_mask, gamma=focal_gamma)
              if use_focal
              else F.binary_cross_entropy(pred_mask, gt_mask))
    bnd    = boundary_loss(pred_mask, gt_mask)
    dice   = dice_loss(pred_mask, gt_mask)

    total  = base + lambda_boundary * bnd + lambda_dice * dice

    components = {
        "loss_base":     base.item(),
        "loss_boundary": bnd.item(),
        "loss_dice":     dice.item(),
        "loss_total":    total.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# Ultralytics trainer hook (optional)
# ---------------------------------------------------------------------------

def patch_ultralytics_loss(trainer, lambda_boundary: float = 0.4, lambda_dice: float = 0.3):
    """
    Monkey-patch an Ultralytics SegmentationTrainer to use circle_seg_loss
    as an auxiliary mask refinement loss on top of the native loss.

    Usage in train.py:
        from ultralytics import YOLO
        from loss import patch_ultralytics_loss

        model = YOLO("yolo11m-seg.pt")
        model.train(...)          # sets up the trainer

        # After first batch, patch:
        patch_ultralytics_loss(model.trainer)

    Note: This adds the auxiliary loss to the existing Ultralytics mask loss
    rather than replacing it — preserving the carefully tuned native loss
    balance while boosting boundary sensitivity.
    """
    original_criterion = trainer.model.criterion

    def patched_criterion(preds, batch):
        native_loss, native_items = original_criterion(preds, batch)

        # Extract predicted and GT masks from batch for auxiliary loss
        # (shape varies by YOLO version; adjust indexing if needed)
        if hasattr(batch, "masks") and batch.masks is not None:
            gt_masks   = batch.masks.float()
            pred_masks = preds[1][0].sigmoid() if isinstance(preds, tuple) else None
            if pred_masks is not None and pred_masks.shape == gt_masks.shape:
                aux_loss, _ = circle_seg_loss(
                    pred_masks, gt_masks,
                    lambda_boundary=lambda_boundary,
                    lambda_dice=lambda_dice,
                )
                native_loss = native_loss + 0.1 * aux_loss  # 0.1 scale to not overpower

        return native_loss, native_items

    trainer.model.criterion = patched_criterion
    print("circle_seg_loss auxiliary loss patched onto trainer.")
