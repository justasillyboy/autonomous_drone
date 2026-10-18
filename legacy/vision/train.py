"""
train.py — Training entry point for YOLO11-seg circle/semicircle/H_landing.

Usage:
    python train.py                          # uses config.yaml defaults
    python train.py --config config.yaml
    python train.py --resume runs/.../last.pt
    python train.py --boundary-loss          # enable auxiliary loss (fixed)
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config loader with safe defaults
# ---------------------------------------------------------------------------

# Every key the training loop may access, with a safe fallback.
# This prevents KeyError crashes when config.yaml is missing a key.
CONFIG_DEFAULTS = {
    "data":          "data/circle_seg.yaml",
    "imgsz":         640,
    "model":         "yolo11n-seg.pt",   # Nano — only size that fits Hailo-8L (13 TOPS)
    "mask_ratio":    4,    # 4 = Hailo-8L recommended; PCIe bus saturates at 2 on Pi5.
    "overlap_mask":  True,
    "epochs":        150,
    "batch":         16,
    "lr0":           0.01,
    "lrf":           0.01,
    "momentum":      0.937,
    "weight_decay":  0.0005,
    "warmup_epochs": 3,
    "close_mosaic":  15,
    "amp":           True,
    "val":           True,
    "save_period":   10,
    "project":       "runs/circle_seg",
    "name":          "yolo11_v1",
    "device":        0,
    "workers":       8,
    "degrees":       180.0,
    "scale":         0.5,
    "fliplr":        0.5,
    "flipud":        0.3,
    "mosaic":        1.0,
    "copy_paste":    0.3,
    "hsv_h":         0.015,
    "hsv_s":         0.7,
    "hsv_v":         0.4,
    "lambda_boundary": 0.4,
    "lambda_dice":     0.3,
}


def load_config(path: str) -> dict:
    """Load YAML config and merge with defaults so no key is ever missing."""
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    # Defaults first, user values override
    cfg = {**CONFIG_DEFAULTS, **user_cfg}
    return cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11-seg")
    parser.add_argument("--config",         default="config.yaml")
    parser.add_argument("--resume",         default=None,
                        help="Checkpoint .pt to resume from")
    parser.add_argument("--boundary-loss",  action="store_true",
                        help="Enable auxiliary boundary+dice loss during training")
    parser.add_argument("--device",         default=None,
                        help="Override device: '0', 'cpu', '0,1'")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Boundary-loss callback  (FIX: hooked BEFORE training, not after)
# ---------------------------------------------------------------------------

def _make_loss_callback(lambda_boundary: float, lambda_dice: float):
    """
    Return an on_train_start callback that patches the loss BEFORE the first
    epoch runs.

    PREVIOUS BUG: patch_ultralytics_loss() was called after model.train()
    returned, meaning 150 epochs completed with default loss and the patch
    was applied to a finished trainer that would never run again.

    FIX: Ultralytics fires 'on_train_start' before the first batch. We hook
    into that event, which guarantees every epoch uses the patched loss.
    """
    def on_train_start(trainer):
        try:
            from loss import circle_seg_loss
            import torch

            original_loss_fn = trainer.model.criterion

            def patched_criterion(preds, batch):
                native_loss, native_items = original_loss_fn(preds, batch)

                # Extract mask tensors — Ultralytics packs them differently
                # across versions; guard with hasattr and shape checks.
                gt_masks   = getattr(batch, "masks", None)
                pred_masks = None
                if isinstance(preds, (tuple, list)) and len(preds) > 1:
                    proto_output = preds[1]
                    if isinstance(proto_output, (tuple, list)) and len(proto_output) > 0:
                        pred_masks = proto_output[0].sigmoid()

                if (gt_masks is not None and pred_masks is not None
                        and pred_masks.shape == gt_masks.float().shape):
                    aux_loss, components = circle_seg_loss(
                        pred_masks, gt_masks.float(),
                        lambda_boundary=lambda_boundary,
                        lambda_dice=lambda_dice,
                    )
                    # Scale auxiliary loss to 10% so it refines without
                    # destabilising the native YOLO loss balance.
                    native_loss = native_loss + 0.10 * aux_loss

                    # Log each component to the trainer's CSV
                    if hasattr(trainer, "tloss") and trainer.tloss is not None:
                        for k, v in components.items():
                            trainer.label_loss_items(
                                {k: torch.tensor(v)}, prefix="aux"
                            )

                return native_loss, native_items

            trainer.model.criterion = patched_criterion
            print("[Loss] Boundary+Dice auxiliary loss active from epoch 1.")

        except Exception as exc:
            print(f"[Loss] WARNING: could not apply auxiliary loss — {exc}")
            print("[Loss] Training continues with default YOLO loss.")

    return on_train_start


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None,
          use_boundary_loss: bool = False) -> None:

    # ── Model init ────────────────────────────────────────────────────────────
    weights = resume if resume else cfg["model"]
    print(f"Loading weights: {weights}")
    model = YOLO(weights)

    # ── Register loss callback BEFORE training starts  (FIX) ─────────────────
    if use_boundary_loss:
        callback = _make_loss_callback(
            lambda_boundary=cfg["lambda_boundary"],
            lambda_dice=cfg["lambda_dice"],
        )
        model.add_callback("on_train_start", callback)
        print("[Loss] Boundary+Dice loss callback registered.")

    # ── Build kwargs — every value has a safe default (FIX) ──────────────────
    train_kwargs = {
        "data":          cfg["data"],
        "epochs":        cfg["epochs"],
        "imgsz":         cfg["imgsz"],
        "batch":         cfg["batch"],
        "lr0":           cfg["lr0"],
        "lrf":           cfg["lrf"],
        "momentum":      cfg["momentum"],
        "weight_decay":  cfg["weight_decay"],
        "warmup_epochs": cfg["warmup_epochs"],
        "close_mosaic":  cfg["close_mosaic"],
        "amp":           cfg["amp"],
        "overlap_mask":  cfg["overlap_mask"],
        "mask_ratio":    cfg["mask_ratio"],
        "val":           cfg["val"],
        "save_period":   cfg["save_period"],
        "project":       cfg["project"],
        "name":          cfg["name"],
        "device":        cfg["device"],
        "workers":       cfg["workers"],
        "degrees":       cfg["degrees"],
        "scale":         cfg["scale"],
        "fliplr":        cfg["fliplr"],
        "flipud":        cfg["flipud"],
        "mosaic":        cfg["mosaic"],
        "copy_paste":    cfg["copy_paste"],
        "hsv_h":         cfg["hsv_h"],
        "hsv_s":         cfg["hsv_s"],
        "hsv_v":         cfg["hsv_v"],
        "resume":        bool(resume),
    }

    print("\n── Training config ──────────────────────────────────────────────")
    for k, v in train_kwargs.items():
        print(f"  {k:<20}: {v}")
    print("─────────────────────────────────────────────────────────────────\n")

    results = model.train(**train_kwargs)

    # ── Summary ───────────────────────────────────────────────────────────────
    best = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best}")
    print(f"\nNext steps:")
    print(f"  Evaluate → python evaluate.py --weights {best} --config config.yaml")
    print(f"  Export   → python export.py   --weights {best} --format onnx --validate")
    return results


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------

def finetune(weights: str, data_yaml: str, epochs: int = 30,
             freeze_layers: int = 10, device: int | str = 0) -> None:
    """Fast domain adaptation: freeze backbone, train neck+head only."""
    model = YOLO(weights)
    model.train(
        data=data_yaml, epochs=epochs, freeze=freeze_layers,
        lr0=0.001, lrf=0.01, batch=8, imgsz=640, amp=True,
        device=device, name="finetune_domain",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    train(cfg, resume=args.resume, use_boundary_loss=args.boundary_loss)
