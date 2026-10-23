"""uav-ml-train: train one experiment config on every seed, then evaluate and export.

    uv run --with ultralytics uav-ml-train ml/configs/train/e1_real_50.yaml \
        --dataset <unzipped dataset-real-286> --runs runs/

For each seed: build the YOLO folder for the config's train_fraction, train with Ultralytics,
evaluate best.pt on the frozen test split with uav-ml-eval (bootstrap CIs) and export ONNX.
Results go to <runs>/<experiment>/seed<k>/ and a summary CSV per experiment. Needs a GPU in
practice (Colab / Kaggle, see ml/notebooks/train_colab.ipynb); Ultralytics is imported only
here so the rest of uav_ml stays light.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from uav_ml.build_dataset import build

SPLIT = Path(__file__).resolve().parents[3] / "ml" / "configs" / "splits" / "real-286.json"
NOT_ULTRALYTICS = ("inherit", "train_fraction", "seeds")


def load_config(path: Path) -> dict:
    """Read an experiment YAML, following `inherit:` chains (child keys win)."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    parent = cfg.get("inherit")
    if parent:
        base = load_config(Path(path).parent / parent)
        cfg = {**base, **{k: v for k, v in cfg.items() if k != "inherit"}}
    if not 0.0 < float(cfg.get("train_fraction", 1.0)) <= 1.0:
        raise ValueError(f"{path}: train_fraction must be in (0, 1]")
    return cfg


def ultralytics_args(cfg: dict, data_yaml: Path, run_dir: Path, seed: int) -> dict:
    args = {k: v for k, v in cfg.items() if k not in NOT_ULTRALYTICS and k != "model"}
    return args | {
        "data": str(data_yaml),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "seed": seed,
        "deterministic": True,
        "exist_ok": True,
    }


def run(
    config: Path, dataset: Path, runs: Path, split: Path = SPLIT
) -> list[dict]:  # pragma: no cover
    from ultralytics import YOLO  # optional, heavy

    from uav_ml import evaluate

    cfg = load_config(config)
    name = config.stem
    data = runs / name / "data"
    build(dataset, split, data, float(cfg.get("train_fraction", 1.0)))
    rows = []
    for seed in cfg.get("seeds", [0]):
        run_dir = runs / name / f"seed{seed}"
        model = YOLO(cfg["model"])
        model.train(**ultralytics_args(cfg, data / "data.yaml", run_dir, seed))
        best = run_dir / "weights" / "best.pt"
        report = run_dir / "eval_test.json"
        evaluate.main(["--weights", str(best), "--dataset", str(dataset), "--split", str(split),
                       "--out", str(report)])  # fmt: skip
        YOLO(str(best)).export(format="onnx", imgsz=cfg["imgsz"], opset=17, simplify=True)
        r = json.loads(report.read_text())
        rows.append(
            {
                "experiment": name,
                "seed": seed,
                "model": cfg["model"],
                "train_fraction": cfg.get("train_fraction", 1.0),
                "mask_ap50": r["mask_ap50"]["value"],
                "mask_ap50_95": r["mask_ap50_95"]["value"],
                "boundary_f1": r["boundary_f1"]["mean"],
            }
        )
    with (runs / name / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="uav-ml-train", description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--split", type=Path, default=SPLIT)
    args = parser.parse_args(argv)
    for row in run(args.config, args.dataset, args.runs, args.split):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
