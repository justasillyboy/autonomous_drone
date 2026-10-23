import json

import cv2
import numpy as np
import pytest
from uav_core.perception import RawInstance
from uav_core.types import TargetClass
from uav_ml import evaluate, splits
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
    mask_iou,
    match,
    mean_ap,
)
from uav_ml.splits import BlockPlan, make_split, resolve, verify

# -- splits ---------------------------------------------------------------------------------


def test_block_plan_cuts_every_chunk_the_same_way():
    plan = BlockPlan()
    first = [plan.subset_of(i) for i in range(50)]
    assert first[:32] == ["train"] * 32 and first[32:34] == [None, None]
    assert first[34:39] == ["val"] * 5 and first[39:41] == [None, None]
    assert first[41:] == ["test"] * 9
    assert [plan.subset_of(i + 50) for i in range(50)] == first


def test_block_plan_must_fill_the_chunk():
    with pytest.raises(ValueError, match="chunk"):
        BlockPlan(chunk=50, train=30)


def _square_label(cls, x1, y1, x2, y2, w, h):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in pts)


def _dataset(root, frames, size=(80, 60)):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir()
    w, h = size
    for f in frames:
        name = f"real_frame_{f:04d}_jpg.rf.abc"
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.circle(img, (40, 30), 12, (0, 0, 200), -1)
        cv2.imwrite(str(root / "train" / "images" / f"{name}.png"), img)
        (root / "train" / "labels" / f"{name}.txt").write_text(
            _square_label(0, 28, 18, 52, 42, w, h)
        )
    return root


def test_make_split_verify_and_resolve(tmp_path):
    ds = _dataset(tmp_path / "ds", [0, 10, 33, 36, 45, 60, 99])
    split = make_split(ds)
    frames = {s: [e["frame"] for e in v] for s, v in split["subsets"].items()}
    assert frames == {"train": [0, 10, 60], "val": [36], "test": [45, 99]}
    assert [e["frame"] for e in split["buffer"]] == [33]
    assert verify(split, ds) == []
    assert next(p.name for p, _ in resolve(split, ds, "val")).startswith("real_frame_0036")
    label = next((ds / "train" / "labels").glob("real_frame_0045*"))
    label.write_text(label.read_text() + "\n")
    assert verify(split, ds) == ["test: real_frame_0045_jpg.rf.abc.png changed"]


def test_split_cli_round_trip(tmp_path, capsys):
    ds = _dataset(tmp_path / "ds", [0, 45])
    out = tmp_path / "split.json"
    assert splits.main(["--dataset", str(ds), "--out", str(out)]) == 0
    assert splits.main(["--dataset", str(ds), "--verify", str(out)]) == 0
    assert "all files match" in capsys.readouterr().out
    (next((ds / "train" / "images").glob("*0045*"))).unlink()
    assert splits.main(["--dataset", str(ds), "--verify", str(out)]) == 1
    with pytest.raises(FileNotFoundError):
        resolve(json.loads(out.read_text()), ds, "test")


def test_frame_index_rejects_other_names(tmp_path):
    with pytest.raises(ValueError):
        splits.frame_index(tmp_path / "foo.png")


# -- metrics --------------------------------------------------------------------------------


def _disc(cx=50, cy=50, r=20, shape=(100, 100)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def test_iou_boundary_f1_and_hausdorff():
    a, shifted = _disc(), _disc(cx=54)
    assert mask_iou(a, a) == 1.0 and 0.6 < mask_iou(a, shifted) < 0.9
    assert boundary_f1(a, a) == 1.0
    assert boundary_f1(a, shifted, tol_px=2) < 0.6 < boundary_f1(a, shifted, tol_px=4)
    assert hausdorff95(a, a) == 0.0
    assert hausdorff95(a, shifted) == pytest.approx(4.0, abs=0.6)
    empty = np.zeros_like(a)
    assert boundary_f1(empty, a) == 0.0 and hausdorff95(empty, a) == float("inf")
    assert mask_iou(empty, empty) == 0.0


def test_circularity_is_close_to_the_ideal_values():
    assert circularity(_disc(r=30, shape=(200, 200), cx=100, cy=100)) == pytest.approx(
        1.0, abs=0.01
    )
    square = np.zeros((300, 300), bool)
    square[50:250, 50:250] = True
    assert circularity(square) == pytest.approx(np.pi / 4, abs=0.01)
    ellipse = np.zeros((300, 300), np.uint8)
    cv2.ellipse(ellipse, (150, 150), (100, 50), 0, 0, 360, 1, -1)
    assert circularity(ellipse.astype(bool)) == pytest.approx(0.842, abs=0.015)
    assert circularity(np.zeros((10, 10), bool)) == 0.0


def test_matching_is_by_class_and_score():
    gt = (Gt("circle", _disc()),)
    preds = (
        Pred("circle", 0.4, _disc()),
        Pred("circle", 0.9, _disc(cx=52)),
        Pred("H_landing", 0.99, _disc()),
    )
    assert [m[0] for m in match(preds, gt, 0.5)] == [1]


def _ev(images):
    return [evaluate_image(im, ["circle", "H_landing"], IOU_THRESHOLDS) for im in images]


def test_average_precision():
    perfect = _ev([Image((Pred("circle", 0.9, _disc()),), (Gt("circle", _disc()),))] * 3)
    assert average_precision(perfect, "circle", 0.5) == pytest.approx(1.0)
    assert average_precision(perfect, "H_landing", 0.5) is None
    # one confident false positive ahead of the true positive halves the precision there
    fp = Pred("circle", 0.9, _disc(cx=20, cy=20, r=5))
    fp_first = _ev([Image((fp, Pred("circle", 0.5, _disc())), (Gt("circle", _disc()),))])
    assert average_precision(fp_first, "circle", 0.5) == pytest.approx(0.5, abs=0.01)
    misses = _ev([Image((), (Gt("circle", _disc()),))])
    assert average_precision(misses, "circle", 0.5) == 0.0
    assert mean_ap(perfect, ["circle"]) == pytest.approx(1.0)
    assert mean_ap([], ["circle"]) == 0.0


def test_bootstrap_ci_brackets_the_statistic():
    ims = [Image((), (Gt("circle", _disc()),))] * 5 + [
        Image((Pred("circle", 0.9, _disc()),), (Gt("circle", _disc()),))
    ] * 5
    value, lo, hi = bootstrap_ci(_ev(ims), lambda s: mean_ap(s, ["circle"], (0.5,)), n=200)
    assert lo <= value <= hi and 0.0 <= lo < hi <= 1.0


# -- evaluation -----------------------------------------------------------------------------


class _Backend:
    """Predicts the labelled red disc (at 1/4 resolution) in every frame."""

    def infer(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h // 4, w // 4), np.float32)
        cv2.circle(mask, (10, 7), 3, 1.0, -1)
        return [RawInstance(TargetClass.CIRCLE, mask, (28, 18, 52, 42), 0.8)]


def test_eval_cli_end_to_end(tmp_path, monkeypatch, capsys):
    ds = _dataset(tmp_path / "ds", [0, 41, 42, 43, 91])
    split = tmp_path / "split.json"
    splits.main(["--dataset", str(ds), "--out", str(split)])
    monkeypatch.setattr(evaluate, "load_backend", lambda weights, conf: _Backend())
    out = tmp_path / "eval.json"
    args = ["--weights", "m.onnx", "--dataset", str(ds), "--split", str(split), "--out", str(out)]
    assert evaluate.main([*args, "--bootstrap", "50"]) == 0
    report = json.loads(out.read_text())
    assert report["images"] == 4 and report["instances"] == 4
    # labels are squares, predictions discs of the inscribed size: matched but imperfect
    assert report["mask_ap50"]["value"] == pytest.approx(1.0)
    assert report["boundary_f1"]["n"] == 4 and 0.0 < report["boundary_f1"]["mean"] < 1.0
    # a 3 px disc at 1/4 resolution, upscaled: blocky, so clearly below a smooth disc's 1.0
    assert 0.7 < report["circularity"]["mean"] < 0.9 and "ci95" in report["radius_rel_err"]
    assert '"subset": "test"' in capsys.readouterr().out
