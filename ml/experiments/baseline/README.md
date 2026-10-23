# Baseline: the legacy models on the frozen real test split

```bash
uv run uav-ml-eval --weights assets/<model>.onnx --dataset <unzipped dataset-real-286> \
    --split ml/configs/splits/real-286.json --subset test --conf 0.25 --out <model>.json
```

Test split: 54 frames, 169 labelled instances (no semicircle in this subset). Values with
95 % bootstrap confidence intervals over frames (1000 resamples).

| Model | Mask AP50 | Mask AP50-95 | AP50 circle | AP50 H_landing | Boundary F1 (2 px) |
|---|---|---|---|---|---|
| `yolo11s-seg_synth_export-2026-04-11.onnx` | 0.37 [0.29, 0.45] | 0.23 [0.19, 0.27] | 0.58 | 0.15 | 0.74 [0.68, 0.80] |
| `yolo11s-seg_synth_export-2026-04-24.onnx` | 0.29 [0.24, 0.37] | 0.22 [0.19, 0.26] | 0.53 | 0.05 | 0.72 [0.66, 0.76] |

Both were trained on synthetic images only (defect D10). They find many fire circles but
almost never the helipad, which looks nothing like the synthetic one; landing on it is
worth 30 points. These are the numbers a model trained on the 178 real training frames has
to beat. An earlier note claimed "at most 0.03 on real frames": that came from the first
four frames only (camera close to the floor) and was wrong as a general statement.

The JSON files hold every metric, including circularity of the predicted circle masks and
centre and radius errors against the labelled circles (the radius error is the relative
error of the vision altitude).

When a circle is found, its geometry is good: on the matched circles of the 04-11 export the
centre is off by 1.4 px on average and the radius by +0.8 % [0.2 %, 1.4 %], so the vision
altitude `Z = f D / (2 r)` would read about 0.8 % low (2.6 cm at the 3.25 m drop height).
Detection, not geometry, is what needs fixing.
