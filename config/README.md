# Field configuration

Files loaded at the boundary (launch file or CLI), never imported by code.

## `hsv_calib.json`

HSV colour ranges fitted from the 286 labelled real frames (`assets/dataset-real-286.zip`)
with:

```bash
uv run uav-hsv fit --dataset <unzipped dataset> --output config/hsv_calib.json
```

Fitted on the first 201 frames (by file name), evaluated on the last 85:

| Ranges | Circles given the right colour | Floor pixels wrongly matched (red / yellow / blue) |
|---|---|---|
| Defaults (legacy `COLOR_HSV`) | 41.8 % | 0.1 % / 0.0 % / 0.0 % |
| Fitted | 95.1 % | 0.2 % / 0.3 % / 2.7 % |

The reference colour of each circle comes from a hue vote, not from human labels (the
dataset has no colour labels); a visual check of 40 random held-out circles agreed with the
vote in all 40. On this camera the yellow targets look pale (low saturation) and the blue
ones look violet (hue about 130-150), which is why the defaults miss most of them.

These frames come from one venue and lighting. Refit at the competition field, or tune by
hand with `uv run --with opencv-python uav-hsv tune --source 0 --input config/hsv_calib.json`.
