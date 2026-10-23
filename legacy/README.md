# Legacy code (competition version, April 2026)

This directory holds the original code of the UAV fire-fighting competition entry, imported
**byte-for-byte** so that every later refactor shows up as a clean diff. Nothing here is
imported by the new packages; it is kept only as a behavioural reference (characterisation
tests) and is deleted step by step as the replacements land.

| Path | What it is |
|---|---|
| `vision/` | What is left of the Python stack: synthetic data generator, training, loss, evaluation, export, telemetry logger, docs. Replaced in M3 |
| `vision/dataset/` | Metadata of the labelled real-image dataset (Roboflow export, class-ID remap script). The images live in the asset release |

## Removed, with their replacements

Deleted at the end of M1. The last version of each is at commit
[`572856f`](https://github.com/justasillyboy/autonomous_drone/tree/572856f/legacy).

| Legacy file | Replaced by |
|---|---|
| `vision/mission.py` (3,095 lines) | `uav_core` (perception, estimation, control, mission, HUD); the characterisation goldens recorded from it stay in `uav_core/tests/characterisation/golden/` |
| `vision/test_video.py`, `vision/postprocess.py` | `uav-run-video` (`tools/`) |
| `vision/camera_calibration.py` | `uav_core.perception.camera` + `uav-calibrate-camera` |
| `vision/hsv_calibrator.py` | `uav-hsv fit` / `uav-hsv tune` (same JSON layout) |
| `ros2/uav_deploy/` | `ros2_ws/src/` (`uav_interfaces`, `uav_perception`, `uav_mission`, `uav_hardware`, `uav_bringup`) |

`vision/export.py --extract-calib-frames` imported `camera_calibration` and no longer runs;
it is replaced with the rest of the training code in M3.

## Snapshot history

The original project had no version control, only folders and zip files. Each one was
imported as its own commit, oldest first:

| Commit message date | Source (under `~/Documents/AI/UAV_Competition_NCKH/`) | Notes |
|---|---|---|
| 2026-04-12 | `fixed_yolo_circle_seg/files.zip` | First vision snapshot |
| 2026-04-13 | `yolo_circle_seg/` | Code and docs only |
| 2026-04-18 / 04-19 | `…/deploy_files_drone_competition.zip`, `…/UAV_node_ROS2.zip` | On-drone subset of `vision/` plus the first ROS 2 package |
| 2026-04-21 | `fixed_yolo_circle_seg/files/` and `…/UAV_node_ROS2/src/uav_deploy/` | Final competition state. `deploy_guide.md` comes from `deploy_files_drone_competition/`, which held the newer copy |

Excluded on import: build outputs (`build/ install/ log/ __pycache__`), a 5.5 GB Linux
virtualenv, macOS metadata, duplicate copies, and the rulebook / internal `.docx` files
(kept privately).

## Binary assets

Weights, videos and datasets are attached to the GitHub Release
[`legacy-assets-v0`](https://github.com/justasillyboy/autonomous_drone/releases/tag/legacy-assets-v0).
`scripts/fetch_assets.sh` downloads them into `assets/` and checks them against
`assets/SHA256SUMS`.

| Asset | Original location | Content |
|---|---|---|
| `yolo11s-seg_synth_2026-04-09.pt` | `yolo_circle_seg/best.pt` | YOLO11s-seg trained on synthetic data only |
| `yolo11n-seg_synth_2026-04-14.pt` | `fixed_yolo_circle_seg/files/best.pt` | YOLO11n-seg trained on synthetic data only |
| `yolo11s-seg_synth_export-2026-04-11.onnx` | `yolo_circle_seg/best.onnx` | ONNX export of a YOLO11s-seg synthetic model (Ultralytics 8.4.37) |
| `yolo11s-seg_synth_export-2026-04-24.onnx` | `fixed_yolo_circle_seg/files/best.onnx` | ONNX export of a YOLO11s-seg synthetic model (Ultralytics 8.4.41) |
| `dataset-real-286.zip` | `fixed_yolo_circle_seg/files/dataset.zip` | 286 real frames with polygon labels (`circle`, `semicircle`, `H_landing`), class IDs already remapped |
| `real-frames-300.zip` | `yolo_circle_seg/real_domain_images_new/` | 300 real frames; 286 of them are the labelled images above, 14 are unlabelled |
| `hailo-calib-frames.zip` | `fixed_yolo_circle_seg/files/hailo_calib/` | 286 frames prepared for Hailo INT8 calibration |
| `debug-images.zip` | `yolo_circle_seg/*.png, *.jpg, *.webp`, `yolo_circle_seg/dataset/` | Sample and debug images, plus one synthetic training sample with its label |
| `output_{2,3,7,8}.mp4` | project root | Annotated test videos |
| `result-v0.mp4`, `result_new-v0.mp4`, `ket_qua.mp4`, `final_flight_result{,_new}.mp4` | `yolo_circle_seg/` | Annotated test videos (04-13 era) |
| `result_new-v1.mp4` | `fixed_yolo_circle_seg/files/` | Annotated test video (04-21 era) |

## Facts established during the import

These correct claims made in the legacy docs:

- **No model was ever trained on real images.** All four weight/ONNX files report
  `trained on data/synthetic/circle_seg.yaml`; the real dataset was exported on 2026-04-23,
  after the last training run.
- **The ONNX that was meant for deployment is YOLO11s, not YOLO11n.** Both `.onnx` files are
  YOLO11s-seg, so `best.onnx` was not an export of the `best.pt` (YOLO11n) next to it, and
  it contradicts the "only `n` fits on Hailo-8L" rule in `vision/CLAUDE.md`.
- **The pipeline never ran end to end on hardware:** there is no `.hef`, no
  `calibration.npz` and no `hsv_calib.json` anywhere in the original folders.
- **"~2 ms inference" in the old README was never measured.**
