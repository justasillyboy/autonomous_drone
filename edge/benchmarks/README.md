# Benchmarks

`uav-edge-bench` times each stage of the perception + mission loop: preprocess (letterbox),
inference, postprocess (decode, NMS, masks), detect (geometry, colour, filters) and the
mission step, and reports p50 / p95 / p99.

## Reference: Mac CPU (not the target hardware)

Apple M1 Pro, ONNX Runtime CPU provider, 200 real 800x600 frames after 10 warm-up frames
(`results/mac-m1-cpu_*.json`):

| Model | Inference p50 / p95 | Everything else (p50) | Total p50 | FPS |
|---|---|---|---|---|
| YOLO11s-seg @ 416 | 70 / 119 ms | 1.6 ms | 72 ms | 12.7 |
| YOLO11s-seg @ 640 | 153 / 179 ms | 2.5 ms | 156 ms | 6.4 |

Inference is 97-98 % of the loop. Decoding, geometry, colour and the mission step together
take about 1-2 ms, so the CPU side will not be the bottleneck on the Pi 5; whether the loop
reaches the 20 Hz target depends on the Hailo-8L. The legacy README's "~2 ms inference" was
never measured; the Pi 5 + Hailo-8L numbers replace it in M4.

## To measure on the Pi 5 (M4)

See `edge/hailo/README.md`: Hailo INT8 vs. ONNX on the Pi's CPU, the 10-minute thermal run,
and YOLO11n vs. YOLO11s vs. YOLOv8n (experiment E3).
