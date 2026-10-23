# Compiling for the Hailo-8L

The Hailo Dataflow Compiler (DFC) runs only on x86-64 Linux and needs the
`hailo_dataflow_compiler` wheel from the Hailo Developer Zone (free account). The Mac cannot
run it; a Colab CPU runtime can.

```bash
pip install ./uav_core ./edge ./ml onnx hailo_dataflow_compiler-*.whl
uav-ml-build --dataset data/real --split ml/configs/splits/real-286.json --out data/real-yolo
uav-edge-compile --onnx best.onnx --calib-dir data/real-yolo/images/train --out best.hef
```

What `uav-edge-compile` does:

1. **Cuts the graph before the DFL decode**, at the 9 per-scale head convolutions (box,
   class, mask coefficients for strides 8 / 16 / 32) and the prototype output. On the Pi,
   `runtime/heads.py` rebuilds the full Ultralytics output from these in numpy;
   `tests/test_heads.py` checks that against the ONNX graph cut at the same nodes.
2. **Calibrates INT8 on real frames** (the frozen training split, letterboxed as at
   runtime), never on synthetic ones, as the legacy notes required. Experiment E4 measures
   what synthetic calibration would cost.
3. Normalises 0-255 to 0-1 on the device (`normalization` in the model script), so the
   runtime feeds uint8 RGB.

On the Pi: Raspberry Pi OS Bookworm with the HailoRT PCIe driver, and **the same HailoRT
version** in the `hailo_platform` wheel as the driver. Then:

```bash
uav-edge-bench --weights best.hef --source flight.mp4 --duration 600 --out pi5-hailo.json
uav-edge-bench --weights best.onnx --source flight.mp4 --frames 300 --out pi5-cpu.json
```

The 10-minute run records SoC temperature and `vcgencmd get_throttled` every 10 s.

YOLO11-seg uses ops the DFC may not support in every version; YOLOv8n-seg is in the Hailo
Model Zoo and is the fallback (experiment E3 already trains it).
