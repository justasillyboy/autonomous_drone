"""
export.py — Export trained YOLO11-seg weights to deployment formats.

Supported targets:
  - ONNX         (cross-platform, CPU/GPU)
  - TensorRT     (NVIDIA GPU, fastest inference)
  - CoreML       (Apple Silicon / iOS)
  - OpenVINO     (Intel CPU/NPU)

Usage:
    python export.py --weights best.pt --format onnx
    python export.py --weights best.pt --format engine --half
    python export.py --weights best.pt --format all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Export configurations per format
# ---------------------------------------------------------------------------

FORMAT_DEFAULTS: dict[str, dict] = {
    "onnx": {
        "simplify": True,
        "opset": 17,
        "dynamic": False,       # set True for variable batch size
    },
    "engine": {                 # TensorRT
        "half": True,           # FP16 — safe for most cards (Ampere+)
        "workspace": 4,         # GB of GPU workspace for TRT optimisation
        "dynamic": False,
    },
    "coreml": {
        "half": True,
        "nms": True,            # bake NMS into CoreML graph
    },
    "openvino": {
        "half": False,          # INT8 optional via --int8 flag
    },
}

SUPPORTED_FORMATS = list(FORMAT_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Single-format export
# ---------------------------------------------------------------------------

def export_model(
    weights: str,
    fmt: str,
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    batch: int = 1,
    device: int | str = 0,
    extra_kwargs: dict | None = None,
) -> str:
    """
    Export a YOLO11-seg model to the specified format.

    Args:
        weights     : Path to trained .pt checkpoint.
        fmt         : Export format string (onnx | engine | coreml | openvino).
        imgsz       : Input resolution — must match training imgsz.
        half        : Enable FP16 quantisation (TRT / CoreML only).
        int8        : Enable INT8 quantisation (TRT only — needs calibration data).
        batch       : Static batch size baked into the graph.
        device      : GPU index or 'cpu'.
        extra_kwargs: Additional kwargs passed to model.export().

    Returns:
        Path to the exported model file.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")

    model = YOLO(weights)
    kwargs = FORMAT_DEFAULTS[fmt].copy()

    # CLI overrides
    if half:
        kwargs["half"] = True
    if int8:
        kwargs["int8"] = True

    kwargs.update({
        "format": fmt,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
    })
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print(f"\nExporting {Path(weights).name} → {fmt.upper()}")
    print(f"  Config: {kwargs}")

    exported_path = model.export(**kwargs)
    print(f"  Saved  → {exported_path}\n")
    return str(exported_path)


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def benchmark(weights: str, fmt: str, imgsz: int = 640, n_runs: int = 100) -> dict:
    """
    Measure inference latency for a deployed model.

    Returns dict with mean_ms, std_ms, fps.
    """
    import time
    import numpy as np

    model = YOLO(weights)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    fps     = 1000.0 / mean_ms

    print(f"\nBenchmark ({fmt.upper()}, {n_runs} runs, imgsz={imgsz})")
    print(f"  Mean latency : {mean_ms:.2f} ± {std_ms:.2f} ms")
    print(f"  Throughput   : {fps:.1f} FPS")
    return {"mean_ms": round(mean_ms, 2), "std_ms": round(std_ms, 2), "fps": round(fps, 1)}


# ---------------------------------------------------------------------------
# Validation — check exported ONNX produces matching outputs
# ---------------------------------------------------------------------------

def validate_onnx_export(
    pt_weights: str,
    onnx_path: str,
    imgsz: int = 640,
    atol: float = 1e-3,
) -> bool:
    """
    Run the same dummy input through the .pt model and the exported ONNX
    and compare outputs. Catches quantisation or graph-simplification errors.

    Requires onnxruntime: pip install onnxruntime-gpu

    Returns True if outputs match within atol.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping ONNX validation.")
        return False

    import numpy as np
    import torch
    from ultralytics import YOLO

    model = YOLO(pt_weights)
    dummy_np  = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    dummy_pt  = torch.from_numpy(dummy_np)

    with torch.no_grad():
        pt_out = model.model(dummy_pt)

    sess   = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: dummy_np})

    # Compare first output tensor (detection output)
    def get_first_tensor(obj):
        if hasattr(obj, "numpy"):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, (list, tuple)) and len(obj) > 0:
            return get_first_tensor(obj[0])
        return np.array(obj)

    pt_arr = get_first_tensor(pt_out)
    ort_arr = get_first_tensor(ort_out)
    ort_arr = ort_out[0]

    if pt_arr.shape != ort_arr.shape:
        print(f"Shape mismatch: PT {pt_arr.shape} vs ONNX {ort_arr.shape}")
        return False

    max_diff = float(np.abs(pt_arr - ort_arr).max())
    passed   = max_diff <= atol
    status   = "PASS" if passed else "FAIL"
    print(f"ONNX validation {status} (max diff = {max_diff:.2e}, atol = {atol})")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO11-seg circle model")
    parser.add_argument("--weights",   required=False, default=None,
                        help="Path to .pt weights (not needed for --extract-calib-frames)")
    parser.add_argument("--format",    default="onnx",
                        choices=SUPPORTED_FORMATS + ["all"],
                        help="Export format (default: onnx)")
    parser.add_argument("--imgsz",     type=int,   default=640)
    parser.add_argument("--half",      action="store_true", help="FP16 export")
    parser.add_argument("--int8",      action="store_true", help="INT8 export (TRT only)")
    parser.add_argument("--batch",     type=int,   default=1)
    parser.add_argument("--device",    default="0")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark after export")
    parser.add_argument("--validate",  action="store_true",
                        help="Validate ONNX output against .pt")
    # Hailo calibration frame extraction
    parser.add_argument("--extract-calib-frames", default=None, metavar="SOURCE",
                        help="Extract real frames for Hailo INT8 calibration. "
                             "Pass a flight video or image folder.")
    parser.add_argument("--calib-output", default="hailo_calib",
                        help="Output dir for calibration frames (default: hailo_calib/)")
    parser.add_argument("--calib",     default=None,
                        help="calibration.npz for undistortion during frame extraction")
    parser.add_argument("--n-calib",   type=int, default=150,
                        help="Number of calibration frames to extract (default 150)")
    args = parser.parse_args()

    # ── Hailo calibration extraction (standalone, no weights needed) ──────────
    if args.extract_calib_frames:
        extract_hailo_calibration_frames(
            source=args.extract_calib_frames,
            output_dir=args.calib_output,
            n_frames=args.n_calib,
            imgsz=args.imgsz,
            calib_path=args.calib,
        )
        import sys; sys.exit(0)

    if args.weights is None:
        print("Error: --weights is required for export. "
              "Use --extract-calib-frames for calibration frame extraction.")
        import sys; sys.exit(1)

    formats = SUPPORTED_FORMATS if args.format == "all" else [args.format]
    exported_paths = []

    for fmt in formats:
        path = export_model(
            weights=args.weights,
            fmt=fmt,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            batch=args.batch,
            device=args.device,
        )
        exported_paths.append((fmt, path))

        if args.benchmark:
            benchmark(path, fmt, imgsz=args.imgsz)

        if args.validate and fmt == "onnx":
            validate_onnx_export(args.weights, path, imgsz=args.imgsz)

    print("\nAll exports complete:")
    for fmt, path in exported_paths:
        print(f"  {fmt.upper():<10} → {path}")


# ---------------------------------------------------------------------------
# Hailo INT8 calibration frame extractor
# ---------------------------------------------------------------------------

def extract_hailo_calibration_frames(
    source: str,
    output_dir: str,
    n_frames: int = 150,
    imgsz: int = 640,
    calib_path: str | None = None,
) -> str:
    """
    Extract real-world calibration frames for Hailo DFC INT8 quantization.

    WHY THIS MATTERS:
      Custom boundary loss trains the model to capture sharp circle edges.
      When Hailo DFC quantizes FP32 → INT8 using SYNTHETIC images, it
      miscalibrates the weight distribution because synthetic pixels are
      too clean and don't match the real IMX219 sensor's noise, vignetting,
      or motion blur. The boundary-sensitive feature maps degrade first.

      Using 100-200 REAL frames captured by the actual camera (post-
      undistortion) preserves edge sharpness through quantization.

    Args:
        source      : Path to a real flight video (mp4/avi) or image folder.
        output_dir  : Directory to save calibration JPEGs.
                      Pass this to Hailo DFC: --calib-set-path <output_dir>
        n_frames    : Number of frames to extract (100-200 recommended).
        imgsz       : Resize frames to this square size (must match YOLO imgsz).
        calib_path  : Path to calibration.npz for IMX219 undistortion.
                      Strongly recommended — extract post-undistort frames.

    Returns:
        Path to output_dir.

    Usage:
        # Step 1: extract real frames
        python export.py --extract-calib-frames flight.mp4 \
                         --calib-output hailo_calib/ \
                         --calib calibration.npz

        # Step 2: compile with Hailo DFC (run in Hailo Docker)
        hailomz compile yolo11s_seg \
            --ckpt best.onnx \
            --calib-set-path hailo_calib/ \
            --classes 3
    """
    import math
    import cv2
    from pathlib import Path as _Path

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load undistorter if calibration provided
    undist = None
    if calib_path:
        try:
            from camera_calibration import IMX219Undistorter
            undist = IMX219Undistorter(calib_path, alpha=0.0, crop=True)
            print(f"[HailoCalib] Undistortion enabled from {calib_path}")
        except Exception as e:
            print(f"[HailoCalib] WARNING: could not load calibration ({e}) — "
                  f"extracting raw frames")

    source_path = _Path(source)

    # ── Image folder source ───────────────────────────────────────────────────
    if source_path.is_dir():
        exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([p for p in source_path.iterdir()
                         if p.suffix.lower() in exts])
        step   = max(1, len(images) // n_frames)
        frames_to_use = images[::step][:n_frames]
        saved = 0
        for i, img_path in enumerate(frames_to_use):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if undist is not None:
                frame = undist.correct(frame)
            frame = cv2.resize(frame, (imgsz, imgsz))
            out_path = out / f"calib_{i:05d}.jpg"
            cv2.imwrite(str(out_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        print(f"[HailoCalib] Saved {saved} frames from image folder → {out}")
        return str(out)

    # ── Video source ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, total // n_frames)
    saved  = 0
    f_idx  = 0
    s_idx  = 0

    print(f"[HailoCalib] Extracting {n_frames} frames from "
          f"{source_path.name} ({total} total, every {step} frames)")

    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % step == 0:
            if undist is not None:
                frame = undist.correct(frame)
            frame = cv2.resize(frame, (imgsz, imgsz))
            out_path = out / f"calib_{s_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            s_idx += 1
        f_idx += 1

    cap.release()
    print(f"[HailoCalib] Saved {saved} calibration frames → {out.resolve()}")
    print(f"\nNext step (Hailo DFC inside Docker):")
    print(f"  hailomz compile yolo11s_seg \\")
    print(f"      --ckpt best.onnx \\")
    print(f"      --calib-set-path {out.resolve()} \\")
    print(f"      --classes 3")
    return str(out)
