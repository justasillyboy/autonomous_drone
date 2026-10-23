"""uav-edge-compile: ONNX -> Hailo-8L .hef with an INT8 calibration set of REAL frames.

    uv run uav-ml-build --dataset <real frames> --split ml/configs/splits/real-286.json \
        --out data/real
    uav-edge-compile --onnx best.onnx --calib-dir data/real/images/train --out best.hef

Runs where the Hailo Dataflow Compiler runs: x86-64 Linux with the `hailo_dataflow_compiler`
wheel from the Hailo Developer Zone (account required), e.g. Colab, see edge/hailo/README.md.
The parts that do not need the DFC are tested on any machine:

- `end_nodes`: cut the graph before the DFL decode, at the 9 per-scale head convolutions and
  the prototype output, which is what the Hailo runtime path expects (runtime/heads.py).
- `calibration_set`: letterboxed RGB training frames from the frozen split. The legacy
  "never calibrate on synthetic frames" rule is kept and becomes experiment E4.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from uav_edge.runtime.letterbox import Letterbox

HEAD_RE = re.compile(r"^/model\.(\d+)/cv([234])\.(\d)/cv\2\.\3\.2/Conv$")


def end_nodes(model) -> list[str]:
    """Names of the head Conv nodes (box, class, coefficients per scale) and of the node
    producing the prototypes (`output1`), from an Ultralytics YOLO-seg ONNX graph."""
    heads = sorted(
        (int(m.group(3)), int(m.group(2)), n.name)
        for n in model.graph.node
        if (m := HEAD_RE.match(n.name))
    )
    if len(heads) != 9:
        raise ValueError(f"expected 9 head convolutions, found {len(heads)}")
    proto_out = model.graph.output[1].name
    proto = next(n.name for n in model.graph.node if proto_out in n.output)
    return [name for _, _, name in heads] + [proto]


def calibration_set(images: Sequence[Path], in_hw: tuple[int, int], n: int = 256) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB, letterboxed exactly as at runtime; frames spread evenly."""
    in_h, in_w = in_hw
    step = max(1, len(images) // n)
    out = []
    for path in list(images)[::step][:n]:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        out.append(Letterbox(w, h, in_w, in_h).apply(bgr)[:, :, ::-1])
    if not out:
        raise ValueError("no readable calibration frames")
    return np.stack(out)


MODEL_SCRIPT = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"


def compile_hef(onnx_path: Path, calib: np.ndarray, out: Path) -> Path:  # pragma: no cover
    import onnx
    from hailo_sdk_client import ClientRunner  # Hailo DFC, x86-64 only

    model = onnx.load(str(onnx_path))
    _, c, h, w = (d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(
        str(onnx_path),
        onnx_path.stem,
        start_node_names=[model.graph.input[0].name],
        end_node_names=end_nodes(model),
        net_input_shapes={model.graph.input[0].name: [1, c, h, w]},
    )
    runner.load_model_script(MODEL_SCRIPT)
    runner.optimize(calib)
    out.write_bytes(runner.compile())
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    import onnx

    parser = argparse.ArgumentParser(prog="uav-edge-compile", description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True, help="real training frames")
    parser.add_argument("--calib-frames", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    model = onnx.load(str(args.onnx))
    _, _, h, w = (d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    images = sorted(p for p in args.calib_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    calib = calibration_set(images, (h, w), args.calib_frames)
    print(f"end nodes: {end_nodes(model)}\ncalibration: {calib.shape}")
    print(compile_hef(args.onnx, calib, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
