"""Hailo-8L backend (HailoRT InferModel API). Runs only on a host with the device and the
`hailo_platform` wheel matching the PCIe driver; everything after inference is the shared,
tested numpy path (heads.decode_heads -> yolo_seg.decode).

Not exercised in CI: there is no device. The post-processing it relies on is checked
against the ONNX graph cut at the same nodes (tests/test_heads.py). The .hef comes from
edge/hailo (M4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from uav_core.perception import RawInstance
from uav_core.types import TargetClass

from uav_edge.runtime.heads import decode_heads, group_outputs
from uav_edge.runtime.letterbox import Letterbox
from uav_edge.runtime.yolo_seg import decode

TIMEOUT_MS = 1000


class HailoBackend:  # pragma: no cover - needs the device
    def __init__(
        self,
        hef_path: str | Path,
        classes: dict[int, TargetClass],
        conf: float = 0.5,
        iou: float = 0.7,
    ):
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._device = VDevice(params)
        self._model = self._device.create_infer_model(str(hef_path))
        self._model.set_batch_size(1)
        self._model.input().set_format_type(FormatType.UINT8)
        for out in self._model.outputs:
            out.set_format_type(FormatType.FLOAT32)
        self._configured = self._model.configure()
        self.in_h, self.in_w = self._model.input().shape[:2]  # HWC
        self.classes, self.conf, self.iou = classes, conf, iou
        self._bindings = self._configured.create_bindings()
        self._outputs = {
            name: np.empty(self._model.output(name).shape, np.float32)
            for name in self._model.output_names
        }
        for name, buf in self._outputs.items():
            self._bindings.output(name).set_buffer(buf)

    def infer(self, frame_bgr: np.ndarray) -> list[RawInstance]:
        lb, rgb = self.preprocess(frame_bgr)
        return self.postprocess(self.run(rgb), lb)

    # The three stages, separately timed by uav-edge-bench.
    def preprocess(self, frame_bgr: np.ndarray) -> tuple[Letterbox, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        lb = Letterbox(w, h, self.in_w, self.in_h)
        return lb, np.ascontiguousarray(lb.apply(frame_bgr)[:, :, ::-1])

    def run(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        self._bindings.input().set_buffer(rgb)
        self._configured.run([self._bindings], TIMEOUT_MS)
        return self._outputs

    def postprocess(self, outputs: dict[str, np.ndarray], lb: Letterbox) -> list[RawInstance]:
        scales, proto = group_outputs(outputs, (self.in_h, self.in_w), len(self.classes))
        proto_nchw = np.transpose(proto, (2, 0, 1))[None]
        return decode(decode_heads(scales), proto_nchw, lb, self.classes, self.conf, self.iou)

    def close(self) -> None:
        self._configured = None
        self._device.release()
