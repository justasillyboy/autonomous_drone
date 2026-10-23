"""The Hailo post-processing path: raw per-scale heads -> the ONNX export's output tensor."""

from pathlib import Path

import numpy as np
import pytest
from uav_edge.runtime.heads import (
    REG_MAX,
    ScaleOutputs,
    decode_heads,
    dfl_distances,
    group_outputs,
    nchw_to_hwc,
)

ASSET = Path(__file__).resolve().parents[2] / "assets" / "yolo11s-seg_synth_export-2026-04-24.onnx"


def _peaked(h, w, value):
    """DFL logits whose softmax puts all mass on bin `value` for all four sides."""
    logits = np.full((h, w, 4, REG_MAX), -50.0, np.float32)
    logits[..., value] = 50.0
    return logits.reshape(h, w, 4 * REG_MAX)


def test_dfl_expectation_of_a_peaked_distribution():
    assert dfl_distances(_peaked(2, 3, 5)) == pytest.approx(np.full((2, 3, 4), 5.0))


def test_decode_heads_box_and_order():
    nc, nm = 3, 32
    grid = {s: 64 // s for s in (8, 16, 32)}
    scales = [  # any order in, stride order out
        ScaleOutputs(s, _peaked(g, g, 2), np.zeros((g, g, nc)), np.ones((g, g, nm)))
        for s, g in sorted(grid.items(), reverse=True)
    ]
    pred = decode_heads(scales)
    assert pred.shape == (1, 4 + nc + nm, 8 * 8 + 4 * 4 + 2 * 2)
    cx, cy, w, h = pred[0, :4, 0]  # first anchor of stride 8: cell (0, 0)
    assert (cx, cy, w, h) == pytest.approx((4.0, 4.0, 32.0, 32.0))  # 2 cells each side
    assert pred[0, 4, 0] == pytest.approx(0.5)  # sigmoid(0)
    assert pred[0, :4, 64] == pytest.approx((8.0, 8.0, 64.0, 64.0))  # first stride-16 anchor


def test_group_outputs_by_shape():
    nc, nm, size = 3, 32, 64
    tensors = {"proto": np.zeros((1, 16, 16, nm))}
    for s in (8, 16, 32):
        g = size // s
        tensors[f"b{s}"] = np.zeros((g, g, 64))
        tensors[f"c{s}"] = np.zeros((g, g, nc))
        tensors[f"m{s}"] = np.zeros((g, g, nm))
    scales, proto = group_outputs(tensors, (size, size), nc)
    assert [s.stride for s in scales] == [8, 16, 32] and proto.shape == (16, 16, nm)
    del tensors["b16"]
    with pytest.raises(ValueError):
        group_outputs(tensors, (size, size), nc)
    with pytest.raises(ValueError, match="unexpected"):
        group_outputs({"x": np.zeros((8, 8, 7))}, (size, size), nc)


@pytest.mark.skipif(not ASSET.exists(), reason="needs scripts/fetch_assets.sh")
def test_matches_the_onnx_graph_cut_at_the_hailo_end_nodes(tmp_path):
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort

    heads = [f"/model.23/cv{k}.{s}/cv{k}.{s}.2/Conv_output_0" for s in range(3) for k in (2, 3, 4)]
    cut = tmp_path / "heads.onnx"
    onnx.utils.extract_model(str(ASSET), str(cut), ["images"], [*heads, "output1"])
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 416, 416), dtype=np.float32)
    full_pred, full_proto = ort.InferenceSession(str(ASSET)).run(None, {"images": x})
    raw = ort.InferenceSession(str(cut)).run(None, {"images": x})
    scales = [
        ScaleOutputs(stride, *(nchw_to_hwc(raw[3 * i + j]) for j in range(3)))
        for i, stride in enumerate((8, 16, 32))
    ]
    pred = decode_heads(scales)
    assert pred.shape == full_pred.shape
    np.testing.assert_allclose(pred, full_pred, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(raw[-1], full_proto, rtol=1e-5, atol=1e-5)
