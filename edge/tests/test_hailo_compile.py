from pathlib import Path

import cv2
import numpy as np
import pytest
from uav_edge.hailo_compile import calibration_set, end_nodes

ASSET = Path(__file__).resolve().parents[2] / "assets" / "yolo11s-seg_synth_export-2026-04-24.onnx"


def test_calibration_set_is_letterboxed_rgb(tmp_path):
    paths = []
    for k in range(10):
        img = np.zeros((60, 80, 3), np.uint8)
        img[..., 2] = 200  # red in BGR
        p = tmp_path / f"{k}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    calib = calibration_set(paths, (64, 64), n=4)
    assert calib.shape == (4, 64, 64, 3) and calib.dtype == np.uint8  # frames 0, 2, 4, 6
    assert calib[0, 32, 32, 0] == 200 and calib[0, 0, 0, 0] == 114  # RGB, grey padding
    with pytest.raises(ValueError):
        calibration_set([tmp_path / "missing.png"], (64, 64))


@pytest.mark.skipif(not ASSET.exists(), reason="needs scripts/fetch_assets.sh")
def test_end_nodes_of_the_real_export():
    onnx = pytest.importorskip("onnx")
    nodes = end_nodes(onnx.load(str(ASSET)))
    assert len(nodes) == 10
    assert nodes[:3] == [
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv4.0/cv4.0.2/Conv",
    ]


def test_end_nodes_need_the_yolo_head():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [x])
    with pytest.raises(ValueError, match="9 head"):
        end_nodes(helper.make_model(graph))
