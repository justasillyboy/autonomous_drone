from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from uav_core.config import PerceptionConfig
from uav_core.perception import detect
from uav_core.perception.geometry import extract_geometry
from uav_core.types import TargetClass
from uav_edge.runtime import Letterbox, class_map, load_backend
from uav_edge.runtime.onnx import OnnxBackend, to_blob
from uav_edge.runtime.ultralytics import from_result
from uav_edge.runtime.yolo_seg import decode, nms

NAMES = {0: "circle", 1: "semicircle", 2: "H_landing"}
CLASSES = class_map(NAMES)
IN = 640
NM = 4
ASSETS = Path(__file__).resolve().parents[2] / "assets"


# -- letterbox ----------------------------------------------------------------------------


def test_letterbox_pads_720p_top_and_bottom():
    lb = Letterbox(1280, 720, IN, IN)
    assert lb.gain == 0.5
    assert lb.pad == (0.0, 140.0)
    img = lb.apply(np.zeros((720, 1280, 3), np.uint8))
    assert img.shape == (IN, IN, 3)
    assert (img[:140] == 114).all() and (img[140:500] == 0).all() and (img[500:] == 114).all()


def test_box_round_trip_and_clip():
    lb = Letterbox(1280, 720, IN, IN)
    boxes = lb.box_to_frame(np.array([[100.0, 200.0, 300.0, 400.0], [-10, 0, 700, 650]]))
    np.testing.assert_allclose(boxes[0], [200, 120, 600, 520])
    np.testing.assert_allclose(boxes[1], [0, 0, 1280, 720])


@pytest.mark.parametrize(("res", "shape"), [(160, (90, 160)), (640, (360, 640))])
def test_crop_masks_keeps_the_frame_aspect(res, shape):
    masks = np.zeros((2, res, res))
    assert Letterbox(1280, 720, IN, IN).crop_masks(masks).shape == (2, *shape)


def test_no_padding_means_no_crop():
    masks = np.zeros((1, 120, 160))
    assert Letterbox(640, 480, 640, 480).crop_masks(masks).shape == (1, 120, 160)


# -- class map ----------------------------------------------------------------------------


def test_class_map_matches_by_name_not_position():
    assert class_map("{0: 'H_landing', 1: 'circle'}") == {
        0: TargetClass.H_LANDING,
        1: TargetClass.CIRCLE,
    }


def test_class_map_rejects_unknown_classes():
    with pytest.raises(ValueError, match="person"):
        class_map({0: "circle", 1: "person"})


# -- decoder ------------------------------------------------------------------------------


def _disc_proto(cx, cy, r, res=IN // 4):
    """Prototype 0 is +10 inside a disc and -10 outside; the others are zero."""
    yy, xx = np.mgrid[:res, :res]
    proto = np.zeros((1, NM, res, res), np.float32)
    proto[0, 0] = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= r**2, 10.0, -10.0)
    return proto


def _pred(rows):
    """rows: (cx, cy, w, h, cls, score) in network-input pixels -> (1, 4+3+NM, N)."""
    out = np.zeros((len(rows), 4 + 3 + NM), np.float32)
    for k, (cx, cy, w, h, c, s) in enumerate(rows):
        out[k, :4] = cx, cy, w, h
        out[k, 4 + c] = s
        out[k, 7] = 1.0  # coefficient for prototype 0
    return out.T[None]


def test_decoded_mask_lands_on_the_right_frame_pixels():
    # A 1280x720 frame with a disc of radius 100 px at (900, 300).
    lb = Letterbox(1280, 720, IN, IN)
    cx_in, cy_in, r_in = 900 * 0.5, 300 * 0.5 + 140, 50  # network-input pixels
    inst = decode(
        _pred([(cx_in, cy_in, 2 * r_in, 2 * r_in, 0, 0.9)]),
        _disc_proto(cx_in / 4, cy_in / 4, r_in / 4),
        lb,
        CLASSES,
        conf_thres=0.5,
    )
    assert len(inst) == 1 and inst[0].cls is TargetClass.CIRCLE
    np.testing.assert_allclose(inst[0].xyxy, (800, 200, 1000, 400))
    geo = extract_geometry(inst[0].mask, 1280, 720, 5)
    assert abs(geo.cx - 900) <= 8 and abs(geo.cy - 300) <= 8
    assert abs(geo.radius - 100) <= 10


def test_low_scores_and_duplicates_are_dropped_but_other_classes_kept():
    lb = Letterbox(640, 640, IN, IN)
    rows = [
        (320, 320, 100, 100, 0, 0.9),
        (322, 322, 100, 100, 0, 0.8),  # duplicate circle -> suppressed
        (320, 320, 100, 100, 2, 0.7),  # same box, other class -> kept
        (100, 100, 50, 50, 0, 0.3),  # below threshold
    ]
    inst = decode(_pred(rows), _disc_proto(80, 80, 12), lb, CLASSES, conf_thres=0.5)
    assert [(i.cls, round(i.conf, 1)) for i in inst] == [
        (TargetClass.CIRCLE, 0.9),
        (TargetClass.H_LANDING, 0.7),
    ]


def test_nothing_above_threshold():
    lb = Letterbox(640, 640, IN, IN)
    assert decode(_pred([(1, 1, 2, 2, 0, 0.1)]), _disc_proto(0, 0, 1), lb, CLASSES, 0.5) == []


def test_nms_orders_by_score():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [1, 1, 10, 10]], float)
    assert nms(boxes, np.array([0.5, 0.9, 0.6]), 0.5).tolist() == [1, 2]


# -- ONNX backend (tiny constant-output model, so CI needs no weights) ---------------------


def _tiny_model(path: Path, pred: np.ndarray, proto: np.ndarray, names=NAMES):
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IN, IN])
    outs = [
        helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(pred.shape)),
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, list(proto.shape)),
    ]
    nodes = [
        helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(pred)),
        helper.make_node("Constant", [], ["output1"], value=numpy_helper.from_array(proto)),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "tiny", [x], outs), opset_imports=[helper.make_opsetid("", 17)]
    )
    if names is not None:
        helper.set_model_props(model, {"names": str(names)})
    model.ir_version = 9
    onnx.save(model, path)
    return path


def test_onnx_backend_end_to_end(tmp_path):
    cx_in, cy_in, r_in = 320.0, 240.0 + 80, 40  # 640x480 frame -> 80 px top padding
    path = _tiny_model(
        tmp_path / "tiny.onnx",
        _pred([(cx_in, cy_in, 2 * r_in, 2 * r_in, 0, 0.95)]),
        _disc_proto(cx_in / 4, cy_in / 4, r_in / 4),
    )
    backend = load_backend(path)
    frame = np.full((480, 640, 3), (0, 0, 200), np.uint8)  # red everywhere
    dets = detect(backend.infer(frame), frame, PerceptionConfig())
    assert len(dets) == 1
    d = dets[0]
    assert d.color == "red" and abs(d.cx - 320) <= 4 and abs(d.cy - 240) <= 4


def test_onnx_model_without_names_is_rejected(tmp_path):
    path = _tiny_model(tmp_path / "bare.onnx", _pred([]), _disc_proto(0, 0, 1), names=None)
    with pytest.raises(ValueError, match="names"):
        OnnxBackend(path)


def test_to_blob_is_rgb_chw_unit_range():
    img = np.zeros((2, 3, 3), np.uint8)
    img[..., 2] = 255  # red in BGR
    blob = to_blob(img)
    assert blob.shape == (1, 3, 2, 3) and blob.dtype == np.float32
    assert blob[0, 0].max() == 1.0 and blob[0, 2].max() == 0.0


def test_unknown_weight_type():
    with pytest.raises(ValueError, match=r"\.hef"):
        load_backend("model.tflite")


@pytest.mark.skipif(
    not (ASSETS / "yolo11s-seg_synth_export-2026-04-24.onnx").exists(),
    reason="needs scripts/fetch_assets.sh",
)
def test_real_export_runs_on_a_frame():
    backend = OnnxBackend(ASSETS / "yolo11s-seg_synth_export-2026-04-24.onnx")
    assert (backend.in_w, backend.in_h) == (416, 416)
    frame = np.full((480, 640, 3), 90, np.uint8)
    for inst in backend.infer(frame):
        x1, y1, x2, y2 = inst.xyxy
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 480


# -- Ultralytics adapter (fake Results, so torch is not needed) ----------------------------


def test_from_result_crops_the_canvas_padding():
    canvas = np.zeros((1, 384, 640), np.float32)  # 1280x720 at imgsz 640, auto padding
    yy, xx = np.mgrid[:384, :640]
    canvas[0] = ((xx - 450) ** 2 + (yy - (150 + 12)) ** 2 <= 50**2).astype(np.float32)
    result = SimpleNamespace(
        orig_shape=(720, 1280),
        masks=SimpleNamespace(data=canvas),
        boxes=_Boxes(np.array([[800.0, 200.0, 1000.0, 400.0]]), np.array([0.0]), np.array([0.9])),
    )
    inst = from_result(result, CLASSES)
    assert inst[0].mask.shape == (360, 640)
    geo = extract_geometry(inst[0].mask, 1280, 720, 5)
    assert abs(geo.cx - 900) <= 4 and abs(geo.cy - 300) <= 4


def test_from_result_without_masks():
    assert from_result(SimpleNamespace(masks=None, boxes=None), CLASSES) == []


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = xyxy, cls, conf

    def __len__(self):
        return len(self.cls)
