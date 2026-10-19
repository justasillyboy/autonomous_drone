import json

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from uav_core.config import DEFAULT_HSV_RANGES, PerceptionConfig
from uav_core.perception import RawInstance, detect
from uav_core.perception.altitude_vision import altitude_from_diameter, f_px_from_calibration
from uav_core.perception.color import color_in_contour, color_of_mask, load_hsv_calibration
from uav_core.perception.filters import overlaps_h_landing
from uav_core.perception.geometry import diameter_ratio, extract_geometry
from uav_core.types import Color, TargetClass

W, H = 640, 480
RED_BGR = (27, 42, 252)


def disc_mask(cx: float, cy: float, r: float, ratio: int = 4) -> np.ndarray:
    mask = np.zeros((H // ratio, W // ratio), np.float32)
    cv2.circle(mask, (round(cx / ratio), round(cy / ratio)), max(1, round(r / ratio)), 1.0, -1)
    return mask


def test_geometry_of_a_centred_disc():
    geo = extract_geometry(disc_mask(320, 240, 80), W, H, edge_clip_margin_px=5)
    assert geo is not None
    assert abs(geo.cx - 320) <= 4 and abs(geo.cy - 240) <= 4
    assert abs(geo.radius - 80) <= 6
    assert geo.fill_ratio > 0.8
    assert not geo.edge_clipped


def test_empty_mask_has_no_geometry():
    assert extract_geometry(np.zeros((120, 160), np.float32), W, H, 5) is None


def test_disc_cut_by_the_border_is_edge_clipped():
    geo = extract_geometry(disc_mask(620, 240, 80), W, H, 5)
    assert geo is not None and geo.edge_clipped


def test_ring_has_low_fill_ratio():
    mask = np.zeros((120, 160), np.float32)
    cv2.circle(mask, (80, 60), 20, 1.0, 1)
    geo = extract_geometry(mask, W, H, 5)
    assert geo is not None and geo.fill_ratio < 0.4


@settings(max_examples=60, deadline=None)
@given(st.integers(8, 200), st.integers(0, W), st.integers(0, H))
def test_geometry_scales_mask_to_frame(r, cx, cy):
    geo = extract_geometry(disc_mask(cx, cy, r), W, H, 5)
    if geo is None:
        return
    inside = r + 6 < cx < W - r - 6 and r + 6 < cy < H - r - 6
    if inside:
        assert not geo.edge_clipped
        assert abs(geo.radius - r) <= 6
    assert 0.0 <= geo.fill_ratio <= 1.0


def test_diameter_ratio():
    assert diameter_ratio(160, 640) == 0.5


def test_color_of_mask_picks_red():
    roi = np.full((40, 40, 3), RED_BGR, np.uint8)
    mask = np.full((40, 40), 255, np.uint8)
    assert color_of_mask(mask, roi, DEFAULT_HSV_RANGES, 0.25) is Color.RED


def test_grey_matches_no_colour():
    roi = np.full((40, 40, 3), (90, 90, 90), np.uint8)
    mask = np.full((40, 40), 255, np.uint8)
    assert color_of_mask(mask, roi, DEFAULT_HSV_RANGES, 0.25) is None


def test_color_in_contour_outside_frame_is_none():
    frame = np.zeros((H, W, 3), np.uint8)
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], np.int32)
    assert color_in_contour(frame, -50, -50, 10, contour, DEFAULT_HSV_RANGES, 0.25) is None


def test_load_hsv_calibration(tmp_path):
    path = tmp_path / "hsv.json"
    path.write_text(json.dumps({c.value: [{"lo": [0, 1, 2], "hi": [10, 20, 30]}] for c in Color}))
    ranges = load_hsv_calibration(path)
    assert ranges[Color.BLUE] == (((0, 1, 2), (10, 20, 30)),)


def test_load_hsv_calibration_needs_every_colour(tmp_path):
    path = tmp_path / "hsv.json"
    path.write_text(json.dumps({"red": []}))
    with pytest.raises(ValueError, match="missing colours"):
        load_hsv_calibration(path)


def test_overlap_with_h_landing():
    assert overlaps_h_landing(100, 100, 20, [(70, 70, 130, 130)], 0.45)
    assert not overlaps_h_landing(300, 300, 20, [(70, 70, 130, 130)], 0.45)


def test_altitude_from_diameter():
    assert altitude_from_diameter(150.0, 1.0, 500.0) == pytest.approx(3.3333, rel=1e-4)
    assert altitude_from_diameter(0.5, 1.0, 500.0) == float("inf")


def test_f_px_from_calibration(tmp_path):
    path = tmp_path / "calibration.npz"
    np.savez(path, camera_matrix=np.array([[490.0, 0, 320], [0, 510.0, 240], [0, 0, 1]]))
    assert f_px_from_calibration(path) == 500.0


def _frame_with_disc(cx, cy, r, bgr=RED_BGR):
    frame = np.full((H, W, 3), (90, 90, 90), np.uint8)
    cv2.circle(frame, (cx, cy), r, bgr, -1)
    return frame


def test_detect_returns_coloured_circle_and_pad_in_order():
    frame = _frame_with_disc(400, 300, 60)
    pad = RawInstance(
        TargetClass.H_LANDING, np.zeros((120, 160), np.float32), (10.0, 10.0, 110.5, 90.0), 0.8
    )
    circle = RawInstance(TargetClass.CIRCLE, disc_mask(400, 300, 60), (340, 240, 460, 360), 0.9)
    dets = detect([pad, circle], frame, PerceptionConfig())
    assert [d.cls for d in dets] == [TargetClass.H_LANDING, TargetClass.CIRCLE]
    assert dets[0].bbox == (10, 10, 110, 90) and dets[0].cx == 60
    assert dets[1].color is Color.RED and dets[1].conf == 0.9


def test_detect_drops_semicircles_rings_and_circles_on_the_pad():
    frame = _frame_with_disc(100, 100, 20)
    ring = np.zeros((120, 160), np.float32)
    cv2.circle(ring, (100, 80), 20, 1.0, 1)
    instances = [
        RawInstance(TargetClass.SEMICIRCLE, disc_mask(500, 300, 40), (0, 0, 1, 1), 0.9),
        RawInstance(TargetClass.CIRCLE, ring, (320, 240, 480, 400), 0.9),
        RawInstance(
            TargetClass.H_LANDING, np.zeros((120, 160), np.float32), (60, 60, 140, 140), 0.9
        ),
        RawInstance(TargetClass.CIRCLE, disc_mask(100, 100, 20), (80, 80, 120, 120), 0.9),
    ]
    dets = detect(instances, frame, PerceptionConfig())
    assert [d.cls for d in dets] == [TargetClass.H_LANDING]
