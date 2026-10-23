import cv2
import numpy as np
import pytest
from uav_core.perception.camera import (
    Calibration,
    Undistorter,
    board_object_points,
    calibrate,
    find_board,
)

W, H = 640, 480
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
DIST = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
COLS, ROWS, SQUARE = 9, 6, 0.025


def _views():
    """Project a board seen from 12 poses through K and DIST (synthetic calibration)."""
    obj = board_object_points(COLS, ROWS, SQUARE)
    rng = np.random.default_rng(0)
    objs, imgs = [], []
    for _ in range(12):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([rng.uniform(-0.12, 0.02), rng.uniform(-0.08, 0.0), rng.uniform(0.35, 0.5)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
        objs.append(obj)
        imgs.append(pts.astype(np.float32))
    return objs, imgs


def test_calibration_recovers_the_intrinsics():
    calib = calibrate(*_views(), (W, H))
    assert calib.rms_px < 1e-3
    np.testing.assert_allclose(calib.camera_matrix, K, atol=0.5)
    np.testing.assert_allclose(calib.dist_coeffs.ravel()[:2], DIST[:2], atol=1e-3)


def test_too_few_views_is_an_error():
    objs, imgs = _views()
    with pytest.raises(ValueError, match=">= 10"):
        calibrate(objs[:3], imgs[:3], (W, H))


def test_save_load_round_trip(tmp_path):
    calib = Calibration(K, DIST, (W, H), 0.3)
    calib.save(tmp_path / "c.npz")
    back = Calibration.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.camera_matrix, K)
    assert back.frame_size == (W, H) and back.rms_px == pytest.approx(0.3)


def test_undistorter_straightens_lines_and_reports_the_new_focal_length():
    und = Undistorter(Calibration(K, DIST, (W, H), 0.0))
    assert und.f_px != K[0, 0]  # alpha = 0 zooms in: altitude must use this value
    # A straight world line, distorted by the lens, comes back straight.
    line = np.array([[x, 0.3, 1.0] for x in np.linspace(-0.5, 0.5, 9)], np.float32)
    distorted, _ = cv2.projectPoints(line, np.zeros(3), np.zeros(3), K, DIST)
    frame = np.zeros((H, W, 3), np.uint8)
    for x, y in distorted.reshape(-1, 2):
        cv2.circle(frame, (round(float(x)), round(float(y))), 3, (255, 255, 255), -1)

    def curvature(img):
        ys, xs = np.nonzero(img[:, :, 0] > 128)
        return abs(np.polyfit(xs, ys, 2)[0])

    assert curvature(und(frame)) < curvature(frame) / 10


def test_undistorter_rejects_other_frame_sizes():
    und = Undistorter(Calibration(K, DIST, (W, H), 0.0))
    with pytest.raises(ValueError, match="1280x720"):
        und(np.zeros((720, 1280, 3), np.uint8))


def test_find_board_on_a_rendered_checkerboard():
    sq = 40
    board = np.kron((np.indices((ROWS + 1, COLS + 1)).sum(0) % 2) * 255, np.ones((sq, sq)))
    img = np.full((H, W), 255, np.uint8)
    img[20 : 20 + board.shape[0], 20 : 20 + board.shape[1]] = board
    corners = find_board(img, COLS, ROWS)
    assert corners is not None and corners.shape == (COLS * ROWS, 1, 2)
    assert find_board(np.full((H, W), 128, np.uint8), COLS, ROWS) is None
