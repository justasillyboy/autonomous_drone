import csv
import json

import cv2
import numpy as np
import pytest
from uav_core.config import DEFAULT_HSV_RANGES, MissionConfig, PerceptionConfig
from uav_core.perception import RawInstance
from uav_core.perception.camera import Calibration
from uav_core.perception.color import load_hsv_calibration
from uav_core.types import Color, Phase, TargetClass
from uav_tools import calibrate_camera, hsv_cli, run_video
from uav_tools.hsv import classify_sample, fit_ranges, hue_ranges, in_ranges, save_hsv_json
from uav_tools.yolo_labels import read_label_file

# -- HSV ----------------------------------------------------------------------------------


def _pixels(h, s=200, v=200, n=500, spread=3, seed=0):
    rng = np.random.default_rng(seed)
    hs = (h + rng.integers(-spread, spread + 1, n)) % 180
    return np.stack([hs, np.full(n, s), np.full(n, v)], axis=1).astype(np.uint8)


@pytest.mark.parametrize(
    ("hue", "expected"), [(2, Color.RED), (176, Color.RED), (30, Color.YELLOW), (120, Color.BLUE)]
)
def test_classify_by_hue_vote(hue, expected):
    assert classify_sample(_pixels(hue)) is expected


def test_pale_or_off_hue_pixels_have_no_colour():
    assert classify_sample(_pixels(30, s=5)) is None
    assert classify_sample(_pixels(75)) is None  # green: no window


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (20, 40, (((20, 1, 2), (40, 255, 255)),)),
        (-8, 6, (((0, 1, 2), (6, 255, 255)), ((172, 1, 2), (180, 255, 255)))),
        (-16, -4, (((164, 1, 2), (176, 255, 255)),)),  # all on the high side: one range
        (170, 185, (((0, 1, 2), (5, 255, 255)), ((170, 1, 2), (180, 255, 255)))),
        (181, 185, (((1, 1, 2), (5, 255, 255)),)),
    ],
)
def test_hue_ranges_handle_the_wrap(lo, hi, expected):
    assert hue_ranges(lo, hi, 1, 2) == expected


def test_fit_recovers_shifted_colours_and_respects_the_background():
    # The camera sees red at 170, yellow pale (S 60) at 25, blue purple-ish at 135.
    circles = [_pixels(170, seed=1), _pixels(25, s=60, seed=2), _pixels(132, seed=3)]
    floor = _pixels(140, s=40, v=150, n=20000, spread=4, seed=4)  # purple floor, low S
    res = fit_ranges(circles, DEFAULT_HSV_RANGES, floor)
    ranges = PerceptionConfig(hsv_ranges=res.ranges).hsv_ranges  # valid config
    for px, color in zip(circles, (Color.RED, Color.YELLOW, Color.BLUE), strict=True):
        assert in_ranges(px, ranges[color]).mean() > 0.9
    assert in_ranges(floor, ranges[Color.BLUE]).mean() <= 0.01
    assert res.samples == {Color.RED: 1, Color.YELLOW: 1, Color.BLUE: 1}


def test_background_never_pushes_s_above_the_circles():
    circle = _pixels(30, s=90)
    unlabelled_twin = _pixels(30, s=90, n=5000, seed=9)  # same colour, counted as background
    res = fit_ranges([circle], DEFAULT_HSV_RANGES, unlabelled_twin)
    assert in_ranges(circle, res.ranges[Color.YELLOW]).mean() >= 0.8


def test_colours_without_samples_keep_the_defaults():
    res = fit_ranges([_pixels(30)], DEFAULT_HSV_RANGES)
    assert res.ranges[Color.RED] == DEFAULT_HSV_RANGES[Color.RED]
    assert res.defaulted == (Color.RED, Color.BLUE)


def test_json_round_trips_through_the_uav_core_loader(tmp_path):
    save_hsv_json(DEFAULT_HSV_RANGES, tmp_path / "h.json")
    assert load_hsv_calibration(tmp_path / "h.json") == DEFAULT_HSV_RANGES
    assert json.loads((tmp_path / "h.json").read_text())["red"][0] == {
        "lo": [0, 120, 80],
        "hi": [10, 255, 255],
    }


# -- YOLO labels + `uav-hsv fit` -----------------------------------------------------------


def _square_label(cls, x1, y1, x2, y2, w, h):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in pts)


def test_label_polygons_become_masks(tmp_path):
    (tmp_path / "a.txt").write_text(_square_label(0, 10, 20, 30, 40, 100, 50) + "\n2 0.1 0.1\n")
    (inst,) = read_label_file(tmp_path / "a.txt", 100, 50)
    assert inst.class_id == 0 and inst.mask[30, 20] and not inst.mask[5, 5]


def _write_dataset(root, n=6):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir()
    bgr = {Color.RED: (60, 30, 200), Color.YELLOW: (150, 230, 235), Color.BLUE: (200, 60, 90)}
    for k in range(n):
        img = np.full((120, 160, 3), (120, 90, 110), np.uint8)  # dull purple floor
        lines = []
        for j, color in enumerate(Color):
            x = 10 + 50 * j
            cv2.circle(img, (x + 15, 60), 15, bgr[color], -1)
            lines.append(_square_label(0, x, 45, x + 31, 76, 160, 120))
        cv2.imwrite(str(root / "train" / "images" / f"f{k:02d}.png"), img)
        (root / "train" / "labels" / f"f{k:02d}.txt").write_text("\n".join(lines))


def test_hsv_fit_cli_writes_a_loadable_file_and_reports(tmp_path, capsys):
    _write_dataset(tmp_path / "ds")
    out = tmp_path / "hsv.json"
    assert hsv_cli.main(["fit", "--dataset", str(tmp_path / "ds"), "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "fit on 5 images, 15 circles (red 5, yellow 5, blue 5)" in text
    assert "held-out evaluation: last 1 images" in text
    fitted = next(line for line in text.splitlines() if line.startswith("fitted"))
    assert "100.0%" in fitted
    assert set(load_hsv_calibration(out)) == set(Color)


def test_hsv_fit_cli_without_circles(tmp_path, capsys):
    (tmp_path / "images").mkdir()
    assert hsv_cli.main(["fit", "--dataset", str(tmp_path), "--output", str(tmp_path / "x")]) == 1


# -- camera calibration --------------------------------------------------------------------


def _board_views(n=12, size=(640, 480)):
    sq, cols, rows = 30, 9, 6
    board = np.kron((np.indices((rows + 1, cols + 1)).sum(0) % 2) * 255, np.ones((sq, sq)))
    board = np.pad(board.astype(np.uint8), sq, constant_values=255)
    bh, bw = board.shape
    rng = np.random.default_rng(0)
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    for _ in range(n):
        jitter = rng.uniform(-40, 40, (4, 2))
        dst = np.float32([[150, 100], [490, 110], [480, 370], [160, 380]]) + jitter
        h = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        gray = cv2.warpPerspective(board, h, size, borderValue=255)
        yield cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_calibrate_camera_from_images(tmp_path, capsys):
    folder = tmp_path / "board"
    folder.mkdir()
    for k, frame in enumerate(_board_views()):
        cv2.imwrite(str(folder / f"{k:02d}.png"), frame)
    out = tmp_path / "calib.npz"
    args = ["--images", str(folder), "--cols", "9", "--rows", "6", "--square-mm", "30"]
    assert calibrate_camera.main([*args, "--output", str(out)]) == 0
    assert "12 board views from 12 frames" in capsys.readouterr().out
    assert Calibration.load(out).frame_size == (640, 480)


def test_calibrate_camera_from_video(tmp_path):
    path = tmp_path / "board.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for frame in _board_views():
        for _ in range(5):
            writer.write(frame)
    writer.release()
    frames = calibrate_camera.frames_from_video(path, every_s=0.5)
    calib, used, scanned = calibrate_camera.run(frames, 9, 6, 0.03)
    # Every 5th frame of 60 is sampled; MJPG artefacts make the fit itself loose.
    assert (scanned, used) == (12, 12) and calib.frame_size == (640, 480)


def test_calibration_rejects_mixed_frame_sizes():
    frames = iter([np.zeros((480, 640, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8)])
    with pytest.raises(ValueError, match="size changed"):
        calibrate_camera.run(frames, 9, 6, 0.03)


def test_missing_video_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        next(calibrate_camera.frames_from_video(tmp_path / "none.mp4", 0.5))


# -- video replay ---------------------------------------------------------------------------


class _FakeBackend:
    """A red disc of radius 60 px at the frame centre in every frame."""

    def infer(self, frame):
        h, w = frame.shape[:2]
        yy, xx = np.mgrid[: h // 4, : w // 4]
        mask = ((xx - w // 8) ** 2 + (yy - h // 8) ** 2 <= 15**2).astype(np.float32)
        x, y = w / 2, h / 2
        return [RawInstance(TargetClass.CIRCLE, mask, (x - 60, y - 60, x + 60, y + 60), 0.9)]


def _red_frames(n, w=640, h=480):
    for _ in range(n):
        img = np.full((h, w, 3), 100, np.uint8)
        cv2.circle(img, (w // 2, h // 2), 60, (0, 0, 220), -1)
        yield img


def test_replay_drops_at_once_on_a_centred_target_at_drop_altitude():
    rows = [
        row
        for _, row in run_video.replay(
            _red_frames(90), 30.0, _FakeBackend(), MissionConfig(),
            run_video.ReplayOptions(altitude_m=3.4),
        )
    ]  # fmt: skip
    assert rows[0]["detections"] == 1
    assert rows[0]["action"] == "DROP" and rows[0]["drop"] == 1
    assert rows[0]["phase"] == Phase.SEARCHING.value  # target scored: on to the next one
    assert rows[-1]["balls"] == 1 and rows[-1]["alt_source"] == "FC"


def test_cli_writes_video_and_log(tmp_path, monkeypatch, capsys):
    src = tmp_path / "in.avi"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for frame in _red_frames(20):
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(run_video, "load_backend", lambda weights, conf: _FakeBackend())
    calib = tmp_path / "c.npz"
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    Calibration(k, np.zeros(5), (640, 480), 0.2).save(calib)
    save_hsv_json(DEFAULT_HSV_RANGES, tmp_path / "h.json")
    (tmp_path / "m.yaml").write_text("drop:\n  align_pixel_thresh_px: 30\n")
    args = [
        "--weights", "x.onnx", "--source", str(src), "--output", str(tmp_path / "o.mp4"),
        "--log", str(tmp_path / "o.csv"), "--calib", str(calib), "--hsv-calib",
        str(tmp_path / "h.json"), "--config", str(tmp_path / "m.yaml"), "--altitude", "3.4",
        "--max-frames", "15",
    ]  # fmt: skip
    assert run_video.main(args) == 0
    assert "15 frames" in capsys.readouterr().out
    with (tmp_path / "o.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15 and rows[0]["action"] == "DROP"
    assert cv2.VideoCapture(str(tmp_path / "o.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) == 15


def test_build_config_applies_overrides(tmp_path):
    save_hsv_json(DEFAULT_HSV_RANGES, tmp_path / "h.json")
    cfg = run_video.build_config(None, tmp_path / "h.json", 487.5)
    assert cfg.camera.f_px == 487.5 and cfg.perception.hsv_ranges == DEFAULT_HSV_RANGES
