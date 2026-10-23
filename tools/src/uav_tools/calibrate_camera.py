"""uav-calibrate-camera: checkerboard calibration from photos or a recorded video.

    uav-calibrate-camera --images board_photos/ --cols 9 --rows 6 --square-mm 25
    uav-calibrate-camera --video board.mp4 --every-s 0.5 --cols 9 --rows 6 --square-mm 25

cols x rows counts INNER corners. Record the board at many positions, tilts and distances,
covering the frame corners (the wide-angle IMX219 distorts most there). Headless: no window
is needed, so it also runs on the Pi over SSH.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from uav_core.perception.camera import (
    Calibration,
    Undistorter,
    board_object_points,
    calibrate,
    find_board,
)
from uav_ml.labels import IMAGE_SUFFIXES

RMS_WARN_PX = 1.0


def frames_from_images(folder: Path) -> Iterator[np.ndarray]:
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            img = cv2.imread(str(path))
            if img is not None:
                yield img


def frames_from_video(path: Path, every_s: float) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) * every_s))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if index % step == 0:
                yield frame
            index += 1
    finally:
        cap.release()


def run(
    frames: Iterator[np.ndarray], cols: int, rows: int, square_m: float, max_views: int = 80
) -> tuple[Calibration, int, int]:
    """Returns (calibration, views used, frames scanned)."""
    obj = board_object_points(cols, rows, square_m)
    image_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    scanned = 0
    for frame in frames:
        scanned += 1
        h, w = frame.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise ValueError(f"frame size changed from {size} to {(w, h)}")
        corners = find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cols, rows)
        if corners is not None:
            image_points.append(corners)
            if len(image_points) >= max_views:
                break
    if size is None:
        raise ValueError("no frames")
    calib = calibrate([obj] * len(image_points), image_points, size)
    return calib, len(image_points), scanned


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uav-calibrate-camera", description=__doc__.split("\n\n")[0]
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, help="folder of board photos")
    src.add_argument("--video", type=Path, help="video of the board")
    parser.add_argument("--every-s", type=float, default=0.5, help="video sampling period")
    parser.add_argument("--cols", type=int, required=True, help="inner corners per row")
    parser.add_argument("--rows", type=int, required=True, help="inner corners per column")
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("calibration.npz"))
    args = parser.parse_args(argv)

    frames = (
        frames_from_images(args.images)
        if args.images
        else frames_from_video(args.video, args.every_s)
    )
    calib, used, scanned = run(frames, args.cols, args.rows, args.square_mm / 1000.0)
    calib.save(args.output)
    und = Undistorter(calib)
    print(f"{used} board views from {scanned} frames, RMS {calib.rms_px:.3f} px -> {args.output}")
    fx = calib.camera_matrix[0, 0]
    print(f"fx={fx:.1f} px; after undistortion (use this for altitude) f_px={und.f_px:.1f}")
    if calib.rms_px > RMS_WARN_PX:
        print(f"WARNING: RMS above {RMS_WARN_PX} px; recollect with more varied board poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
