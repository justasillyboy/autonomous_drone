"""Camera intrinsics: calibration file, precomputed undistortion, focal length.

The .npz layout is the legacy one (camera_matrix, dist_coeffs, frame_size, rms), so old
calibration files still load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_GOOD_VIEWS = 10


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray
    frame_size: tuple[int, int]  # (width, height) the calibration was made at
    rms_px: float

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        with np.load(path) as data:
            return cls(
                camera_matrix=data["camera_matrix"].astype(np.float64),
                dist_coeffs=data["dist_coeffs"].astype(np.float64),
                frame_size=tuple(int(v) for v in data["frame_size"]),
                rms_px=float(data["rms"]),
            )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            frame_size=np.array(self.frame_size),
            rms=np.array(self.rms_px),
        )


def calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    frame_size: tuple[int, int],
) -> Calibration:
    """cv2.calibrateCamera over at least MIN_GOOD_VIEWS checkerboard views."""
    if len(image_points) < MIN_GOOD_VIEWS:
        raise ValueError(f"need >= {MIN_GOOD_VIEWS} board views, got {len(image_points)}")
    rms, k, dist, _, _ = cv2.calibrateCamera(object_points, image_points, frame_size, None, None)
    return Calibration(k, dist, frame_size, float(rms))


def board_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """3-D inner-corner positions of a checkerboard on z = 0, in metres."""
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def find_board(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Sub-pixel inner corners of a cols x rows checkerboard, or None."""
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


class Undistorter:
    """Remaps frames with maps computed once (cv2.remap per frame is the cheap part).

    alpha = 0 keeps only valid pixels, so the output is zoomed in slightly. The focal length
    of that output is `f_px` (from the new camera matrix), which is what the pinhole altitude
    estimate needs. The legacy code used the original matrix's fx instead, which biases the
    vision altitude by the zoom factor.
    """

    def __init__(self, calib: Calibration, alpha: float = 0.0):
        w, h = calib.frame_size
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            calib.camera_matrix, calib.dist_coeffs, (w, h), alpha, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            calib.camera_matrix, calib.dist_coeffs, None, new_k, (w, h), cv2.CV_16SC2
        )
        self.frame_size = (w, h)
        self.camera_matrix = new_k
        self.f_px = float(new_k[0, 0])

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.frame_size:
            raise ValueError(f"frame is {w}x{h}, calibration is {self.frame_size}")
        return cv2.remap(frame_bgr, self._map1, self._map2, cv2.INTER_LINEAR)
