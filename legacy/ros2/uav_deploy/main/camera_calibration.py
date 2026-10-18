"""
camera_calibration.py — IMX219 wide-angle lens calibration & undistortion.

Corrects barrel distortion from wide-angle / fisheye lenses commonly paired
with the IMX219 sensor (Raspberry Pi Camera v2, Arducam wide-angle variants).
As visible in check_right.png, the drone camera exhibits significant radial
distortion that warps circle shapes at frame edges.

Two-stage workflow:
  Stage 1 — CALIBRATE: Capture checkerboard frames, compute camera matrix
            and distortion coefficients, save to calibration.npz
  Stage 2 — UNDISTORT: Load saved coefficients, apply to any image or frame

Usage:
    # Stage 1 — collect frames from camera and calibrate
    python camera_calibration.py calibrate --source 0 --rows 9 --cols 6 \
           --square-mm 25.0 --output calibration.npz

    # Stage 1 — calibrate from a folder of pre-captured checkerboard images
    python camera_calibration.py calibrate --source calib_imgs/ --rows 9 --cols 6 \
           --square-mm 25.0 --output calibration.npz

    # Stage 2 — undistort a single image
    python camera_calibration.py undistort --calib calibration.npz \
           --input distorted.jpg --output corrected.jpg

    # Stage 2 — undistort a video
    python camera_calibration.py undistort --calib calibration.npz \
           --input flight.mp4 --output corrected.mp4

    # Quick test — preview live undistortion from camera
    python camera_calibration.py preview --calib calibration.npz --source 0
"""

import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# IMX219 sensor defaults
# (override with --rows / --cols / --square-mm for your actual checkerboard)
# ---------------------------------------------------------------------------

DEFAULT_ROWS    = 9       # inner corners along height
DEFAULT_COLS    = 6       # inner corners along width
DEFAULT_SQUARE  = 25.0    # mm (print-and-measure your board)
MIN_GOOD_FRAMES = 20      # minimum frames with detected checkerboard


# ---------------------------------------------------------------------------
# Stage 1 — Calibration
# ---------------------------------------------------------------------------

def collect_and_calibrate(
    source,
    rows: int,
    cols: int,
    square_mm: float,
    output_path: str,
    capture_interval_s: float = 0.8,
    max_frames: int = 80,
) -> dict:
    """
    Collect checkerboard frames from a camera or image folder, then
    compute and save calibration coefficients.

    Args:
        source           : Camera index (int) OR path to image folder (str).
        rows, cols       : Inner corner count of the calibration board.
        square_mm        : Physical size of one square in millimetres.
        output_path      : Where to save the .npz calibration file.
        capture_interval_s: Seconds between auto-captures (live camera only).
        max_frames       : Stop collecting after this many good frames.

    Returns:
        Dict with camera_matrix, dist_coeffs, rms_error.
    """
    board_size   = (cols, rows)
    obj_pts_base = np.zeros((rows * cols, 3), np.float32)
    obj_pts_base[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points_all = []   # 3D world points
    img_points_all = []   # 2D image points
    frame_size     = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ── From image folder ────────────────────────────────────────────────────
    if isinstance(source, str) and Path(source).is_dir():
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        files    = []
        for p in patterns:
            files.extend(glob.glob(str(Path(source) / p)))
        files.sort()
        print(f"Found {len(files)} images in {source}")

        for fpath in files:
            img  = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_size = (gray.shape[1], gray.shape[0])

            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points_all.append(obj_pts_base.copy())
                img_points_all.append(corners_sub)
                print(f"  OK: {Path(fpath).name}")
            else:
                print(f"  SKIP (no board): {Path(fpath).name}")

            if len(obj_points_all) >= max_frames:
                break

    # ── From live camera ─────────────────────────────────────────────────────
    else:
        cam_idx = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {cam_idx}")

        # IMX219 common resolutions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Calibration mode — hold checkerboard in different positions.")
        print("Press [SPACE] to capture manually, [Q] to finish.\n")

        last_capture = 0.0
        saved = Path("calib_frames")
        saved.mkdir(exist_ok=True)

        while len(obj_points_all) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, board_size, None)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, board_size, corners, found)

            count_txt = f"Captured: {len(obj_points_all)}/{max_frames}"
            cv2.putText(display, count_txt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("IMX219 Calibration", display)

            key  = cv2.waitKey(1) & 0xFF
            now  = time.time()
            auto = (now - last_capture) >= capture_interval_s

            if found and (key == ord(' ') or auto):
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points_all.append(obj_pts_base.copy())
                img_points_all.append(corners_sub)
                last_capture = now
                fname = saved / f"frame_{len(obj_points_all):04d}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"  Captured frame {len(obj_points_all)}")

            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── Compute calibration ──────────────────────────────────────────────────
    n_good = len(obj_points_all)
    if n_good < MIN_GOOD_FRAMES:
        raise RuntimeError(
            f"Only {n_good} good frames — need at least {MIN_GOOD_FRAMES}. "
            "Recollect with more varied board positions."
        )

    print(f"\nCalibrating from {n_good} frames ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all, img_points_all, frame_size, None, None
    )
    print(f"Calibration RMS reprojection error: {rms:.4f} px")
    if rms > 1.0:
        print("WARNING: RMS > 1.0 px — consider recollecting with better coverage.")

    # ── Per-image reprojection errors ────────────────────────────────────────
    errors = []
    for i in range(n_good):
        proj, _ = cv2.projectPoints(
            obj_points_all[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(img_points_all[i], proj, cv2.NORM_L2) / len(proj)
        errors.append(err)
    print(f"Per-frame reprojection error: mean={np.mean(errors):.4f} "
          f"max={np.max(errors):.4f} px")

    # ── Save ────────────────────────────────────────────────────────────────
    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        frame_size=np.array(frame_size),
        rms=np.array(rms),
    )
    print(f"Calibration saved -> {output_path}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs":   dist_coeffs,
        "frame_size":    frame_size,
        "rms":           rms,
    }


# ---------------------------------------------------------------------------
# Stage 2 — Undistortion
# ---------------------------------------------------------------------------

def load_calibration(calib_path: str) -> tuple:
    """
    Load camera matrix and distortion coefficients from a .npz file.

    Returns:
        (camera_matrix, dist_coeffs, frame_size)
    """
    data = np.load(calib_path)
    cam_mat    = data["camera_matrix"]
    dist_coeff = data["dist_coeffs"]
    frame_size = tuple(data["frame_size"].tolist())
    print(f"Loaded calibration from {calib_path}  (RMS={float(data['rms']):.4f} px)")
    return cam_mat, dist_coeff, frame_size


def build_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_size: tuple,
    alpha: float = 0.0,
) -> tuple:
    """
    Pre-compute undistortion + rectification maps for fast per-frame remapping.

    Args:
        camera_matrix : 3x3 intrinsic matrix from calibration.
        dist_coeffs   : Distortion coefficients from calibration.
        frame_size    : (width, height) of the frames to undistort.
        alpha         : 0.0 = crop to valid pixels (no black borders),
                        1.0 = keep all original pixels (black borders appear).
                        For drone vision, 0.0 is preferred (no dead zones).

    Returns:
        (map1, map2, new_camera_matrix, roi)
    """
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, frame_size, alpha, frame_size
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_cam_mat,
        frame_size, cv2.CV_16SC2
    )
    return map1, map2, new_cam_mat, roi


def undistort_image(
    img: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    roi: tuple | None = None,
    crop: bool = True,
) -> np.ndarray:
    """
    Apply pre-computed undistortion maps to a single image.

    Args:
        img   : BGR image to correct.
        map1, map2 : Pre-computed remap tables from build_undistort_maps().
        roi   : Region of interest returned by getOptimalNewCameraMatrix.
        crop  : If True, crop to ROI (removes black borders after undistortion).

    Returns:
        Undistorted BGR image.
    """
    dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    if crop and roi is not None:
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y + h, x:x + w]
    return dst


def undistort_file(
    input_path: str,
    output_path: str,
    calib_path: str,
    alpha: float = 0.0,
    crop: bool = True,
) -> None:
    """
    Undistort an image or video file and save the result.

    Automatically detects file type from extension.
    """
    cam_mat, dist_coeff, calib_size = load_calibration(calib_path)

    ext = Path(input_path).suffix.lower()
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if is_video:
        _undistort_video(input_path, output_path, cam_mat, dist_coeff, calib_size, alpha, crop)
    else:
        _undistort_single_image(input_path, output_path, cam_mat, dist_coeff, calib_size, alpha, crop)


def _undistort_single_image(input_path, output_path, cam_mat, dist_coeff, calib_size, alpha, crop):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    h, w = img.shape[:2]
    frame_size = (w, h)

    # Recompute maps for this image's resolution (may differ from calibration res)
    map1, map2, _, roi = build_undistort_maps(cam_mat, dist_coeff, frame_size, alpha)
    corrected = undistort_image(img, map1, map2, roi, crop)

    cv2.imwrite(output_path, corrected)
    print(f"Saved undistorted image -> {output_path}  ({w}x{h} -> {corrected.shape[1]}x{corrected.shape[0]})")


def _undistort_video(input_path, output_path, cam_mat, dist_coeff, calib_size, alpha, crop):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    map1, map2, _, roi = build_undistort_maps(cam_mat, dist_coeff, (w, h), alpha)

    # Determine output size after crop
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_out = undistort_image(dummy, map1, map2, roi, crop)
    out_h, out_w = dummy_out.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    print(f"Undistorting video {total} frames ({w}x{h} -> {out_w}x{out_h}) ...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        corrected = undistort_image(frame, map1, map2, roi, crop)
        writer.write(corrected)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames")

    cap.release()
    writer.release()
    print(f"Saved undistorted video -> {output_path}")


# ---------------------------------------------------------------------------
# Integration helper — use inside postprocess.py / test_video.py
# ---------------------------------------------------------------------------

class IMX219Undistorter:
    """
    Drop-in undistorter for use inside postprocess.py and test_video.py.

    Usage in test_video.py:
        from camera_calibration import IMX219Undistorter
        undist = IMX219Undistorter("calibration.npz")

        while cap.isOpened():
            ret, frame = cap.read()
            frame = undist.correct(frame)   # <-- add this one line
            results = model(frame, verbose=False)
            ...
    """

    def __init__(self, calib_path: str, alpha: float = 0.0, crop: bool = False):
        """
        Args:
            calib_path : Path to calibration.npz produced by this script.
            alpha      : 0.0 = crop to valid area; 1.0 = full image with borders.
            crop       : If True, crop to ROI after undistortion.
                         Keep False for video pipelines to preserve frame size.
        """
        self.cam_mat, self.dist_coeff, self.calib_size = load_calibration(calib_path)
        self.alpha  = alpha
        self.crop   = crop
        self._maps  = {}   # cache maps per (w, h) — handles resolution changes

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a single frame. Maps are cached per resolution.

        Args:
            frame : BGR frame from cv2.VideoCapture or cv2.imread.

        Returns:
            Undistorted frame (same dtype, similar size).
        """
        h, w = frame.shape[:2]
        key  = (w, h)
        if key not in self._maps:
            map1, map2, _, roi = build_undistort_maps(
                self.cam_mat, self.dist_coeff, (w, h), self.alpha
            )
            self._maps[key] = (map1, map2, roi)

        map1, map2, roi = self._maps[key]
        return undistort_image(frame, map1, map2, roi, self.crop)


# ---------------------------------------------------------------------------
# Live preview mode
# ---------------------------------------------------------------------------

def preview_undistortion(calib_path: str, source=0):
    """
    Show live side-by-side: original (distorted) | corrected.
    Useful for verifying calibration quality before deployment.
    Press [Q] to quit.
    """
    cam_mat, dist_coeff, calib_size = load_calibration(calib_path)
    undist = IMX219Undistorter(calib_path, alpha=0.0, crop=False)

    cam_idx = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {cam_idx}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Preview mode — press [Q] to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected = undist.correct(frame)
        h = min(frame.shape[0], corrected.shape[0])

        # Resize both to same height for side-by-side display
        orig_disp = cv2.resize(frame,     (int(frame.shape[1] * h / frame.shape[0]),     h))
        corr_disp = cv2.resize(corrected, (int(corrected.shape[1] * h / corrected.shape[0]), h))

        cv2.putText(orig_disp, "Original (distorted)",   (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(corr_disp, "Corrected (undistorted)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack([orig_disp, corr_disp])
        cv2.imshow("IMX219 Undistortion Preview", combined)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IMX219 camera calibration and undistortion"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # calibrate
    cal = sub.add_parser("calibrate", help="Collect frames and compute calibration")
    cal.add_argument("--source",    default="0",    help="Camera index or image folder")
    cal.add_argument("--rows",      type=int, default=DEFAULT_ROWS)
    cal.add_argument("--cols",      type=int, default=DEFAULT_COLS)
    cal.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE)
    cal.add_argument("--output",    default="calibration.npz")
    cal.add_argument("--max-frames", type=int, default=80)

    # undistort
    und = sub.add_parser("undistort", help="Apply calibration to image or video")
    und.add_argument("--calib",  required=True, help="Path to calibration.npz")
    und.add_argument("--input",  required=True, help="Input image or video path")
    und.add_argument("--output", required=True, help="Output path")
    und.add_argument("--alpha",  type=float, default=0.0,
                     help="0.0=crop valid area (default), 1.0=full frame with borders")
    und.add_argument("--no-crop", action="store_true", help="Do not crop to valid ROI")

    # preview
    prev = sub.add_parser("preview", help="Live side-by-side undistortion preview")
    prev.add_argument("--calib",  required=True)
    prev.add_argument("--source", default="0")

    args = parser.parse_args()

    if args.command == "calibrate":
        collect_and_calibrate(
            source=args.source,
            rows=args.rows,
            cols=args.cols,
            square_mm=args.square_mm,
            output_path=args.output,
            max_frames=args.max_frames,
        )

    elif args.command == "undistort":
        undistort_file(
            input_path=args.input,
            output_path=args.output,
            calib_path=args.calib,
            alpha=args.alpha,
            crop=not args.no_crop,
        )

    elif args.command == "preview":
        preview_undistortion(calib_path=args.calib, source=args.source)


if __name__ == "__main__":
    main()
