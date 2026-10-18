"""
hsv_calibrator.py — Real-time HSV colour calibration tool for competition field use.

Run this at the venue 15 minutes before the match to tune HSV thresholds under
the actual lighting conditions.  The tool adjusts the three fire-circle colours
(Red, Yellow, Blue) interactively and saves the result to a JSON file that
mission.py loads at startup via load_hsv_calibration().

Usage:
    python hsv_calibrator.py --source 0 --output hsv_calib.json
    python hsv_calibrator.py --source 0 --output hsv_calib.json --load hsv_calib.json
    python hsv_calibrator.py --source sample_frame.jpg --output hsv_calib.json

Controls:
    [R]   Switch to Red   colour mode
    [Y]   Switch to Yellow colour mode
    [B]   Switch to Blue  colour mode
    [S]   Save current calibration to --output path
    [L]   Load calibration from --output path (restore saved state)
    [C]   Toggle undistorted camera feed (requires --calib)
    [Q] / ESC  Quit

Integration with mission.py:
    import mission
    mission.load_hsv_calibration("hsv_calib.json")

    # OR pass at startup via CLI:
    python test_video.py --weights best.pt --source 0 --hsv-calib hsv_calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default HSV ranges — mirrors mission.py COLOR_HSV at competition defaults
# ---------------------------------------------------------------------------

DEFAULT_RANGES: dict[str, list[dict]] = {
    "red": [
        {"lo": [0,   120,  80], "hi": [10,  255, 255]},  # lower red hue
        {"lo": [165, 120,  80], "hi": [180, 255, 255]},  # upper red hue
    ],
    "yellow": [
        {"lo": [18, 100, 100], "hi": [38, 255, 255]},
    ],
    "blue": [
        {"lo": [100, 130,  50], "hi": [145, 255, 255]},
    ],
}

# Trackbar window names
CTRL_WIN = "HSV Calibrator — Controls"
PREV_WIN = "HSV Calibrator — Preview"

# Displayed order of colours to cycle through
COLOR_CYCLE = ["red", "yellow", "blue"]

# BGR preview colours for the mask overlay
COLOR_BGR = {
    "red":    (0,   40, 220),
    "yellow": (0,  220, 220),
    "blue":   (220, 60,   0),
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CalibState:
    def __init__(self, defaults: dict[str, list[dict]]):
        import copy
        self.ranges = copy.deepcopy(defaults)
        self.active_color = "red"
        self.show_combined = False   # show all-colour mask vs single-colour

    def active_ranges(self) -> list[dict]:
        return self.ranges[self.active_color]

    def to_json(self) -> dict:
        return {c: [{"lo": list(r["lo"]), "hi": list(r["hi"])}
                    for r in ranges]
                for c, ranges in self.ranges.items()}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        print(f"[HSV] Saved → {path}")
        _print_mission_snippet(self.ranges)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        for color, ranges in raw.items():
            if color in self.ranges:
                self.ranges[color] = [{"lo": list(r["lo"]), "hi": list(r["hi"])}
                                       for r in ranges]
        print(f"[HSV] Loaded ← {path}")


# ---------------------------------------------------------------------------
# Trackbar helpers
# ---------------------------------------------------------------------------

# Trackbar names per range index (red has two ranges, yellow/blue have one)
def _tb(field: str, idx: int = 0) -> str:
    return f"[{idx}] {field}"


def _create_trackbars(win: str, state: CalibState) -> None:
    """Destroy and recreate trackbars for the currently active colour."""
    cv2.destroyWindow(win)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 400)

    color  = state.active_color
    ranges = state.active_ranges()

    for idx, r in enumerate(ranges):
        prefix = f"Range {idx+1}/" + str(len(ranges)) + "  " if len(ranges) > 1 else ""
        cv2.createTrackbar(f"{prefix}H_lo [{idx}]", win, int(r["lo"][0]), 179,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 0, v, "lo"))
        cv2.createTrackbar(f"{prefix}S_lo [{idx}]", win, int(r["lo"][1]), 255,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 1, v, "lo"))
        cv2.createTrackbar(f"{prefix}V_lo [{idx}]", win, int(r["lo"][2]), 255,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 2, v, "lo"))
        cv2.createTrackbar(f"{prefix}H_hi [{idx}]", win, int(r["hi"][0]), 179,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 0, v, "hi"))
        cv2.createTrackbar(f"{prefix}S_hi [{idx}]", win, int(r["hi"][1]), 255,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 1, v, "hi"))
        cv2.createTrackbar(f"{prefix}V_hi [{idx}]", win, int(r["hi"][2]), 255,
                           lambda v, c=color, i=idx: _set_val(state, c, i, 2, v, "hi"))
        # Separator
        if idx < len(ranges) - 1:
            cv2.createTrackbar(f"--- range {idx+2} ---", win, 0, 1,
                               lambda v: None)


def _set_val(state: CalibState, color: str, range_idx: int,
             channel: int, value: int, bound: str) -> None:
    state.ranges[color][range_idx][bound][channel] = value


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------

def _build_mask(hsv: np.ndarray, ranges: list[dict]) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    return combined


def _build_all_masks(hsv: np.ndarray, state: CalibState) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, ranges in state.ranges.items():
        combined = cv2.bitwise_or(combined, _build_mask(hsv, ranges))
    return combined


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _draw_overlay(frame: np.ndarray, mask: np.ndarray,
                  state: CalibState, match_pct: float) -> np.ndarray:
    """Blend mask over frame with colour tint, draw stats."""
    bgr_tint = np.zeros_like(frame)
    bgr_tint[:] = COLOR_BGR[state.active_color]

    tinted = cv2.bitwise_and(bgr_tint, bgr_tint, mask=mask)
    result = cv2.addWeighted(frame, 0.6, tinted, 0.5, 0)

    # Overlay mask border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, COLOR_BGR[state.active_color], 2)

    # Status bar
    h, w = result.shape[:2]
    bar_txt = (f"  [{state.active_color.upper()}]  "
               f"match: {match_pct*100:.1f}%  |  "
               f"[R]ed [Y]ellow [B]lue  [S]ave [L]oad [Q]uit")
    cv2.rectangle(result, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(result, bar_txt, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Current range values
    for idx, r in enumerate(state.active_ranges()):
        txt = (f"R{idx+1}: H[{r['lo'][0]}-{r['hi'][0]}] "
               f"S[{r['lo'][1]}-{r['hi'][1]}] V[{r['lo'][2]}-{r['hi'][2]}]")
        cv2.putText(result, txt, (8, 22 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_BGR[state.active_color], 1, cv2.LINE_AA)
    return result


# ---------------------------------------------------------------------------
# JSON → mission.py snippet printer
# ---------------------------------------------------------------------------

def _print_mission_snippet(ranges: dict) -> None:
    print("\n─── Paste this into mission.py  COLOR_HSV ──────────────────────")
    print("COLOR_HSV = {")
    for color, rlist in ranges.items():
        print(f'    "{color}": [')
        for r in rlist:
            print(f'        ({tuple(r["lo"])}, {tuple(r["hi"])}),')
        print("    ],")
    print("}")
    print("────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(source, output: str, load_path: str | None = None,
        calib_path: str | None = None) -> None:

    state = CalibState(DEFAULT_RANGES)
    if load_path and Path(load_path).exists():
        state.load(load_path)

    # Undistortion (optional)
    undist = None
    if calib_path:
        try:
            from camera_calibration import IMX219Undistorter
            undist = IMX219Undistorter(calib_path, alpha=0.0, crop=True)
            print(f"[HSV] Undistortion enabled from {calib_path}")
        except Exception as e:
            print(f"[HSV] Could not load calibration ({e}) — showing raw feed")

    # Open source
    is_image = isinstance(source, str) and Path(source).exists() and \
               Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    if is_image:
        static_frame = cv2.imread(source)
        if static_frame is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        if undist:
            static_frame = undist.correct(static_frame)
        cap = None
    else:
        src_idx = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ── Critical: Lock Auto White Balance (AWB) and Auto Exposure (AEC) ──
        #
        # WHY this matters (Threat 4 from architecture review):
        #   The competition floor has uneven lighting — the Safe Zone may be
        #   slightly dim while the fire-circle area is under direct spotlights.
        #   When the drone flies from one zone to the other, the camera's AWB
        #   and AEC algorithms shift the ENTIRE colour spectrum.  Red circles
        #   calibrated at H=0-10 in the Safe Zone may appear at H=15-25 (orange)
        #   under the spotlights, making the calibration useless mid-flight.
        #
        # Fix: lock AWB and AEC BEFORE running the calibrator, under the same
        #   lighting conditions you will fly in (i.e. at the fire-circle end of
        #   the field, not in the shade).  Keep them locked for the full mission.
        #
        # OpenCV V4L2 controls (Linux / Raspberry Pi):
        #   CAP_PROP_AUTO_WB     = 0  → disable auto white balance
        #   CAP_PROP_AUTO_EXPOSURE = 1 → manual mode (V4L2: 1=manual, 3=auto)
        #   CAP_PROP_WB_TEMPERATURE = <value> → lock to a fixed colour temperature
        #   CAP_PROP_EXPOSURE    = <value> → fixed exposure time
        #
        # Recommended procedure:
        #   1. Position the drone above F1/F2 (red circles, brightest spot).
        #   2. Let the camera auto-expose for 2-3 seconds.
        #   3. Read back the current gain/exposure with CAP_PROP_EXPOSURE.
        #   4. Lock those values.  Now run the calibrator — the HSV you save
        #      will remain valid throughout the entire flight.

        # Attempt to lock AWB and AEC via V4L2 / OpenCV backend.
        # These calls are silently ignored on platforms that don't support them
        # (e.g. macOS, Windows, non-V4L2 cameras) — no crash risk.
        awb_ok  = cap.set(cv2.CAP_PROP_AUTO_WB, 0)       # 0 = disable auto WB
        # CAP_PROP_AUTO_EXPOSURE: 1 = manual, 3 = auto (V4L2 convention)
        aec_ok  = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
        if awb_ok and aec_ok:
            print("[HSV] [OK] AWB and AEC locked — colour will be stable across the field.")
        elif awb_ok:
            print("[HSV] [OK] AWB locked. AEC lock not supported by this backend "
                  "(set exposure manually in your camera driver if possible).")
        else:
            print("[HSV] ⚠ AWB/AEC lock NOT supported by this camera backend.")
            print("      If using a Pi camera, use libcamera-still --awbgains or")
            print("      v4l2-ctl --set-ctrl=auto_white_balance=0 before running.")
            print("      Without locking, calibrated HSV may drift under spotlights.")

        static_frame = None

    # Create windows
    cv2.namedWindow(PREV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREV_WIN, 960, 540)
    _create_trackbars(CTRL_WIN, state)

    print(f"\n[HSV] Calibrator ready.  Output → {output}")
    print("  Keys:  R=Red  Y=Yellow  B=Blue  S=Save  L=Load  C=Combined  Q=Quit\n")

    while True:
        if is_image:
            frame = static_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("[HSV] Camera stream ended.")
                break
            if undist:
                frame = undist.correct(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if state.show_combined:
            mask = _build_all_masks(hsv, state)
        else:
            mask = _build_mask(hsv, state.active_ranges())

        total_px  = mask.shape[0] * mask.shape[1]
        match_pct = float(np.count_nonzero(mask)) / max(total_px, 1)

        display = _draw_overlay(frame, mask, state, match_pct)
        cv2.imshow(PREV_WIN, display)

        key = cv2.waitKey(30 if not is_image else 50) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            state.active_color = "red"
            _create_trackbars(CTRL_WIN, state)
            print("[HSV] Switched to RED")
        elif key == ord('y'):
            state.active_color = "yellow"
            _create_trackbars(CTRL_WIN, state)
            print("[HSV] Switched to YELLOW")
        elif key == ord('b'):
            state.active_color = "blue"
            _create_trackbars(CTRL_WIN, state)
            print("[HSV] Switched to BLUE")
        elif key == ord('s'):
            state.save(output)
        elif key == ord('l'):
            if Path(output).exists():
                state.load(output)
                _create_trackbars(CTRL_WIN, state)
            else:
                print(f"[HSV] No saved file at {output}")
        elif key == ord('c'):
            state.show_combined = not state.show_combined
            print(f"[HSV] Combined mask: {state.show_combined}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive HSV colour calibration for competition field use"
    )
    parser.add_argument("--source",  default="0",
                        help="Camera index (0, 1, ...) or path to image/video")
    parser.add_argument("--output",  default="hsv_calib.json",
                        help="Path to save/load calibration JSON (default: hsv_calib.json)")
    parser.add_argument("--load",    default=None,
                        help="Load existing calibration on startup")
    parser.add_argument("--calib",   default=None,
                        help="calibration.npz for IMX219 undistortion (optional)")
    args = parser.parse_args()

    run(
        source     = args.source,
        output     = args.output,
        load_path  = args.load or (args.output if Path(args.output).exists() else None),
        calib_path = args.calib,
    )
