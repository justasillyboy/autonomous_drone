"""
postprocess.py — Single-image inference for circle / semicircle / H_landing.

Used for debugging individual frames and annotating static images.
For video + full mission logic, use test_video.py instead.

All thresholds, filters, geometry helpers and drawing utilities are imported
from mission.py — single source of truth. Any constant changed in mission.py
is automatically reflected here.

Key differences vs previous version:
  - Uses _extract_geometry() (contours on small mask, scale coords up) instead
    of cv2.resize(mask, full_frame) which was the slow path on Pi 5.
  - Circles use AltitudeEstimator (Z = f_px × D_real / d_px) instead of the
    old DIAMETER_RATIO_THRESHOLD = 0.70 which violated the 3.0m altitude rule.
  - Outputs a FlightCommand dataclass (same interface as test_video.py).
  - H_landing still uses d_ratio for approach — visually appropriate for a
    static single-frame debug tool.

Usage:
    python postprocess.py --weights best.pt --image frame.jpg --output out.jpg
    python postprocess.py --weights best.pt --image frame.jpg --output out.jpg \\
                          --calib calibration.npz --conf 0.4 --finals
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

# ── Import shared logic from mission.py — single source of truth ─────────────
# NOTE: Only functions/classes are imported by name. All YAML-tunable constants
# (DROP_ALTITUDE_MIN, PX_TO_MS_FIRE, etc.) are accessed as `mission.X` inside
# main() AFTER load_mission_config_yaml() runs — importing them by name here
# would bind them at import time and silently ignore --config-yaml overrides.
import mission
from mission import (
    # Pure functions (no value binding issue)
    collect_h_landing_bboxes,
    overlaps_h_landing,
    compute_diameter_ratio,
    AltitudeEstimator,
    # Competition targets (not YAML-tunable)
    FIRE_REAL_DIAMETERS,
    # Geometry / color / drawing helpers
    _extract_geometry,
    color_of_mask,
    _decide_action,
    _draw_ratio_bar,
    _draw_altitude_bar,
    _color_bgr,
    # Dataclass (reference, not value)
    FlightCommand,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-image postprocessing for YOLO11-seg"
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--image",   required=True, help="Input image path")
    parser.add_argument("--output",  required=True, help="Output image path")
    parser.add_argument("--conf",    type=float, default=0.5)
    parser.add_argument("--calib",   default=None,
                        help="calibration.npz — enables undistortion + real f_px")
    parser.add_argument("--f-px",    type=float, default=None,
                        help="Override focal length in pixels (default: from calib or 500)")
    parser.add_argument("--finals",  action="store_true",
                        help="Use Finals target sizes / scoring labels")
    parser.add_argument("--hsv-calib", default=None,
                        help="HSV calibration JSON produced by hsv_calibrator.py. "
                             "Run hsv_calibrator.py at the venue before the match "
                             "to tune colour thresholds under actual lighting.")
    parser.add_argument("--config-yaml", default=None,
                        help="Path to config.yaml for on-field constant tuning. "
                             "Overrides mission.py defaults without editing source. "
                             "Example: --config-yaml config.yaml")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    model  = YOLO(args.weights)

    # ── HSV colour calibration (optional, recommended for competition) ────────
    if args.config_yaml:
        from mission import load_mission_config_yaml
        load_mission_config_yaml(args.config_yaml)

    if args.hsv_calib:
        from mission import load_hsv_calibration
        try:
            load_hsv_calibration(args.hsv_calib)
            print(f"[HSV] Colour calibration loaded from {args.hsv_calib}")
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[HSV] WARNING: could not load {args.hsv_calib}: {e} — using built-in defaults")

    # ── Undistortion + focal length ───────────────────────────────────────────
    undist = None
    f_px   = args.f_px
    if args.calib:
        from camera_calibration import IMX219Undistorter
        undist = IMX219Undistorter(args.calib, alpha=0.0, crop=True)
        if f_px is None:
            est  = AltitudeEstimator.from_calibration(args.calib)
            f_px = est.f_px
        print(f"[Calib] f_px={f_px:.1f} from {args.calib}")
    if f_px is None:
        f_px = mission.DEFAULT_F_PX

    alt_est = AltitudeEstimator(f_px=f_px)

    # ── Load and undistort image ───────────────────────────────────────────────
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    if undist is not None:
        img = undist.correct(img)

    frame_h, frame_w = img.shape[:2]

    # ── Inference ─────────────────────────────────────────────────────────────
    result = model(img, conf=args.conf)[0]
    names  = model.names   # {0: 'circle', 1: 'semicircle', 2: 'H_landing'}

    # ── Cross-class NMS prep ──────────────────────────────────────────────────
    h_bboxes = collect_h_landing_bboxes(result, names)

    # ── Per-detection processing ──────────────────────────────────────────────
    if result.masks is not None and result.boxes is not None:
        for mask_obj, box in zip(result.masks.data, result.boxes):
            cls_id     = int(box.cls[0].item())
            class_name = names.get(cls_id, "")
            conf_score = float(box.conf[0].item())

            # ── Optimized geometry: contours on small mask, scale up ──────────
            # (avoids cv2.resize to full frame — critical on Pi 5 / Hailo-8L)
            geo = _extract_geometry(mask_obj, frame_w, frame_h)
            if geo is None:
                continue
            cx, cy, radius, fr, cnt_frame, edge_clipped = geo

            # ── Circle ───────────────────────────────────────────────────────
            if class_name == "circle":
                # Filter 1: fill ratio (rejects court arcs, hollow rings)
                if fr < mission.FILL_RATIO_MIN_CIRCLE:
                    cv2.drawContours(img, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                # Filter 2: size cap (rejects giant court boundary lines)
                d_ratio = compute_diameter_ratio(radius, frame_w)
                if d_ratio > mission.MAX_FIRE_DIAMETER_RATIO:
                    cv2.drawContours(img, [cnt_frame], -1, (40, 40, 40), 1)
                    continue

                # Filter 3: cross-class NMS (rejects circles inside H_landing)
                if h_bboxes and overlaps_h_landing(cx, cy, radius, h_bboxes):
                    continue

                # Colour detection
                detected_color = color_of_mask(
                    _roi_mask(img, cnt_frame, cx, cy, radius, frame_w, frame_h),
                    img[max(0, cy-radius):min(frame_h, cy+radius),
                        max(0, cx-radius):min(frame_w, cx+radius)],
                ) if radius > 0 else None

                # Altitude estimation
                # EDGE-CLIP GUARD: if the circle is partially off-screen, the
                # enclosing radius is that of the VISIBLE ARC, not the full
                # circle — Z = f*D/(2r) would wildly overestimate altitude.
                # Fall back to DROP_ALTITUDE_TARGET so we don't command a dive.
                D_real    = FIRE_REAL_DIAMETERS.get(detected_color or "", 1.0)
                Z_est_raw = alt_est.estimate(radius * 2.0, D_real)
                Z_est     = mission.DROP_ALTITUDE_TARGET if edge_clipped else Z_est_raw

                altitude_safe = (mission.DROP_ALTITUDE_MIN <= Z_est <= mission.DROP_ALTITUDE_MAX)
                # Never trigger a drop on a partially-off-screen circle: the
                # altitude estimate is the fallback constant, not measured.
                if edge_clipped:
                    altitude_safe = False

                offset_x      = cx - frame_w // 2
                offset_y      = cy - frame_h // 2
                aligned       = (abs(offset_x) < mission.ALIGN_PIXEL_THRESH and
                                 abs(offset_y) < mission.ALIGN_PIXEL_THRESH)

                action = ("DROP"    if (altitude_safe and aligned) else
                          "ALIGN"   if altitude_safe else
                          "DESCEND" if Z_est > mission.DROP_ALTITUDE_MAX else "ASCEND")
                if edge_clipped:
                    action = f"EDGE_CLIP/{action}"

                # FlightCommand (same interface as test_video.py)
                cmd_vx = float(np.clip(-offset_y * mission.PX_TO_MS_FIRE, -1.0, 1.0))
                cmd_vy = float(np.clip( offset_x * mission.PX_TO_MS_FIRE, -1.0, 1.0))
                cmd_vz = (0.0  if altitude_safe else
                           0.6  if Z_est > mission.DROP_ALTITUDE_MAX else   # too high → descend (NED +)
                          -0.5)                                               # too low  → ascend  (NED -)
                cmd = FlightCommand(vx=cmd_vx, vy=cmd_vy, vz=cmd_vz,
                                    yaw_lock=True, phase="FIRE", action=action)
                print(cmd)

                color_bgr = _color_bgr(detected_color) if detected_color else (0, 255, 0)
                cv2.drawContours(img, [cnt_frame], -1, color_bgr, 2)
                cv2.circle(img, (cx, cy), 5, color_bgr, -1)
                # Frame-centre crosshair
                fc_x, fc_y = frame_w // 2, frame_h // 2
                cv2.line(img, (fc_x-15, fc_y), (fc_x+15, fc_y), (255,255,255), 1)
                cv2.line(img, (fc_x, fc_y-15), (fc_x, fc_y+15), (255,255,255), 1)

                alt_src = "VIS"   # single-image tool: no live sensor available
                alt_str = f"Z={Z_est:.2f}m[{alt_src}]" if Z_est < 99 else "Z=?"
                edge_str = " [EDGE]" if edge_clipped else ""
                label   = (f"Circle [{detected_color or '?'}] | "
                           f"{alt_str}{edge_str} fr={fr:.2f} | {action} | {conf_score:.2f}")
                label_y = max(20, cy - radius - 12)
                cv2.putText(img, label, (max(0, cx - 80), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                _draw_altitude_bar(img, Z_est, frame_w, frame_h, color_bgr)

            # ── Semicircle (TZ obstacle ring) ─────────────────────────────────
            elif class_name == "semicircle":
                cv2.drawContours(img, [cnt_frame], -1, (0, 0, 255), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                d_ratio = compute_diameter_ratio(radius, frame_w)
                cv2.putText(img,
                            f"Semicircle | d={d_ratio:.2f} | conf={conf_score:.2f}",
                            (max(0, cx - 55), max(20, cy - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ── H_landing ─────────────────────────────────────────────────────
            elif class_name == "H_landing":
                x1, y1, x2, y2 = (int(box.xyxy[0][k].item()) for k in range(4))
                pad_w   = x2 - x1
                pad_cx  = (x1 + x2) // 2
                pad_cy  = (y1 + y2) // 2
                d_ratio = float(pad_w) / float(frame_w)
                action  = _decide_action(d_ratio)

                offset_x = pad_cx - frame_w // 2
                offset_y = pad_cy - frame_h // 2
                cmd_vx   = float(np.clip(-offset_y * mission.PX_TO_MS_HELIPAD, -0.5, 0.5))
                cmd_vy   = float(np.clip( offset_x * mission.PX_TO_MS_HELIPAD, -0.5, 0.5))
                cmd_vz   = 0.3 if action != "LAND" else 0.4
                cmd = FlightCommand(vx=cmd_vx, vy=cmd_vy, vz=cmd_vz,
                                    yaw_lock=True, phase="FIND_HELIPAD", action=action)
                print(cmd)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.circle(img, (pad_cx, pad_cy), 6, (255, 128, 0), -1)
                cv2.line(img, (pad_cx-20, pad_cy), (pad_cx+20, pad_cy), (255,128,0), 1)
                cv2.line(img, (pad_cx, pad_cy-20), (pad_cx, pad_cy+20), (255,128,0), 1)
                label = (f"H_landing | d={d_ratio:.2f} | {action} | {conf_score:.2f}")
                cv2.putText(img, label, (x1, max(14, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
                _draw_ratio_bar(img, d_ratio, frame_w, frame_h, (255, 128, 0))

    cv2.imwrite(args.output, img)
    print(f"Saved -> {args.output}")


# ---------------------------------------------------------------------------
# Helper: ROI mask for colour detection (avoids full-frame resize)
# ---------------------------------------------------------------------------

def _roi_mask(frame, cnt_frame, cx, cy, radius, frame_w, frame_h):
    """Return a small binary mask in the circle's bounding box."""
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(frame_w, cx + radius)
    y2 = min(frame_h, cy + radius)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cnt_roi  = (cnt_frame.reshape(-1, 2) - np.array([x1, y1], dtype=np.int32))
    cnt_roi  = np.clip(cnt_roi, 0,
                       [roi_mask.shape[1] - 1, roi_mask.shape[0] - 1])
    cv2.fillPoly(roi_mask, [cnt_roi.reshape(-1, 1, 2)], 255)
    return roi_mask


if __name__ == "__main__":
    main()
