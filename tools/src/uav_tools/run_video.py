"""uav-run-video: replay a recorded flight through perception + mission and render the HUD.

    uav-run-video --weights model.onnx --source flight.mp4 --output hud.mp4 --log run.csv \\
        [--config mission.yaml] [--hsv-calib hsv_calib.json] [--calib calibration.npz] \\
        [--finals] [--with-takeoff] [--altitude 3.4]

Replaces the legacy test_video.py and postprocess.py. Time comes from the video (frame index
/ fps), so results do not depend on how fast this machine runs. A recorded video carries no
sensor data: without --altitude the mission falls back to the vision altitude, as in the
legacy tool. Starts in SEARCHING unless --with-takeoff (videos usually begin in the air).
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from uav_core.config import CameraConfig, MissionConfig, load_mission_config, with_hsv_ranges
from uav_core.mission import MissionStateMachine
from uav_core.perception import detect
from uav_core.perception.camera import Calibration, Undistorter
from uav_core.perception.color import load_hsv_calibration
from uav_core.types import Round, SensorSnapshot
from uav_core.viz import draw_hud
from uav_edge.runtime import Backend, load_backend

LOG_FIELDS = (
    "frame", "t_s", "phase", "action", "vx", "vy", "vz", "yaw_rate_dps", "drop",
    "score", "penalties", "balls", "target_index", "alt_source", "detections", "infer_ms",
)  # fmt: skip


@dataclass(frozen=True)
class ReplayOptions:
    round_: Round = Round.QUALIFIER
    skip_takeoff: bool = True
    altitude_m: float | None = None


def replay(
    frames: Iterable[np.ndarray],
    fps: float,
    backend: Backend,
    cfg: MissionConfig,
    opts: ReplayOptions | None = None,
    undistort: Undistorter | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, object]]]:
    """Yield (HUD frame, log row) per input frame."""
    opts = opts if opts is not None else ReplayOptions()
    sm = MissionStateMachine(cfg, round_=opts.round_)
    if opts.skip_takeoff:
        sm.skip_takeoff()
    sensors = SensorSnapshot(fc_alt_m=opts.altitude_m)
    for index, raw in enumerate(frames):
        frame = undistort(raw) if undistort is not None else raw
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        instances = backend.infer(frame)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        dets = detect(instances, frame, cfg.perception)
        t_s = index / fps
        out = sm.step(t_s, dets, sensors, (w, h))
        c, s = out.command, out.status
        row = {
            "frame": index, "t_s": round(t_s, 4), "phase": s.phase.value, "action": c.action,
            "vx": round(c.vx, 4), "vy": round(c.vy, 4), "vz": round(c.vz, 4),
            "yaw_rate_dps": round(c.yaw_rate_dps, 2), "drop": int(c.drop_ball),
            "score": s.score, "penalties": s.penalties, "balls": s.balls_dropped,
            "target_index": s.target_index, "alt_source": s.alt_source.value,
            "detections": len(dets), "infer_ms": round(infer_ms, 2),
        }  # fmt: skip
        yield draw_hud(frame, out, dets, cfg, altitude_m=opts.altitude_m), row


def build_config(config: Path | None, hsv_calib: Path | None, f_px: float | None) -> MissionConfig:
    cfg = load_mission_config(config) if config else MissionConfig()
    if hsv_calib:
        cfg = with_hsv_ranges(cfg, load_hsv_calibration(hsv_calib))
    if f_px is not None:
        cfg = cfg.model_copy(update={"camera": CameraConfig(f_px=f_px)})
    return cfg


def _read(cap: cv2.VideoCapture, limit: int | None) -> Iterator[np.ndarray]:
    count = 0
    while limit is None or count < limit:
        ok, frame = cap.read()
        if not ok:
            return
        count += 1
        yield frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-run-video", description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help=".onnx or .pt")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="HUD video (.mp4)")
    parser.add_argument("--log", type=Path, help="per-frame CSV")
    parser.add_argument("--config", type=Path, help="mission YAML")
    parser.add_argument("--hsv-calib", type=Path)
    parser.add_argument("--calib", type=Path, help="calibration.npz: undistort and set f_px")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--finals", action="store_true")
    parser.add_argument("--with-takeoff", action="store_true")
    parser.add_argument("--altitude", type=float, help="constant FC altitude, m")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    undistort = Undistorter(Calibration.load(args.calib)) if args.calib else None
    cfg = build_config(args.config, args.hsv_calib, undistort.f_px if undistort else None)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        parser.error(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = ReplayOptions(
        round_=Round.FINALS if args.finals else Round.QUALIFIER,
        skip_takeoff=not args.with_takeoff,
        altitude_m=args.altitude,
    )
    writer = None
    log_file = args.log.open("w", newline="") if args.log else None
    log = csv.DictWriter(log_file, LOG_FIELDS) if log_file else None
    if log:
        log.writeheader()
    rows = 0
    last: dict[str, object] = {}
    try:
        backend = load_backend(args.weights, conf=args.conf)
        for img, row in replay(_read(cap, args.max_frames), fps, backend, cfg, opts, undistort):
            if args.output and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output), fourcc, fps, img.shape[1::-1])
            if writer is not None:
                writer.write(img)
            if log:
                log.writerow(row)
            rows += 1
            last = row
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log_file:
            log_file.close()
    print(
        f"{rows} frames; final phase {last.get('phase')}, score {last.get('score')}, "
        f"balls {last.get('balls')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
