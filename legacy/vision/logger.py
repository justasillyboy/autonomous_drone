"""
logger.py — Runtime telemetry for competition debugging.

Logs per-frame metrics (FPS, inference latency, detections, mission phase)
to a CSV file. The file is written incrementally so it survives a crash
mid-flight and can be inspected immediately after landing.

Usage — integrate into test_video.py:
    from logger import TelemetryLogger
    log = TelemetryLogger("run_001.csv")

    while cap.isOpened():
        t0 = time.perf_counter()
        results = model(frame, ...)
        inference_ms = (time.perf_counter() - t0) * 1000

        frame = mission.update(frame, results[0])
        log.record(frame_idx, inference_ms, results[0], mission)

    log.close()
    log.summary()          # prints mean FPS, drop events, score
"""

from __future__ import annotations

import csv
import time
from pathlib import Path


class TelemetryLogger:
    """
    Writes one CSV row per frame with:
      frame_idx, wall_time_s, inference_ms, fps,
      n_detections, phase, balls_dropped, score,
      top_class, top_conf, top_d_ratio
    """

    COLUMNS = [
        "frame_idx",
        "wall_time_s",
        "inference_ms",
        "fps",
        "n_detections",
        "phase",
        "action",
        "balls_dropped",
        "score",
        "top_class",
        "top_conf",
        "top_d_ratio",
        # FlightCommand fields — critical for post-flight control debugging
        "cmd_vx",
        "cmd_vy",
        "cmd_vz",
        "cmd_yaw_rate",
        "cmd_drop_ball",
    ]

    def __init__(self, path: str = "telemetry.csv", fps_window: int = 30):
        self._path   = Path(path)
        self._file   = open(path, "w", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()

        # Rolling FPS
        from collections import deque
        self._times: deque = deque(maxlen=fps_window)
        self._t_start = time.perf_counter()

        print(f"[Logger] Telemetry -> {self._path.resolve()}")

    # ── Per-frame recording ───────────────────────────────────────────────────

    def record(self, frame_idx: int, inference_ms: float,
               yolo_result, mission) -> None:
        """
        Call once per frame after model inference and mission.update().

        Args:
            frame_idx     : Frame counter (int).
            inference_ms  : Inference time in milliseconds.
            yolo_result   : Single Ultralytics Results object.
            mission       : MissionStateMachine instance.
        """
        now = time.perf_counter()
        self._times.append(now)
        fps = ((len(self._times) - 1) /
               max(self._times[-1] - self._times[0], 1e-6)
               if len(self._times) > 1 else 0.0)

        # Best detection this frame
        top_class   = ""
        top_conf    = 0.0
        top_d_ratio = 0.0
        n_det       = 0

        if yolo_result.boxes is not None and len(yolo_result.boxes):
            n_det     = len(yolo_result.boxes)
            confs     = yolo_result.boxes.conf.cpu().numpy()
            best_idx  = int(confs.argmax())
            top_conf  = float(confs[best_idx])
            cls_id    = int(yolo_result.boxes.cls[best_idx].item())
            top_class = yolo_result.names.get(cls_id, str(cls_id))

            # Approximate diameter ratio from bbox width
            xyxy     = yolo_result.boxes.xyxy[best_idx].cpu().numpy()
            bbox_w   = float(xyxy[2] - xyxy[0])
            img_w    = yolo_result.orig_shape[1]
            top_d_ratio = bbox_w / max(img_w, 1)

        # Extract FlightCommand from mission for control debugging
        cmd = getattr(mission, "last_command", None)

        self._writer.writerow({
            "frame_idx":    frame_idx,
            "wall_time_s":  round(now - self._t_start, 3),
            "inference_ms": round(inference_ms, 2),
            "fps":          round(fps, 1),
            "n_detections": n_det,
            "phase":        mission.phase.name,
            "action":       cmd.action if cmd else "",
            "balls_dropped": mission.balls_dropped,
            "score":        mission.score,
            "top_class":    top_class,
            "top_conf":     round(top_conf, 3),
            "top_d_ratio":  round(top_d_ratio, 3),
            "cmd_vx":       round(cmd.vx,       3) if cmd else 0.0,
            "cmd_vy":       round(cmd.vy,       3) if cmd else 0.0,
            "cmd_vz":       round(cmd.vz,       3) if cmd else 0.0,
            "cmd_yaw_rate": round(cmd.yaw_rate, 2) if cmd else 0.0,
            "cmd_drop_ball": int(cmd.drop_ball)    if cmd else 0,
        })

    # ── Session summary ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()

    def summary(self) -> dict:
        """
        Read back the CSV and print a human-readable competition summary.
        Call after close().

        Returns dict with key metrics for programmatic use.
        """
        import csv as _csv
        rows = []
        with open(self._path) as f:
            rows = list(_csv.DictReader(f))

        if not rows:
            print("[Logger] No data recorded.")
            return {}

        fps_vals      = [float(r["fps"])          for r in rows if r["fps"]]
        latency_vals  = [float(r["inference_ms"]) for r in rows if r["inference_ms"]]
        drop_frames   = [r for r in rows if r.get("cmd_drop_ball", "0") == "1"]
        land_frames   = [r for r in rows if r["phase"] == "LANDING"]
        recovery_frames = [r for r in rows if r["phase"] == "RECOVERY"]
        final_score   = int(rows[-1]["score"]) if rows[-1]["score"] else 0
        final_balls   = int(rows[-1]["balls_dropped"]) if rows[-1]["balls_dropped"] else 0
        total_frames  = len(rows)
        duration_s    = float(rows[-1]["wall_time_s"]) if rows[-1]["wall_time_s"] else 0

        mean_fps     = sum(fps_vals) / len(fps_vals)   if fps_vals    else 0
        min_fps      = min(fps_vals)                    if fps_vals    else 0
        mean_lat     = sum(latency_vals) / len(latency_vals) if latency_vals else 0
        max_lat      = max(latency_vals)                if latency_vals else 0

        print("\n" + "=" * 56)
        print("  Competition telemetry summary")
        print("=" * 56)
        print(f"  Duration       : {duration_s:.1f} s  ({total_frames} frames)")
        print(f"  Mean FPS       : {mean_fps:.1f}  (min {min_fps:.1f})")
        print(f"  Mean latency   : {mean_lat:.1f} ms  (max {max_lat:.1f} ms)")
        print(f"  Balls dropped  : {final_balls} / 5")
        print(f"  Drop triggers  : {len(drop_frames)} frames with cmd_drop_ball=1")
        print(f"  Landing frames : {len(land_frames)}")
        print(f"  Recovery frames: {len(recovery_frames)}")
        print(f"  Final score    : {final_score} pts")
        print(f"  CSV saved      : {self._path.resolve()}")
        print("=" * 56)

        if mean_fps < 15:
            print("  WARNING: mean FPS below 15 — consider --skip-frames 1 or smaller model")
        if max_lat > 150:
            print(f"  WARNING: peak latency {max_lat:.0f} ms — check GPU load")

        return {
            "mean_fps":    round(mean_fps, 1),
            "min_fps":     round(min_fps, 1),
            "mean_lat_ms": round(mean_lat, 1),
            "max_lat_ms":  round(max_lat, 1),
            "balls":       final_balls,
            "score":       final_score,
            "frames":      total_frames,
            "duration_s":  round(duration_s, 1),
        }
