"""
test_video.py — Video inference with full mission state machine.

Usage:
    python test_video.py --weights best.pt --source flight.mp4 --output result.mp4
    python test_video.py --weights best.pt --source 0 --output result.mp4 --calib calibration.npz
    python test_video.py --weights best.pt --source flight.mp4 --output result.mp4 \\
                         --skip-frames 1 --log telemetry.csv
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

import cv2
from ultralytics import YOLO

from mission import MissionStateMachine, DEFAULT_F_PX
from logger  import TelemetryLogger


# ---------------------------------------------------------------------------
# Threaded frame capture + undistortion
# ---------------------------------------------------------------------------

class FrameBuffer:
    """
    Background thread that reads frames from the camera and applies
    undistortion, keeping a 1-frame deep queue so the inference thread
    always has the latest frame ready without blocking on I/O.

    Why this matters:
      The synchronous pattern (read → undistort → infer → draw → write) forces
      the NPU/GPU to wait idle while the CPU undistorts the next frame.
      With FrameBuffer, undistortion runs concurrently with YOLO inference,
      so each step runs at its own natural speed.

    Usage:
        buf = FrameBuffer(cap, undist=undist_obj)
        buf.start()
        while True:
            frame = buf.read()
            if frame is None: break
            results = model(frame, ...)
        buf.stop()
    """

    def __init__(self, cap: cv2.VideoCapture, undist=None, maxsize: int = 2):
        self._cap    = cap
        self._undist = undist
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event     = threading.Event()
        self._thread: threading.Thread | None = None
        self.latest_size: tuple[int, int] = (0, 0)  # (w, h) after undistort

    def start(self) -> "FrameBuffer":
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name="FrameCapture")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def read(self) -> tuple[cv2.typing.MatLike, float] | None:
        """
        Block until a new frame is available (max 0.5s).
        Returns (frame, grab_timestamp) where grab_timestamp is
        time.perf_counter() recorded immediately after cap.read() —
        use this as frame_timestamp in mission.update() for IMU time-sync.
        Returns None when the stream has ended.
        """
        try:
            return self._q.get(timeout=0.5)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            grab_ts = time.perf_counter()   # record grab time immediately after read
            if not ret:
                self._q.put(None)   # sentinel: stream ended
                break
            if self._undist is not None:
                frame = self._undist.correct(frame)
            h, w = frame.shape[:2]
            self.latest_size = (w, h)
            # Discard oldest frame if queue is full (always stay current)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((frame, grab_ts))


# ---------------------------------------------------------------------------
# Rolling FPS display
# ---------------------------------------------------------------------------

class FPSTracker:
    def __init__(self, window: int = 30):
        from collections import deque
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)

    def draw(self, frame, fps: float) -> None:
        h, w = frame.shape[:2]
        color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO11-seg video inference with mission state machine"
    )
    parser.add_argument("--weights",     required=True, help="Path to best.pt")
    parser.add_argument("--source",      required=True,
                        help="Video file path or camera index (0, 1, ...)")
    parser.add_argument("--output",      required=True, help="Output video path")
    parser.add_argument("--conf",        type=float, default=0.5)
    parser.add_argument("--calib",       default=None,
                        help="(Optional) calibration.npz for IMX219 undistortion + f_px")
    parser.add_argument("--f-px",        type=float, default=None,
                        help="Override focal length in pixels (default: from calib or 500)")
    parser.add_argument("--finals",      action="store_true",
                        help="Use Finals scoring (F2=50pts moving target, PID tuned higher)")
    parser.add_argument("--takeoff-alt", type=float, default=3.3,
                        help="Target takeoff altitude in metres (default 3.3, rule min 3.0)")
    parser.add_argument("--skip-takeoff", action="store_true",
                        help="Skip TAKEOFF phase immediately (use when testing on video "
                             "or when FC handles takeoff externally)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Process every Nth frame (0=all). Use 1 to halve GPU load.")
    parser.add_argument("--smooth",      type=float, default=0.4,
                        help="Temporal smoothing alpha 0-1 (default 0.4)")
    parser.add_argument("--log",         default=None,
                        help="Path for telemetry CSV (e.g. run_001.csv). "
                             "Omit to skip logging.")
    parser.add_argument("--hsv-calib",   default=None,
                        help="HSV calibration JSON produced by hsv_calibrator.py. "
                             "Run hsv_calibrator.py at the venue before the match "
                             "to tune colour thresholds under actual lighting.")
    parser.add_argument("--config-yaml", default=None,
                        help="Path to config.yaml for on-field constant tuning. "
                             "Overrides mission.py defaults without editing source. "
                             "Example: --config-yaml config.yaml")
    args = parser.parse_args()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = YOLO(args.weights)

    # ── HSV colour calibration (optional, recommended for competition) ────────
    # Load tuned colour thresholds produced by hsv_calibrator.py.
    # Do this before any inference starts so every frame uses the field-tuned
    # ranges, not the hardcoded defaults in mission.py.
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
        from mission import AltitudeEstimator
        # crop=True removes black border pixels that appear after undistortion.
        # These borders would otherwise inflate frame_w/frame_h and skew
        # d_ratio calculations (d_ratio = pad_width / frame_w).
        undist = IMX219Undistorter(args.calib, alpha=0.0, crop=True)
        if f_px is None:
            est  = AltitudeEstimator.from_calibration(args.calib)
            f_px = est.f_px
        print(f"[Calib] Loaded: {args.calib}  f_px={f_px:.1f}  crop=True")

    # ── Mission state machine ──────────────────────────────────────────────────
    mission = MissionStateMachine(
        smoothing_alpha=args.smooth,
        f_px=f_px if f_px is not None else DEFAULT_F_PX,
        is_finals=args.finals,
        takeoff_alt=args.takeoff_alt,
    )
    if args.finals:
        print("[Mission] FINALS mode: F2=moving target, PID tuned higher")
    if args.skip_takeoff:
        mission.skip_takeoff()
        print("[Mission] --skip-takeoff: starting directly in SEARCHING")

    # ── Telemetry logger ──────────────────────────────────────────────────────
    tlog: TelemetryLogger | None = None
    if args.log:
        tlog = TelemetryLogger(args.log)

    # ── Video I/O ──────────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Error: cannot open source!")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Inform the mission state machine of the actual pipeline FPS.
    # This ensures time-based thresholds (altitude penalty, drop cooldown) are
    # correct regardless of whether the source is 20 fps, 30 fps, or 60 fps.
    mission.fps_hint = fps

    # VideoWriter is created after the first frame so dimensions reflect
    # the post-undistort size (crop=True may reduce width/height slightly).
    out          = None
    writer_size  = (width, height)   # will be updated on first frame

    fps_tracker = FPSTracker(window=30)

    print("Processing...")
    print("Mission: F1(red)->F2(red)->F3(yellow)->F4(yellow)->F5(blue)->Land")
    if args.skip_frames:
        print(f"Frame-skip: processing 1 in every {args.skip_frames + 1} frames")

    # ── Sensor stub warning ───────────────────────────────────────────────────
    # These stubs are safe for offline video testing but MUST be wired to real
    # hardware before competition flight.  The warnings below make a misconfigured
    # deployment immediately visible in the log.
    print("[WARN] fc_alt stub active — altitude priority ladder will fall back to "
          "vision-only (Z_est). Wire mavlink_conn.altitude_m() before competition flight.")
    print("[WARN] sensor_data stub active — attitude gate, ToF block, geofence, and "
          "Finals feedforward are all inactive. Wire SensorData before competition flight.")

    # ── Threaded frame capture ────────────────────────────────────────────────
    # FrameBuffer reads + undistorts in a background thread so the inference
    # thread is never blocked waiting on camera I/O or CPU undistortion.
    frame_buf = FrameBuffer(cap, undist=undist).start()

    frame_idx  = 0
    last_frame = None

    while True:
        result_buf = frame_buf.read()
        if result_buf is None:
            break   # stream ended
        frame, grab_ts = result_buf

        # Lazy VideoWriter: created on first frame so size reflects actual
        # post-undistort dimensions (crop=True may reduce width/height).
        if out is None:
            h_f, w_f = frame.shape[:2]
            writer_size = (w_f, h_f)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(args.output, fourcc, fps, writer_size)
            if (w_f, h_f) != (width, height):
                print(f"[Calib] Frame size after undistort: "
                      f"{width}x{height} -> {w_f}x{h_f}")

        # Frame-skip: write last annotated frame to keep video smooth
        if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 0:
            if last_frame is not None:
                out.write(last_frame)
            frame_idx += 1
            continue

        # Inference (timed for telemetry)
        t0 = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        result = results[0]

        # Mission update
        # ── COMPETITION WIRING (must complete before first flight) ─────────────
        # Replace the stubs below with real sensor reads.  All fields default to
        # None so the state machine degrades gracefully in offline video testing,
        # but on the drone you MUST provide at minimum fc_alt + tof + IMU data.
        #
        # from mission import SensorData
        # fc_alt = mavlink_conn.altitude_m()          # MAVLink GLOBAL_POSITION_INT
        # sensors = SensorData(
        #     tof_altitude_m  = tof_sensor.read_m(),  # MTF-01 rangefinder
        #     of_vx_ms        = flow.vx_ms(),         # MTF-01 optical flow
        #     of_vy_ms        = flow.vy_ms(),
        #     pitch_deg       = imu.pitch_deg,         # BNO080
        #     roll_deg        = imu.roll_deg,
        #     pitch_rate_dps  = imu.pitch_rate_dps,
        #     roll_rate_dps   = imu.roll_rate_dps,
        # )
        # frame_ts = grab_ts  # FrameBuffer records grab time; use that directly
        fc_alt   = None   # STUB — wire to mavlink_conn.altitude_m()
        sensors  = None   # STUB — wire to SensorData(...) as above
        frame = mission.update(frame, result, current_altitude=fc_alt,
                               sensor_data=sensors, frame_timestamp=grab_ts)

        # FPS overlay
        current_fps = fps_tracker.tick()
        fps_tracker.draw(frame, current_fps)

        if tlog is not None:
            tlog.record(frame_idx, inference_ms, result, mission)

        last_frame = frame.copy()
        out.write(frame)
        frame_idx += 1

        if mission.mission_complete:
            print("[Mission] Complete - stopping.")
            break

    frame_buf.stop()
    cap.release()
    if out is not None:
        out.release()

    if tlog is not None:
        tlog.close()
        tlog.summary()

    print(f"\nDone -> {args.output} ({frame_idx} frames)")
    print(f"Result: {mission.balls_dropped}/5 balls | {mission.score} pts")


if __name__ == "__main__":
    main()