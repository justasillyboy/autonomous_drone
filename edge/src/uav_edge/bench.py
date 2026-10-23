"""uav-edge-bench: per-stage latency of the perception + mission loop on this machine.

    uv run uav-edge-bench --weights model.onnx --source frames/ --frames 300 --out bench.json
    uv run uav-edge-bench --weights model.hef --source flight.mp4 --duration 600   # on the Pi

Stages: preprocess (letterbox), inference, postprocess (decode, NMS, masks), detect
(geometry, colour, filters), mission step. Reports p50/p95/p99 in ms, end-to-end FPS, CPU
use of this process and, on a Raspberry Pi, SoC temperature and the throttling flags over
time (a 10-minute run shows whether the Pi throttles). Replaces the never-measured "~2 ms
inference" of the legacy README with numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from uav_core.config import MissionConfig
from uav_core.mission import MissionStateMachine
from uav_core.perception import detect
from uav_core.types import SensorSnapshot

from uav_edge.runtime import Backend, load_backend

STAGES = ("preprocess", "inference", "postprocess", "detect", "mission_step", "total")
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")


def frames(source: Path, limit: int | None) -> Iterator[np.ndarray]:
    """Frames from a video or an image folder, looping until `limit` (or once if None)."""
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        images = [img for p in paths if (img := cv2.imread(str(p))) is not None]
        if not images:
            raise FileNotFoundError(f"no images in {source}")
        count = 0
        while limit is None or count < limit:
            for img in images:
                if limit is not None and count >= limit:
                    return
                yield img
                count += 1
            if limit is None:
                return
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {source}")
    count = 0
    while limit is None or count < limit:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                return
        yield frame
        count += 1


def soc_status() -> dict[str, object]:
    """Temperature (deg C) and `vcgencmd get_throttled` on a Pi; empty elsewhere."""
    out: dict[str, object] = {}
    if THERMAL.exists():
        out["temp_c"] = int(THERMAL.read_text()) / 1000.0
    try:
        res = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        )
        out["throttled"] = res.stdout.strip().split("=")[-1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def percentiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values) * 1000.0
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (50, 95, 99)} | {
        "mean": round(float(a.mean()), 3)
    }


def run(
    backend: Backend,
    source: Iterator[np.ndarray],
    warmup: int = 10,
    duration_s: float | None = None,
    status_every_s: float = 10.0,
) -> dict:
    staged = all(hasattr(backend, m) for m in ("preprocess", "run", "postprocess"))
    cfg = MissionConfig()
    sm = MissionStateMachine(cfg)
    sm.skip_takeoff()
    times: dict[str, list[float]] = {s: [] for s in STAGES}
    status: list[dict] = []
    t_start, cpu_start, next_status = time.perf_counter(), time.process_time(), 0.0
    n = 0
    for i, frame in enumerate(source):
        t0 = time.perf_counter()
        if staged:
            lb, x = backend.preprocess(frame)
            t1 = time.perf_counter()
            raw = backend.run(x)
            t2 = time.perf_counter()
            instances = backend.postprocess(raw, lb)
        else:
            t1 = t0
            instances = backend.infer(frame)
            t2 = time.perf_counter()
        t3 = time.perf_counter()
        dets = detect(instances, frame, cfg.perception)
        t4 = time.perf_counter()
        sm.step(i / 30.0, dets, SensorSnapshot(fc_alt_m=3.4), frame.shape[1::-1])
        t5 = time.perf_counter()
        if i >= warmup:
            n += 1
            for name, dt in zip(
                STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0), strict=True
            ):
                times[name].append(dt)
        elapsed = t5 - t_start
        if elapsed >= next_status:
            status.append({"t_s": round(elapsed, 1), **soc_status()})
            next_status += status_every_s
        if duration_s is not None and elapsed >= duration_s:
            break
    wall = time.perf_counter() - t_start
    if not staged:
        times.pop("preprocess")
        times.pop("postprocess")
    return {
        "frames": n,
        "staged": staged,
        "latency_ms": {k: percentiles(v) for k, v in times.items() if v},
        "fps": round(n / sum(times["total"]), 2) if n else 0.0,
        "cpu_percent": round(100.0 * (time.process_time() - cpu_start) / wall, 1),
        "soc": status,
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-edge-bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help=".onnx, .pt or .hef")
    parser.add_argument("--source", type=Path, required=True, help="video or image folder")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--duration", type=float, help="seconds; overrides --frames (thermal runs)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    limit = None if args.duration else args.frames + args.warmup
    source = frames(args.source, limit if not args.duration else 10**9)
    report = {
        "weights": args.weights.name,
        **run(load_backend(args.weights), source, args.warmup, args.duration),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
