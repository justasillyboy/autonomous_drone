"""uav-hsv: produce the HSV calibration JSON that perception loads.

    uav-hsv fit  --dataset data/real --output hsv_calib.json   # from labelled real images
    uav-hsv tune --source 0 --output hsv_calib.json            # sliders on a live camera

`fit` needs no screen and prints how well the default and fitted ranges recognise the
labelled circles, and how much background each colour would wrongly match.
`tune` needs an OpenCV build with a GUI: `uv run --with opencv-python uav-hsv tune ...`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from uav_core.config import DEFAULT_HSV_RANGES, PerceptionConfig
from uav_core.perception.color import color_of_mask, load_hsv_calibration
from uav_core.types import Color
from uav_ml.labels import iter_dataset

from uav_tools.hsv import HsvRanges, classify_sample, fit_ranges, in_ranges, save_hsv_json

CIRCLE_CLASS_ID = 0


@dataclass(frozen=True)
class Sample:
    """One labelled circle: its HSV pixels, the frame and mask (for the uav_core matcher)."""

    hsv: np.ndarray
    bgr: np.ndarray
    mask: np.ndarray  # uint8, 255 inside


def collect(dataset: Path, class_id: int = CIRCLE_CLASS_ID, only: set[Path] | None = None):
    """Circle samples plus background HSV pixels (everything outside every label)."""
    samples: list[Sample] = []
    background: list[np.ndarray] = []
    for path, img, instances in iter_dataset(dataset):
        if only is not None and path not in only:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        labelled = np.zeros(img.shape[:2], bool)
        for inst in instances:
            labelled |= inst.mask
            if inst.class_id == class_id and inst.mask.any():
                ys, xs = np.nonzero(inst.mask)
                y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = inst.mask[y1:y2, x1:x2].astype(np.uint8) * 255
                samples.append(Sample(hsv[inst.mask], img[y1:y2, x1:x2], crop))
        background.append(hsv[~labelled][::50])  # subsample: background is huge
    bg = np.concatenate(background) if background else np.zeros((0, 3), np.uint8)
    return samples, bg


def recognition_rate(samples: Sequence[Sample], ranges: HsvRanges, threshold: float) -> float:
    """Share of circles that uav_core's matcher assigns the colour their hue vote says."""
    hits = 0
    scored = 0
    for s in samples:
        truth = classify_sample(s.hsv)
        if truth is None:
            continue
        scored += 1
        hits += color_of_mask(s.mask, s.bgr, ranges, threshold) == truth
    return hits / scored if scored else 0.0


def report(samples, background, ranges_by_name: dict[str, HsvRanges], threshold: float) -> str:
    lines = ["ranges     circles recognised   background matched (red/yellow/blue)"]
    for name, ranges in ranges_by_name.items():
        rate = recognition_rate(samples, ranges, threshold)
        bg = [in_ranges(background, ranges[c]).mean() if len(background) else 0.0 for c in Color]
        lines.append(f"{name:<10} {rate:>17.1%}   " + " / ".join(f"{b:.1%}" for b in bg))
    return "\n".join(lines)


def cmd_fit(args: argparse.Namespace) -> int:
    images = sorted(p for p in args.dataset.rglob("*") if p.parent.name == "images")
    n_fit = len(images) - int(len(images) * args.holdout)
    fit_set, test_set = set(images[:n_fit]), set(images[n_fit:])
    samples, background = collect(args.dataset, args.class_id, fit_set)
    if not samples:
        print(f"no class-{args.class_id} labels under {args.dataset}", file=sys.stderr)
        return 1
    result = fit_ranges((s.hsv for s in samples), DEFAULT_HSV_RANGES, background)
    ranges = PerceptionConfig(hsv_ranges=result.ranges).hsv_ranges  # validates
    save_hsv_json(ranges, args.output)
    counts = ", ".join(f"{c.value} {n}" for c, n in result.samples.items())
    print(f"fit on {n_fit} images, {len(samples)} circles ({counts}) -> {args.output}")
    if result.defaulted:
        print("no samples, defaults kept for: " + ", ".join(c.value for c in result.defaulted))
    if test_set:
        samples, background = collect(args.dataset, args.class_id, test_set)
        print(f"held-out evaluation: last {len(test_set)} images (contiguous, by file name)")
    threshold = PerceptionConfig().color_match_threshold
    print(report(samples, background, {"default": DEFAULT_HSV_RANGES, "fitted": ranges}, threshold))
    return 0


def _gui_available() -> bool:
    try:
        cv2.namedWindow("uav-hsv-probe")
        cv2.destroyWindow("uav-hsv-probe")
    except cv2.error:
        return False
    return True


def cmd_tune(args: argparse.Namespace) -> int:  # pragma: no cover - interactive
    if not _gui_available():
        print("this OpenCV has no GUI; run: uv run --with opencv-python uav-hsv tune ...")
        return 1
    ranges = {c: [list(map(list, r)) for r in rs] for c, rs in _start_ranges(args).items()}
    source = int(args.source) if args.source.isdigit() else args.source
    still = cv2.imread(args.source) if isinstance(source, str) else None
    cap = None if still is not None else cv2.VideoCapture(source)
    color = Color.RED
    win = "uav-hsv tune: [r]ed [y]ellow [b]lue  [s]ave  [q]uit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def make_bars():
        for k, ((lo, hi)) in enumerate(ranges[color]):
            for ch, name, top in ((0, "H", 180), (1, "S", 255), (2, "V", 255)):
                cv2.createTrackbar(
                    f"{k}:{name}lo", win, lo[ch], top, lambda v, b=lo, c=ch: b.__setitem__(c, v)
                )
                cv2.createTrackbar(
                    f"{k}:{name}hi", win, hi[ch], top, lambda v, b=hi, c=ch: b.__setitem__(c, v)
                )

    make_bars()
    while True:
        frame = still if still is not None else cap.read()[1]
        if frame is None:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], np.uint8)
        for lo, hi in ranges[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.putText(view, color.value, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("r"), ord("y"), ord("b")):
            color = {ord("r"): Color.RED, ord("y"): Color.YELLOW, ord("b"): Color.BLUE}[key]
            cv2.destroyWindow(win)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            make_bars()
        elif key == ord("s"):
            final = {c: tuple((tuple(lo), tuple(hi)) for lo, hi in rs) for c, rs in ranges.items()}
            save_hsv_json(PerceptionConfig(hsv_ranges=final).hsv_ranges, args.output)
            print(f"saved {args.output}")
        elif key in (ord("q"), 27):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


def _start_ranges(args: argparse.Namespace) -> HsvRanges:
    if args.input:
        return load_hsv_calibration(args.input)
    return DEFAULT_HSV_RANGES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav-hsv", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit", help="fit ranges from YOLO-seg labelled images")
    fit.add_argument("--dataset", type=Path, required=True, help="folder with images/ + labels/")
    fit.add_argument("--output", type=Path, default=Path("hsv_calib.json"))
    fit.add_argument("--class-id", type=int, default=CIRCLE_CLASS_ID)
    fit.add_argument(
        "--holdout",
        type=float,
        default=0.3,
        help="share of images (last by name) kept for evaluation",
    )
    fit.set_defaults(func=cmd_fit)
    tune = sub.add_parser("tune", help="adjust ranges with sliders (needs a GUI)")
    tune.add_argument("--source", required=True, help="camera index, video or image file")
    tune.add_argument("--input", type=Path, help="start from this JSON instead of the defaults")
    tune.add_argument("--output", type=Path, default=Path("hsv_calib.json"))
    tune.set_defaults(func=cmd_tune)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
