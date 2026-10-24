"""Monte Carlo report: aggregate referee results/<round>-<seed>.json files. No ROS imports.

    python3 -m uav_sim.report results/ --out results/report.md

Success = all five targets hit and a landing on the helipad within 300 s. The success rate
gets a Wilson 95 % interval; scores get mean, quartiles and a histogram; failures are counted
by kind; where the mission's self-reported score was recorded, the referee's score is
compared with it (a gap means the mission believes it scored what it did not).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

MAX_SCORE = 200


def load_results(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        if "score" in data and "failures" in data:  # skip direction_check.json and others
            out.append(data)
    return out


def is_success(r: dict) -> bool:
    return not r["targets_missed"] and r["landed_on_helipad"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(results: Sequence[dict]) -> dict:
    n = len(results)
    scores = sorted(r["score"] for r in results)
    wins = sum(is_success(r) for r in results)
    gaps = [
        r["self_reported_score"] - r["score"]
        for r in results
        if r.get("self_reported_score") is not None
    ]

    def q(f: float) -> float:
        return scores[min(n - 1, int(f * n))] if n else 0.0

    return {
        "runs": n,
        "success_rate": wins / n if n else 0.0,
        "success_ci95": wilson(wins, n),
        "score_mean": sum(scores) / n if n else 0.0,
        "score_quartiles": (q(0.25), q(0.5), q(0.75)),
        "score_histogram": Counter(min(s // 20 * 20, MAX_SCORE) for s in scores),
        "failures": Counter(f for r in results for f in r["failures"]),
        "self_report_gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "self_report_overclaims": sum(g > 0 for g in gaps),
    }


def markdown(s: dict) -> str:
    lo, hi = s["success_ci95"]
    q1, med, q3 = s["score_quartiles"]
    lines = [
        f"# Monte Carlo report ({s['runs']} runs)",
        "",
        f"- Success (5 targets + helipad landing): **{s['success_rate']:.0%}** "
        f"(95 % CI {lo:.0%}-{hi:.0%})",
        f"- Referee score: mean {s['score_mean']:.1f} / {MAX_SCORE}, quartiles {q1} / {med} / {q3}",
    ]
    if s["self_report_gap_mean"] is not None:
        lines.append(
            f"- Self-reported minus referee score: mean {s['self_report_gap_mean']:+.1f}, "
            f"over-claimed in {s['self_report_overclaims']} runs"
        )
    lines += ["", "| Score bin | Runs |", "|---|---|"]
    lines += [f"| {b}-{b + 19} | {c} |" for b, c in sorted(s["score_histogram"].items())]
    lines += ["", "| Failure | Runs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["failures"].most_common()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = markdown(summarise(load_results(args.folder)))
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
