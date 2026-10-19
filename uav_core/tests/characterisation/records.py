"""Golden records: one row per mission step, stored as gzip-compressed JSON lines."""

from __future__ import annotations

import gzip
import io
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
FLOAT_FIELDS = ("vx", "vy", "vz", "yaw_rate", "elapsed")
FLOAT_ATOL = 1e-9
# Legacy action strings embed frame counters ("ZOH_COAST_5/33"); the port counts seconds and
# drops them (ADR-0001), so both sides are compared without the counter.
_COUNTER = re.compile(r"^(ZOH_COAST|NO_DETECT)_\d+/\d+")


@dataclass(frozen=True)
class Record:
    i: int
    t: float
    phase: str  # FlightCommand.phase
    action: str
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    yaw_lock: bool
    drop: bool
    mission_phase: str  # phase after the step
    score: int
    penalties: int
    balls: int
    target_idx: int
    alt_src: str
    elapsed: float


def normalise_action(action: str) -> str:
    return _COUNTER.sub(r"\1", action)


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.jsonl.gz"


def write_golden(path: Path, records: Iterable[Record]) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for rec in records:
            gz.write((json.dumps(asdict(rec), sort_keys=True) + "\n").encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def read_golden(path: Path) -> list[Record]:
    with gzip.open(path, "rt") as fh:
        return [Record(**json.loads(line)) for line in fh]


def compare(expected: Sequence[Record], actual: Sequence[Record], limit: int = 10) -> list[str]:
    """Return human-readable mismatches (at most `limit`); an empty list means equal."""
    problems: list[str] = []
    if len(expected) != len(actual):
        problems.append(f"length: expected {len(expected)} steps, got {len(actual)}")
    for exp, act in zip(expected, actual, strict=False):
        for f in fields(Record):
            a, b = getattr(exp, f.name), getattr(act, f.name)
            if f.name == "action":
                a, b = normalise_action(a), normalise_action(b)
            if f.name in FLOAT_FIELDS:
                same = math.isclose(a, b, rel_tol=0.0, abs_tol=FLOAT_ATOL)
            else:
                same = a == b
            if not same:
                problems.append(f"step {exp.i} (t={exp.t:.2f}) {f.name}: expected {a!r}, got {b!r}")
                break
        if len(problems) >= limit:
            break
    return problems
