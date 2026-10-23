"""Write the golden files: `uv run python uav_core/tests/characterisation/record_golden.py`.

The goldens were recorded from legacy/vision/mission.py (see git history, commit 572856f, for
the legacy runner). That file is gone: re-record from uav_core only for a deliberate
behaviour change, and say why in the commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `characterisation` importable

from characterisation.new_runner import run_new
from characterisation.records import golden_path, write_golden
from characterisation.scenarios import SCENARIOS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="scenario names (default: all)")
    args = parser.parse_args(argv)
    selected = [s for s in SCENARIOS if not args.names or s.name in args.names]
    for scenario in selected:
        records = run_new(scenario)
        write_golden(golden_path(scenario.name), records)
        print(f"{scenario.name}: {len(records)} steps -> {golden_path(scenario.name).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
