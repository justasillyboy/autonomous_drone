"""Write the golden files: `uv run python uav_core/tests/characterisation/record_golden.py`.

--source legacy (default) runs legacy/vision/mission.py; --source new runs uav_core, for
deliberate behaviour changes once the legacy file is gone (say why in the commit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `characterisation` importable

from characterisation.records import golden_path, write_golden
from characterisation.scenarios import SCENARIOS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("legacy", "new"), default="legacy")
    parser.add_argument("names", nargs="*", help="scenario names (default: all)")
    args = parser.parse_args(argv)
    if args.source == "legacy":
        from characterisation.legacy_runner import run_legacy as run
    else:
        from characterisation.new_runner import run_new as run
    selected = [s for s in SCENARIOS if not args.names or s.name in args.names]
    for scenario in selected:
        records = run(scenario)
        write_golden(golden_path(scenario.name), records)
        print(f"{scenario.name}: {len(records)} steps -> {golden_path(scenario.name).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
