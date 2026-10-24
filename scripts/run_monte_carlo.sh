#!/usr/bin/env bash
# N headless simulated matches with consecutive seeds, then the referee report.
#   scripts/run_monte_carlo.sh 50 qualifier [first_seed]
# Needs the simulation image (docker compose -f docker/compose.yaml build sim) and the model in
# assets/. Each run is capped at 400 s of wall time per 300 s match on top of startup.
set -euo pipefail
N=${1:-50}
ROUND=${2:-qualifier}
FIRST=${3:-0}
WEIGHTS=${WEIGHTS:-yolo11s-seg_synth_export-2026-04-24.onnx}
OUT=results/mc-$ROUND
mkdir -p "$OUT"
for ((seed = FIRST; seed < FIRST + N; seed++)); do
  echo "=== $ROUND seed $seed"
  docker run --rm -v "$PWD/assets:/assets:ro" -v "$PWD/$OUT:/results" uav-sim:latest \
    timeout --signal=INT 600 ros2 launch uav_bringup sim.launch.py \
    weights:="/assets/$WEIGHTS" seed:="$seed" round:="$ROUND" random_layout:=true \
    results_dir:=/results > "$OUT/seed$seed.log" 2>&1 || true
done
docker run --rm -v "$PWD/$OUT:/results" uav-sim:latest \
  python3 -m uav_sim.report /results --out /results/report.md
cat "$OUT/report.md"
