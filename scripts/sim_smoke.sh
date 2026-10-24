#!/usr/bin/env bash
# Simulation smoke test, run by .github/workflows/sim-smoke.yml after the image is built.
#   scripts/sim_smoke.sh <image> <weights file name under assets/>
# Every phase runs even if an earlier one fails; each prints the tail of its log as a GitHub
# annotation (readable without downloading logs) and the script fails if any phase failed.
set -uo pipefail
IMAGE=$1
WEIGHTS=/assets/$2
mkdir -p results
STATUS=0

annotate() {  # level, title, file: the last 60 lines as one annotation
  local body
  body=$(tail -n 60 "$3" | sed 's/%/%25/g' | sed ':a;N;$!ba;s/\n/%0A/g')
  echo "::$1 title=$2::$body"
}

phase() {  # name, timeout seconds, command...
  local name=$1 secs=$2
  shift 2
  echo "=== $name"
  docker run --rm -v "$PWD/assets:/assets:ro" -v "$PWD/results:/results" "$IMAGE" \
    timeout --signal=INT "$secs" "$@" > "results/$name.log" 2>&1
  local rc=$?
  annotate notice "$name (exit $rc)" "results/$name.log"
  return $rc
}

phase world 120 bash -c 'set -x; ros2 run uav_sim make_world --seed 0 --out /results/world \
  && export GZ_SIM_RESOURCE_PATH=/results/world/models:$ARDUPILOT_GAZEBO/models \
  && which gz && gz sdf -k /results/world/competition_field.sdf \
  && gz sdf -k /results/world/models/iris_downcam/model.sdf' || STATUS=1

phase direction_check 600 ros2 launch uav_bringup sim.launch.py weights:="$WEIGHTS" \
  mode:=direction_check results_dir:=/results
if [ -f results/direction_check.json ]; then
  python3 -c "import json,sys; r=json.load(open('results/direction_check.json')); \
print(f\"::{'notice' if r['passed'] else 'error'}::direction check {r}\"); sys.exit(0 if r['passed'] else 1)" || STATUS=1
else
  echo "::error::direction check wrote no result"; STATUS=1
fi

phase qualifier 900 ros2 launch uav_bringup sim.launch.py weights:="$WEIGHTS" seed:=0 \
  results_dir:=/results
if [ -f results/qualifier-00000.json ]; then
  python3 -c "import json; r=json.load(open('results/qualifier-00000.json')); \
print(f\"::notice::referee score {r['score']}, hit {r['targets_hit']}, landed {r['landed_on_helipad']}, failures {r['failures']}\")"
else
  echo "::error::the referee wrote no result"; STATUS=1
fi
exit $STATUS
