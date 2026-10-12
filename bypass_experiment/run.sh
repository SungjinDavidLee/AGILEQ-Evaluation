#!/bin/bash
# usage: ./run.sh <run_name> [config]    (starts CARLA if nothing listens on port 2000)
set -u
RUN=$1; CFG=${2:-configs/iter5_vis.json}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
: "${CARLA_ROOT:?set CARLA_ROOT to the CARLA 0.9.15 directory}"
cd "$HERE"
if ! (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; then
    "$CARLA_ROOT/CarlaUE4.sh" -quality_level=Low -benchmark -fps=15 -RenderOffScreen -prefernvidia \
        -carla-world-port=2000 > /dev/null 2>&1 &
    for _ in $(seq 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null && break; done
    sleep 10
fi
mkdir -p logs
$PY run_bypass_demo.py --run_name "$RUN" --config "$CFG" 2>&1 | tee "logs/${RUN}_console.log"
$PY make_videos.py --run_name "$RUN"
$PY - "$RUN" <<'PY'
import sys, json
e = json.load(open(f'logs/{sys.argv[1]}/episodes.json'))[0]
print({k: e.get(k) for k in ['steps', 'f_entries', 'f_returns', 'f_to_g', 'e_entries', 'collision_vehicle', 'success',
                             'demo_timeout', 'route_completion']})
PY
