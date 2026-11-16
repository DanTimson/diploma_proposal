#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py=.venv/bin/python
[[ -x "$py" ]] || { echo 'Run bash scripts/setup_env.sh all first.'; exit 1; }
mode=${1:-smoke}
case "$mode" in
 smoke)
  root=${2:-local-runs/D-004-smoke}
  "$py" -m diploma_starters.frontier_screen prepare --out "$root-data"
  exec "$py" -m diploma_starters.frontier_screen screen --workload "$root-data" --out "$root" ;;
 prepare)
  exec "$py" -m diploma_starters.frontier_screen prepare --dataset sift --n 100000 \
   --old-queries 64 --queries 64 --out "${2:-local-runs/D-004-sift-data}" ;;
 pilot)
  exec "$py" -m diploma_starters.frontier_screen screen --workload "${2:-local-runs/D-004-sift-data}" \
   --out "${3:-local-runs/D-004-pilot}" --k 100 --pool 5000 --budgets 128 512 2048 ;;
 acorn)
  exec "$py" scripts/acorn_probe.py --workload "${2:-local-runs/D-004-sift-data}" \
   --out "${3:-local-runs/D-004-acorn}" ;;
 upstream)
  data=${2:-local-runs/D-004-sift-data}; out=${3:-local-runs/D-004-upstream}
  "$py" -m diploma_starters.frontier_screen export-upstream --workload "$data" --out "$out-input"
  exec "$py" scripts/diskann_probe.py --run-dir "$out-input" --upstream external/DiskANN-diversity \
   --out "$out" --group-caps 1 4 16 64 --num-diverse 16 ;;
 *) echo 'Usage: bash scripts/run_d004.sh smoke|prepare|pilot|acorn|upstream [data-dir] [output-dir]'; exit 2 ;;
esac
