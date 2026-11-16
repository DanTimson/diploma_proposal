#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=.venv/bin/python
[[ -x "$python_bin" ]] || { echo 'Run bash scripts/setup_env.sh all first.'; exit 1; }
mode=${1:-smoke}
run_dir=${2:-local-runs/D-003-$mode}
case "$mode" in
 smoke) exec "$python_bin" -m diploma_starters.group_screen --n 1000 --queries 8 --out "$run_dir" ;;
 pilot) exec "$python_bin" -m diploma_starters.group_screen --dataset sift --n 100000 --queries 64 --k 100 --out "$run_dir" ;;
 upstream) exec "$python_bin" scripts/diskann_probe.py --run-dir "${2:-local-runs/D-003-pilot}" --upstream external/DiskANN-diversity --out "${3:-local-runs/D-003-upstream}" ;;
 *) echo 'Usage: bash scripts/run_d003.sh smoke|pilot [output-dir]'; echo '       bash scripts/run_d003.sh upstream [pilot-dir] [output-dir]'; exit 2 ;;
esac
