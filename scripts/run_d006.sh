#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
action="${1:-help}"
if (($#)); then shift; fi
out="${D006_OUT:-local-runs/D-006-fanns}"
case "$action" in
  smoke)
    python -m diploma_starters.explore prepare --dataset small --out local-runs/D-006-smoke --queries 2 --build-threads 1 "$@"
    python -m diploma_starters.explore run --out local-runs/D-006-smoke --name smoke --query-count 2 --ks 10 100 --budgets 64 128 --repeats 1
    ;;
  prepare)
    python -m diploma_starters.explore prepare --dataset medium --out "$out" --queries 32 --build-threads "${BUILD_THREADS:-4}" "$@"
    ;;
  pilot)
    python -m diploma_starters.explore run --out "$out" --name pilot --query-count 4 --ks 10 100 --budgets 256 1024 --repeats 2 "$@"
    ;;
  screen)
    python -m diploma_starters.explore run --out "$out" --name screen --query-count 16 --ks 10 100 --budgets 256 1024 --repeats 2 "$@"
    ;;
  package)
    python -m diploma_starters.explore package --out "$out" --name screen --destination D-006-review.zip "$@"
    ;;
  *)
    echo 'Usage: bash scripts/run_d006.sh {smoke|prepare|pilot|screen|package} [CLI options]'
    echo 'Optional graph builders: prepare --diskann external/DiskANN-diversity'
    echo 'Environment: BUILD_THREADS=4, D006_OUT=local-runs/D-006-fanns'
    ;;
esac
