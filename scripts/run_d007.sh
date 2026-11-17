#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
action="${1:-help}"
if (($#)); then shift; fi
prepared="${D006_PREPARED:-local-runs/D-006-fanns}"
upstream="${D007_UPSTREAM:-external/DiskANN-diversity}"
case "$action" in
  smoke)
    python -m diploma_starters.baseline_controls run --prepared local-runs/D-006-smoke \
      --out local-runs/D-007-smoke --queries 2 --repeats 1 --smoke "$@"
    ;;
  upstream-smoke)
    python -m diploma_starters.baseline_controls run --prepared "$prepared" \
      --out local-runs/D-007-upstream-smoke --queries 2 --repeats 1 --smoke --upstream "$upstream" "$@"
    ;;
  pilot)
    python -m diploma_starters.baseline_controls run --prepared "$prepared" \
      --out local-runs/D-007-pilot --queries 4 --upstream "$upstream" "$@"
    ;;
  screen)
    python -m diploma_starters.baseline_controls run --prepared "$prepared" \
      --out local-runs/D-007-controls --queries 16 --upstream "$upstream" "$@"
    ;;
  package)
    python -m diploma_starters.baseline_controls package --out local-runs/D-007-controls \
      --destination D-007-review.zip "$@"
    ;;
  *)
    echo 'Usage: bash scripts/run_d007.sh {smoke|upstream-smoke|pilot|screen|package} [CLI options]'
    echo 'Reuses D006_PREPARED and D007_UPSTREAM; defaults to existing D-006 paths.'
    ;;
esac
