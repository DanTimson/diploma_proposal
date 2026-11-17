#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
action="${1:-help}";if (($#));then shift;fi
case "$action" in
 smoke)
  python -m diploma_starters.final_screen run --prepared local-runs/D-006-smoke \
    --out local-runs/D-008-smoke --queries 2 --repeats 1 --smoke "$@";;
 screen)
  python -m diploma_starters.final_screen run --prepared "${D006_PREPARED:-local-runs/D-006-fanns}" \
    --out local-runs/D-008-final --queries 16 --repeats 3 \
    --upstream "${D007_UPSTREAM:-external/DiskANN-diversity}" "$@";;
 decide)
  python -m diploma_starters.d008_decision --out local-runs/D-008-final "$@";;
 package)
  python -m diploma_starters.final_screen package --out local-runs/D-008-final \
    --destination D-008-review.zip "$@";;
 *) echo 'Usage: bash scripts/run_d008.sh {smoke|screen|decide|package}';;
esac
