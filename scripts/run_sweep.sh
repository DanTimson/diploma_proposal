#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
action="${1:-start}"
run_dir="${2:-local-runs/D-002-sweep}"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Run bash scripts/setup_env.sh all first.\n' >&2; exit 2
fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
case "$action" in
  start|resume|retry)
    mkdir -p -- "$run_dir"
    extra=()
    if [[ "$action" == retry ]]; then extra+=(--retry-failed); fi
    nohup .venv/bin/python -m diploma_starters.sweep --run-dir "$run_dir" "${extra[@]}" >> "$run_dir/launcher.log" 2>&1 < /dev/null &
    printf 'Launched process %s. Check startup/errors: tail -f %q/launcher.log\n' "$!" "$run_dir"
    printf 'Summary: %s/SUMMARY.md\nKeep Windows awake; terminal detachment does not prevent system sleep.\n' "$run_dir"
    ;;
  status) .venv/bin/python -m diploma_starters.sweep --run-dir "$run_dir" --status ;;
  stop) .venv/bin/python -m diploma_starters.sweep --run-dir "$run_dir" --stop ;;
  *) printf 'Usage: bash scripts/run_sweep.sh [start|resume|retry|status|stop] [run-directory]\n' >&2;exit 2 ;;
esac
