#!/usr/bin/env bash
# Run with bash; activation must happen in the calling shell afterwards.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
profile="${1:-audit}"
case "$profile" in
  base) requirement='.' ;;
  audit) requirement='.[audit]' ;;
  ann) requirement='.[ann]' ;;
  all) requirement='.[audit,ann]' ;;
  *) printf 'Usage: bash scripts/setup_env.sh [base|audit|ann|all]\n' >&2; exit 2 ;;
esac
if [[ "$profile" == ann || "$profile" == all ]]; then
  if ! grep -Eq '^ann[[:space:]]*=' pyproject.toml; then
    printf 'The ann extra is absent. Apply D-001 first, or use the audit profile.\n' >&2
    exit 2
  fi
fi

if [[ -e .venv || -L .venv ]]; then
  if [[ ! -f .venv/pyvenv.cfg || ! -x .venv/bin/python ]]; then
    printf '.venv exists but is not a usable Linux/WSL venv; leaving it untouched.\n' >&2
    exit 2
  fi
fi

if command -v uv >/dev/null 2>&1; then
  if [[ ! -e .venv ]]; then
    uv venv --python "${PYTHON_BIN:-3.12}" .venv
  fi
  uv pip install --python .venv/bin/python -e "$requirement"
else
  if [[ ! -e .venv ]]; then
    python_bin="${PYTHON_BIN:-python3.12}"
    if ! command -v "$python_bin" >/dev/null 2>&1; then
      printf 'Install uv or Python 3.12, or set PYTHON_BIN to an available supported Python.\n' >&2
      exit 2
    fi
    "$python_bin" -m venv .venv
  fi
  .venv/bin/python -m pip install -e "$requirement"
fi

mkdir -p .local/env local-runs external
receipt="$(.venv/bin/python -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')"
.venv/bin/python --version > ".local/env/$receipt-python.txt"
if command -v uv >/dev/null 2>&1; then
  uv pip freeze --python .venv/bin/python > ".local/env/$receipt-packages.txt"
else
  .venv/bin/python -m pip freeze --all > ".local/env/$receipt-packages.txt"
fi
printf 'Ready: %s/.venv (%s profile).\n' "$repo_root" "$profile"
printf 'Activate in your shell: source .venv/bin/activate\n'
printf 'Verify: python -m unittest discover -s tests -v\n'
