# Local development setup

Use WSL/Linux and a project-local `.venv`. The setup helper prefers `uv`, with Python 3.12 for a new environment, and falls back to `python3.12 -m venv` plus pip. An existing environment is reused, not deleted. Set `PYTHON_BIN` to override the interpreter used to create a new environment; it does not change an existing venv. The helper installs dependencies from the network; uv may download Python if needed.

From the repository root:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
diploma-starters smoke
```

Default `audit` installs the existing audit tools. Available profiles are `base` (standard-library tools), `audit`, `ann`, and `all`. The last two require D-001's `ann` extra and fail clearly if it is not present. After applying D-001, reuse the same venv:

```bash
bash scripts/setup_env.sh all
source .venv/bin/activate
python -m unittest discover -s tests -v
diploma-ann --out local-runs/d-local-clean.json synthetic --mode clean
```

The initial repository has 11 tests; with D-001 and the ANN dependencies, 25 should execute and pass. ANN tests skipped due to missing dependencies are not an ANN validation. The shell helper cannot activate its parent shell, so `source` is a separate command. No system Python packages are modified.

## What stays local

| Location | Intended content | Tracked? |
|---|---|---|
| `.venv/` | Python environment | No |
| root `*.patch`, `*.diff` | Downloaded review patches | No |
| `.local/` | Personal notes, temporary reports, scratch artifacts, environment receipts | No |
| `external/` | Downloaded datasets and third-party assets | No |
| `local-runs/` | Raw/generated runs and disposable outputs | No |
| `results/` | Deliberately selected, reviewed evidence for the project | Yes |
| `evidence/`, `topics/`, `src/`, `tests/` | Source records, research and implementation | Yes |

Ignore rules for patches are root-only, so future patch fixtures elsewhere are not hidden. JSON, CSV, Markdown, PDFs and ZIPs are not globally ignored: place local-only artifacts under `.local/`. The setup helper saves timestamped Python/package receipts in `.local/env/`; these record the installed environment but are not a portable dependency lockfile. Re-running setup can update unpinned dependencies. A dedicated lockfile can be introduced once the dependency set settles.

For preferences that should apply only to your checkout, edit the path returned by:

```bash
git rev-parse --git-path info/exclude
```

For example, `/my-notes/` or `/personal-report.pdf` in that file affects only your checkout. This is an alternative to adding a shared `.gitignore` rule. Git ignores do not untrack existing files and do not make local files private from other processes or backups. If you accidentally tracked a local artifact, inspect it before explicitly removing it from the index; setup performs no automatic untracking.

## Applying preparation and D-001 in either order

Apply **one** preparation variant, according to your local checkout:

- Initial repository, no D-001: use `PREP-001-initial.patch`.
- D-001 already applied: use `PREP-001-after-D001.patch`.

Always run `git apply --check PATH` before `git apply PATH`. The variants produce the same setup files. They differ only in whether `.gitignore` already exists.

If preparation is applied **before** the original D-001 patch, D-001's smaller `.gitignore` addition is redundant. Apply D-001 with that one file excluded:

```bash
git apply --check --exclude=.gitignore /path/to/D-001-hnsw-smoke.patch
git apply --exclude=.gitignore /path/to/D-001-hnsw-smoke.patch
```

The preparation ignore file already includes all D-001 rules. Do not exclude any other files or force a failed application; use a rebased patch if the checkout has other conflicting changes.
