# D-001 handoff

Patch base: `900040d9b02723d2b8e00a5aecfabbe3aede3606` (`init`) from `DanTimson/diploma_proposal`.

No remote changes, commits, PRs or pushes were made. The patch was prepared in an isolated checkout. Existing A/B/C proposals and result files are preserved; README and BRIEFING gain the dated D-first decision, optional dependencies/entrypoint are added, and source locks are extended.

From your repository root, inspect then apply the downloaded patch (substitute its actual path):

```bash
git status --short
git rev-parse HEAD
git apply --check /path/to/D-001-hnsw-smoke.patch
git apply /path/to/D-001-hnsw-smoke.patch
git diff --check
git status --short
```

If `--check` fails, do not force it or reset local work; provide the error/current diff for a rebased patch. Newly created files are untracked until you choose to stage them; plain `git diff` does not display untracked contents. Review `topics/D_hnsw/`, the three `ann*.py` modules, `tests/test_ann.py`, and the summary JSON as well.

Validate in an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ann]'
python -m unittest discover -s tests -v
diploma-ann --out local-runs/d-local-clean.json synthetic --mode clean
```

Expected: 25 tests pass with the ANN extra installed. Tests do not assert timings. See README for real-data commands and protocol for interpretation. The review patch contains aggregate results only. `D-001-run-receipts.zip` contains per-query JSON from the final seven workloads plus preserved earlier diagnostic runs; they are not required to execute the code. Run hashes and implementation hashes are in the aggregate record.

Suggested next discussion: confirm D's primary meaning (retrieval index versus influence graph); select a medium-scale workload; agree tolerable quality loss and one bounded iteration before considering A/B. The present patch makes the retrieval interpretation executable without committing to a custom topology.
