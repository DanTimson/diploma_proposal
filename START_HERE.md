# D-004 — frontier and filtering controls

Applies to repository commit 33e1b20 (D smoke tests). No remote changes were made.
Read D-004-results.md first. Existing ACORN-1 largely fixes the earlier strict-filter
recall failure; exact prefiltering is faster on this 100k workload. This is not
positive evidence for a new graph algorithm. Thirty-two held-out queries remain unused.

Extract D-004-frontier-controls.patch into your repository root, then:

```bash
git apply --check D-004-frontier-controls.patch
git apply D-004-frontier-controls.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
bash scripts/run_d004.sh smoke
python scripts/package_d004.py --out D-004-review.zip local-runs/D-004-smoke
```

Send D-004-review.zip. Default tests run 47 tests with one external integration
skip; all 47 passed here with ACORN built. The patch applies cleanly to your updated
commit. Smoke runs need no download or ACORN installation. Use a new output name
if that directory already exists (`bash scripts/run_d004.sh smoke local-runs/D-004-smoke-2`).

The supplied pilot is already executed; no need to rerun it merely to read its
results. If desired, `bash scripts/run_d004.sh prepare` then `bash scripts/run_d004.sh
pilot` reproduce the development experiment. Allow several minutes plus the initial
~525 MB download; the full grid took ~104 seconds here. Commands for actual ACORN
and expanded official DiskANN search are in D-004-guide.md. The latter is prepared
but not run in these receipts. Do not launch simultaneous jobs for timing comparisons.

Receipts include per-query outputs and references, configuration/source hashes,
raw logs and query-split metadata. Large datasets, graph files and compiled binaries
are excluded. Early full-grid runs overlapped and are quality-only evidence;
D-004-timing / D-004-acorn*-timing are the sequential timing evidence. One pilot
source-hash timing discrepancy is explained in its provenance-note.json, with
both source versions retained; the measured computations were unchanged.
