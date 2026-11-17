# D-005 decision round

Base: 0c3ed5e (test runs). No remote repository changes were made.
Read ROUND3-REVIEW.md for the adjudication and measured FANNS/B/kernel checks.
The next user-run experiment is same-graph RACORN calibration, replacing the
previously suggested blind over-fetch expansion as the immediate priority.

Extract the patch into the project root, then:

```bash
git apply --check D-005-decision-round.patch
git apply D-005-decision-round.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
bash scripts/setup_racorn.sh
python scripts/run_racorn_control.py smoke --out local-runs/D-005-racorn-smoke
python scripts/run_racorn_control.py pilot --n 100000 --out local-runs/D-005-racorn-100k
python scripts/run_racorn_control.py pilot --n 1000000 --out local-runs/D-005-racorn-1m
python scripts/package_d004.py --out D-005-review.zip local-runs/D-005-racorn-smoke local-runs/D-005-racorn-100k local-runs/D-005-racorn-1m
```

Run sequentially and stop if a command fails. Send D-005-review.zip after
completion, or the failed run's launcher.log. New output directories are
required. The 100k/1M runs are prepared but not executed in this delivery.
Keep the D-004 held-out queries unused; the runner uses development IDs only.

Expected tests: 50 run, 49 pass, one optional original-ACORN test skipped.
The pinned RACORN driver was compiled and smoke-tested separately.
RUN_GUIDE.md explains runtime estimates, scope, prerequisites, and optional
reproduction commands for the already executed metadata/kernel/B checks.

The sparse exact backend is optional; old dense behavior and stored D-004
results remain unchanged. The B pilot tests literal copies and text dedup,
not new inference, full-corpus retrieval, natural independence, or ANN utility.
