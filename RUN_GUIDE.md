# D-005 / B-001: review-driven decision round

Patch base: `0c3ed5e15748547aa190141cedb49f05a33e1d2b`.
Read REVIEW.md first. This patch preserves old results and the default exact
kernel. It adds checks and a bounded next control, not a new research method.

## Apply and test

```bash
git apply --check D-005-decision-round.patch
git apply D-005-decision-round.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Expected default: 50 tests, one optional ACORN test skipped unless its binary
is explicitly enabled. No extra dependency is needed for the native exact check.

## The next user-run experiment

```bash
bash scripts/setup_racorn.sh
python scripts/run_racorn_control.py smoke --out local-runs/D-005-racorn-smoke
python scripts/run_racorn_control.py pilot --n 100000 --out local-runs/D-005-racorn-100k
python scripts/run_racorn_control.py pilot --n 1000000 --out local-runs/D-005-racorn-1m
```

Run these sequentially. The build uses the upstream documented compiler command
and pinned NumKong AVX2/FMA kernels; it requires a suitable x86-64 CPU and g++.
No MKL, CUDA or new system packages are required by this build. `RACORN_DIR` and
`CXX` can change the checkout location/compiler; if moving it, pass `--upstream`
to the runner. Existing unexpected revisions/modifications are rejected.

The 100k and 1M runs are **not executed in this delivery**. Allow roughly 2–10
minutes for 100k and 10–45 minutes for 1M including graph construction; these
are provisional scheduling estimates, not measured runtimes. Each process has
a two-hour timeout (`--timeout` overrides it). First-time SIFT download adds
about 521 MB; an existing matching `external/sift1m` cache is reused. Logs are
written during execution to `<output>/launcher.log`. Output directories must
be new; failed runs are preserved and are not automatically resumed.

Protocol:

- Actual unmodified RACORN artifact, revision
  `b1a587ad2f5d4048cf50c8e2710e5070864590c3`; NumKong
  `09aaff9d20dadbdbe133d3aba41b3c768c49888a`.
- HNSW, ACORN-1, RACORN-1, exact prefilter with supplied passing IDs, and exact
  prefilter including predicate scan. All methods share the artifact's graph,
  distance implementation, predicate generation and measurement driver.
- M=16, construction ef=100; one build/search thread. This differs from D-004;
  compare methods **within this run**, not isolated latency numbers across runs.
- Selectivities 1%, 10%, 50%; upstream allowlist alpha=0 and -0.8; search ef
  100/200/400/800/1600; k=100; three measured passes after one warmup.
  The entry gate can dispatch ACORN/RACORN to ordinary HNSW at high local
  selectivity. The artifact's measured correlation `ce` is retained.
- Exactly the 32 D-004 development query IDs, read from the committed frozen
  manifest. No new query selection and no held-out queries. The base is the
  first N SIFT vectors (nested prefixes), not D-004's random 100k sample.
- Each run gets fresh private index/GT directories to avoid stale cache reuse.
  The upstream driver's GT and metrics are retained. This adapter does not
  export and independently re-evaluate raw result IDs; do not claim that level
  of validation or numerical reproduction of the paper's tables.
- **Filter-only control, no group uniqueness.** The upstream driver's word
  "group" means a workload cell, not our diversity-group constraint.

Predeclared reading rule: compare the fastest tested configuration reaching
mean recall >=0.95 in each cell. Record failures to reach the target rather
than comparing low-recall fast runs. The summary selects configurations on
development queries only; no claim of unbiased held-out superiority follows.

This is baseline calibration, not the diploma contribution. If ordinary
existing methods explain the effect, do not invent a group-aware variant
until a natural grouped workload exposes a residual tradeoff. If a crossover
appears, it establishes a regime worth testing, not application utility.

Package completed runs with the existing packager:

```bash
python scripts/package_d004.py --out D-005-review.zip \
  local-runs/D-005-racorn-smoke \
  local-runs/D-005-racorn-100k \
  local-runs/D-005-racorn-1m
```

If a run fails, send its launcher log before rerunning. Do not include an
incomplete directory in that packaging command.

## Additional checks already executed locally

```bash
python scripts/audit_fanns_groups.py --out local-runs/D-005-fanns-groups.json
python scripts/check_exact_backends.py --out local-runs/D-005-exact-check.json
```

The audit downloads/verifies only the 41 MB metadata file, not the 1.64 GB
medium vectors or 44.8 GB large vectors. The exact check uses synthetic vectors
and compares identical answers before timing dense and sparse aggregation.
The sparse kernel is opt-in via `NativeExact(..., implementation='sparse')`;
it is not assumed universally faster. The wrapper still enumerates an N-sized
eligibility bitmap, so total query cost is not solely proportional to N*s*d.

## B-001 saved-prediction pilot

The bundle already contains a complete baseline-plus-copy run. Reproduction,
if the original extraction is at `external/scifact-open`:

```bash
python scripts/evidence_copy_pilot.py --data-dir external/scifact-open \
  --out local-runs/B-001-copy-pilot
```

Requires pandas and pyarrow (the existing `audit` extra). Use the actual path
to the previously extracted SciFact-Open release. The required files are
`data/claims.jsonl`, `data/corpus_candidates.jsonl` and
`prediction/model_predictions.parqet` (upstream spelling).

No embeddings or new model inference: fixed MultiVerS-20 predictions, 50
candidates per claim. Only doc ID, predicted label, confidence and observable
text hash reach selection. No annotation ranks, gold labels or provenance
features are used. Exact copies of the highest-confidence candidate are added
0/1/5/20 times; selection budgets are 5/10/20. This is a deterministic mechanism
check, not three pseudo-independent seed trials.

Metrics separate evidence-document retrieval from correctly signed predictions.
Duplicate slots cannot multiply true positives. Text hashing removes literal
copies; it does not discover natural source independence. Results and ceilings
are in REVIEW.md. MMR, stance-aware selection, paraphrases, full-corpus retrieval,
and a second corpus remain future work, not implemented claims.
