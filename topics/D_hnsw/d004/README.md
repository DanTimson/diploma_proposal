# D-004: filtering, group caps, and genuine continuation

Base: commit `33e1b20adf43066f2f4da5ef4dc88351993e2232` (D-003).
The patch adds an experiment, not a claim of a new ANN algorithm.
Read RESULTS.md for measured outcomes and limits.

## Questions this experiment separates

1. Does applying a group cap before eligibility discard the only useful representative?
2. Does preserving the frontier avoid work repeated by increasing-budget restarts?
3. Does an existing stronger filtered-search implementation already resolve the apparent gap?

The exact objective remains: up to k eligible items, at most one per fixed group;
maximize cardinality first, then minimize additive squared L2 distance.
This is not truth inference, reputation estimation, arbitrary diversity, or adversarial identity detection.

## Components and their meaning

| Component | What it measures | What it does not establish |
|---|---|---|
| Compiled exact prefilter | Full constrained optimum; float64 L2 accumulation | ANN scalability at millions/billions |
| FAISS native selector | Existing D-003 HNSW with larger initial pool | ACORN; resumable traversal |
| Custom resume/restart pair | Same exported HNSW level-0 graph, same search/output rules | Faithful FAISS HNSW search or a competitive new method |
| ACORN-1 adapter | Actual pinned upstream search + timed exact reranking/group selection | Tuned ACORN-gamma, numerical paper reproduction |
| Expanded DiskANN adapter | Official per-seller search caps 1/4/16/64; construction NumDiverse=16 | A measured result until explicitly run |

The custom traversal uses a fixed entry, a nearest-first frontier, and a visited
set retained between checkpoints. It omits the hierarchical entry descent and
retains every discovered candidate; frontier memory can grow linearly with N.
Exported int64 adjacency also duplicates the original FAISS graph. Its caps are
shortlist caps only, **not** the per-group queue restrictions in DiskANN-diversity.
Do not compare their cap values as if the algorithms were identical.
Ineligible nodes remain routing nodes. Predicate-first output selection is
not predicate-aware graph traversal.

At each checkpoint, the restart control begins again and repeats all previous
checkpoints; the resume control retains its state. Both report cumulative
distance evaluations and cumulative latency INCLUDING every checkpoint's
selection. These are costs of a progressive schedule, not one-shot search at
the last budget. A nonempty frontier is not a certificate of exactness;
disconnected graphs can leave unseen points even after frontier exhaustion.

## Data and query separation

Preparation replays D-003's 100k database with its original 64-query RNG
consumption, preserving vectors, group labels, ranks and base IDs. It then
samples 64 different SIFT queries, excluding all 64 D-003 IDs. The first 32
are development queries; the other 32 are held out. Group labels and ranks
are never regenerated in response to the fresh query count. Dataset arrays,
query IDs and the split are hashed before measurement.

All supplied pilot results use **tune only**. Do not inspect test results and
then continue tuning while calling those queries held out. Freeze the chosen
configuration and decision rule first. For a confirmatory claim, use multiple
index seeds and query-level uncertainty intervals; 32 development queries and
one index seed are only a screen.

These are published SIFT vectors with newly generated group/predicate labels.
They overlap prior benchmark practice but do not reproduce published numerical
tables or establish a natural reputation/recommendation task.

## Apply and smoke (from repository root)

```bash
git apply --check D-004-frontier-controls.patch
git apply D-004-frontier-controls.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
bash scripts/run_d004.sh smoke
```

Default tests skip the optional external ACORN integration check. The smoke
does not download data, build ACORN, or use test queries. A working g++ is
required, as in D-003. Output directories must be new; existing receipts are
never silently overwritten. Use a different output suffix for a rerun.

## Development pilot

```bash
bash scripts/run_d004.sh prepare
bash scripts/run_d004.sh pilot
```

Preparation downloads about 525 MB if the existing SIFT cache is absent.
The pilot uses one CPU thread, N=100,000, k=100, pool=5,000, expansion
checkpoints 128/512/2048, caps 1/4/16/unlimited, and 10%/1% eligibility.
Budgeted expansion counts are not equal to distance evaluations; both are
reported. The cap=0 spelling means unlimited. A larger initial FAISS pool
changes its operating point relative to D-003; an equal ef label does not
make those earlier measurements directly comparable.

Run timed experiments sequentially, not beside other CPU-heavy sweeps.
Use the measured timings in RESULTS.md as machine-specific context, not a
promise for another CPU. There is no GPU kernel in this experiment.

## Actual ACORN baseline (optional external build)

Install compiler/build dependencies if missing; these commands are for Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libblas-dev liblapack-dev zlib1g-dev
bash scripts/setup_acorn.sh
D004_ACORN_BINARY=.local/d004-native/acorn_probe python -m unittest discover -s tests -p test_acorn_integration.py -v
bash scripts/run_d004.sh acorn
```

The setup pins upstream commit `c259f11cdbec880671658eaa0d8747e2cc9de79d`
and applies one CMake build-interface correction. No upstream search or
construction code is edited. `CMAKE`, `CXX`, `BUILD_JOBS`, and `ACORN_DIR`
can override local tooling; extra arguments go to CMake. Unexpected tracked
upstream modifications cause an error. The binary/library/source hashes are
checked before running. The upstream FAISS fork runs in a separate process
to avoid symbol collisions with pip's FAISS.

The upstream initial-candidate path can emit an ineligible candidate. The
adapter rechecks eligibility and records `upstream_ineligible_candidates`;
invalid IDs still fail the run. A dedicated integration test checks empty,
singleton and all-eligible masks. Final selection is timed with search.
ACORN construction uses rank-bucket metadata; the exact query bitmap governs
eligibility. It does not use diversity-group labels as filter labels.

The simple CLI rebuilds the same seeded index for each pool/condition. Build
time is excluded from query latency and retained in logs. This costs extra
wall time but keeps each subprocess independently reproducible. `--gamma 12`
is available for a separate run; it is not a tested default or an assertion
that this is the best value. The upstream fork and pip FAISS have different
versions/builds, another reason to avoid algorithm-wide speed claims.

## Official diverse-search extension

Once the D-003 upstream DiskANN build is available:

```bash
bash scripts/run_d004.sh upstream
```

This exports only the development split, then runs ordinary search and caps
1/4/16/64 on both standard and diverse graphs, with NumDiverse=16 on the latter.
It is still post-filter composition. The upstream CLI timings exclude Python
selection and must not enter the end-to-end timing table. This optional preset
has not been benchmarked in the supplied D-004 receipts.

## Send results back

```bash
python scripts/package_d004.py --out D-004-review.zip local-runs/D-004-smoke
```

Add completed pilot/ACORN/upstream output directories as more arguments if run.
The archive includes JSON and logs, excluding datasets, graph indices and
native libraries. No repository commit or push is required for result review.

## Sources

- ACORN paper: https://dl.acm.org/doi/10.1145/3654923 ; official implementation:
  https://github.com/guestrin-lab/ACORN/tree/c259f11cdbec880671658eaa0d8747e2cc9de79d
- Graph-Based Algorithms for Diverse Similarity Search (ICML 2025):
  https://proceedings.mlr.press/v267/anand25a.html ; upstream pinned in D-003.
- pgvector iterative scans are an existing engineering precedent for extending
  searches after filtering, not evidence of novelty in this control:
  https://github.com/pgvector/pgvector#iterative-index-scans
- D-003's sources.json locks the SIFT mirror and SHA-256 values.
