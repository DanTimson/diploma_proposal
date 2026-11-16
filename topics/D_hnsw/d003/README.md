# D-003: baseline and objective selection

This milestone changes the question from "is HNSW useful?" to "what happens when
eligibility and a partition cap interact, against an existing diverse-search algorithm?"
D-001/D-002 remain unchanged. Do not compare latency numbers across their protocols.

## Scope and evidence levels

1. **Algorithm artifact:** upstream Microsoft DiskANN `diversity` branch at
   `659041e1b12e3ab41c70ad12d11a29a883f17e15`; original algorithms, no source edits.
2. **Adapted benchmark:** SIFT real vectors with newly seeded synthetic groups.
   This is not a numerical reproduction of the ICML paper's plots. The paper's
   exact assignment/seed and all build knobs have not been recovered.
3. **New diagnostic:** independent eligibility masks on the same vectors and
   groups; four conditions. It is not natural provenance, relevance, or fraud data.
4. **No new algorithm claim:** upstream group-aware search and a relaxed internal
   group cap are controls. No learned graph, ACORN adapter, or welfare method is
   implemented in this patch.

Read RESULTS.md for measured outcomes and DATA_AUDIT.md for source limitations.

## Run the self-contained screen (Linux/WSL)

Requires existing ANN dependencies and `g++` (no GPU). C++ exact baseline is built
locally into `.local/d003-native`; no system installation by our code.

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
bash scripts/run_d003.sh smoke
bash scripts/run_d003.sh pilot
```

The pilot explicitly downloads ~525 MB of checksummed SIFT files if absent.
Outputs include vectors, IDs, query/attribute assignments, references, selected
IDs, per-query metrics, checkpoints, and manifests. Existing output directories
are refused: choose a new directory for each experiment. These runs are foreground,
not D-002 background jobs; leave the terminal open. No resume is implemented.

Defaults: random 100k subset, 64 randomly sampled queries, k=100, data seed
20261115, index seed 11, HNSW M=32/efConstruction=200; search ef=128/512, initial
pool=200, cap=5000. Both ef settings are diagnostic, not tuned winners. First half
of sampled queries is marked tuning, second half confirmation. Neither partition
supports a population-level 99% guarantee at this size. Masks keep 50%, 10%, 1%
of items and are independent of vector similarity and groups; generated masks
are nested per query. Changing sample dimensions currently regenerates metadata;
for a future query-only extension, preserve the exported workload and its groups
rather than assume changing --queries keeps the groups fixed. 80% of items receive group 0, others uniformly one of 999
groups for the 100k case, following the paper's SIFT-skewed distribution.

The graph is reused across conditions. Eligibility is applied before final group
selection. `hnsw_post` restarts with larger pools; it is NOT resumable traversal.
`hnsw_native` uses FAISS IDSelector; it is NOT ACORN.

## Upstream diverse-search adapter

Typical Ubuntu build prerequisites, installed by you if missing:

```bash
sudo apt install build-essential cmake libboost-program-options-dev libaio-dev libgoogle-perftools-dev libmkl-full-dev
bash scripts/setup_diverse.sh
bash scripts/run_d003.sh upstream
```

Package availability varies by Ubuntu release; do not substitute MKL packages
without checking your package manager. setup_diverse.sh never installs packages
or modifies an existing different checkout. Pass custom CMake paths as additional
arguments (MKL_PATH, MKL_INCLUDE_PATH, OMP_PATH, CMAKE_PREFIX_PATH). BUILD_JOBS=2
by default. `-include cstdint` is a compile-only workaround for the upstream
missing include on recent GCC. Our recorded host build used private Debian package
extractions; the exact CMake command is in delivered receipts, not an Ubuntu recipe.

Upstream builds standard and diversity-enabled graphs, runs ordinary search and
diverse search with internal per-group limits 1 and 4, then independently selects
eligible one-per-group results. Build degree=64, Lbuild=200, alpha=1.2; all build
and search threads=1. Construction blocker threshold `--num-diverse` defaults to 1 (upstream default).
This weak construction setting is NOT a tuned reproduction of the paper; values
such as 8/16/32 must be studied before judging diverse construction.
Search candidate counts=200/800/3200. These settings are
explicit diagnostic choices, not claimed to recover every paper parameter.

**Critical:** upstream CLI dispatches filtered search and diverse search separately.
Our joint condition is *diverse search then post-filter*, not native filtered-diverse
traversal. Cap 4 applies to its candidate queue; final output cap remains 1.
Logs report upstream mean search latency, excluding our later mask/group selection.
Do not compare those numbers to the Python screen's end-to-end p50 or claim a
speedup from CLI elapsed wall time (which includes loading and saving).

## Exact controls and metrics

- `exact_cpp`: direct eligible-vector scan, double accumulation, group maxima and
  partial top-k in C++. Eligible-ID enumeration and temporary allocation are timed.
- `exact_faiss`: compiled FAISS Flat with selector and expanding candidate pool;
  pool may reach n to guarantee sufficient exact candidates. Includes wrapper cost.
- Neither is assumed universally optimal; batched throughput/GPU remain separate.
- **Item recall** is the primary approximation metric for matching the published
  diverse-NN definition. Group recall alone can hide a wrong representative.
- Record group recall, attainable fill, eligibility/group violations, average
  squared-L2 regret and distance ratio. Infeasible k uses the exact attainable
  cardinality as denominator; raw selections allow inspection.
- Score regret and ratio are computed only at equal cardinality. No division by a
  signed inner product. These metrics are geometric, not application utility.
- Tie handling uses ID order; tied alternatives can affect item recall without
  changing the objective. No result should be accepted from recall alone.

## Next decision

Do not launch ever-larger sweeps automatically. First inspect whether ordinary
larger pools or a looser upstream candidate-group cap already repair a failure.
Only then consider a native combined traversal, compared against ACORN + group
selection, exact-subset search, true resumable traversal, and per-group methods.
Keep degree/memory/build budgets explicit. Separate filter selectivity from its
correlation with geometry and groups; current masks test only independent filters.

A systems diploma can be framed as a reproducible comparative study with a targeted
extension, subject to supervisor acceptance. A commercial opportunity or reputation
benefit is not established by these experiments. Natural groups and an externally
measured utility objective remain prerequisites for application-level claims.

## Share a run

```bash
python scripts/package_d003.py --out .local/D-003-review.zip local-runs/D-003-pilot local-runs/D-003-upstream
```

Send `.local/D-003-review.zip`. The packer keeps JSON, logs, and result ID/distance
binaries; excludes source vectors and index files; requires completed runs and
refuses overwrites. Keep the original run folders locally.
