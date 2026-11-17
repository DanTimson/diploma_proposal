# D-006: natural groups, filter controls, bounded mechanism exploration

This is a **development screen**, not evidence of a novel method. It follows
D-005's filter-only comparison by asking whether a one-per-group objective
changes what routing work is useful. It includes actual FAISS HNSW baselines
and deliberately simpler diagnostic graph traversal. It does not rename the
diagnostic traversal HNSW, ACORN, RACORN, or the published Diverse Search.

The patch adds files only. Its public-repository base is
`0c3ed5e15748547aa190141cedb49f05a33e1d2b`; the public checkout did not contain
D-005. No D-005 files or previous analyses are overwritten. Run your complete
local test suite after applying, since your local checkout is authoritative.

## Apply and verify

From the project root, after putting the patch there:

```bash
git apply --check D-006-natural-exploration.patch
git apply D-006-natural-exploration.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
bash scripts/run_d006.sh smoke
```

Existing ANN extras, g++, and FAISS 1.12.0 are required. If this is a fresh venv:
`python -m pip install -e '.[ann]'`. No new Python dependency is introduced.

Smoke uses FANNS-small: 1,000 database vectors, two queries, k=10/100, and
distance/shortlist budgets 64/128. Some k=100 conditions are infeasible on this
small dataset; this is recorded, not treated as successful retrieval of 100 items.
The original query file is about 164 MB even for FANNS-small.

## Recommended local experiment

Use your existing pinned Diverse DiskANN build if available:

```bash
BUILD_THREADS=4 bash scripts/run_d006.sh prepare --diskann external/DiskANN-diversity
bash scripts/run_d006.sh pilot
```

If the upstream executables are absent, the existing
`bash scripts/setup_diverse.sh` builds them using the previously documented
dependencies. D-006 checks the upstream revision and executable before starting
large downloads/builds. It does not install system packages or change BLAS defaults.

For an initial HNSW-only screen, omit `--diskann`. Choose this before preparation;
changing the builder set later requires a new `D006_OUT` directory. The complete
construction comparison needs both DiskANN graphs, so HNSW-only results cannot
settle that part of the question.

Inspect the pilot:

```bash
cat local-runs/D-006-fanns/runs/pilot/summary.json
cat local-runs/D-006-fanns/runs/pilot/em_replay.json
```

Then run the bounded screen, with no further implementation needed:

```bash
bash scripts/run_d006.sh screen
bash scripts/run_d006.sh package
```

Send **D-006-review.zip**. It includes the plan, environment/build receipts,
per-query rows, selections, exact references, metrics, counters, published EM
replay, and summary. It excludes vectors and indexes. To send the pilot instead:

```bash
python -m diploma_starters.explore package \
  --out local-runs/D-006-fanns --name pilot --destination D-006-pilot-review.zip
```

For foreground logging use `set -o pipefail` before piping a command to `tee`.
There is no background launcher, and `COMPLETE .../runs/screen` plus
`summary.json` status `complete` signals completion. A process interrupted after
some query/condition/k cells finishes can resume with **the identical command**.
Cells have plan and content checksums; a changed plan requires a new `--name`.
Completed graph builds have hash-checked checkpoints. An interrupted unfinished
graph build must restart; completed builds are reused. Do not run two preparations
into the same directory. Query runs reject concurrent use via `RUNNING.lock`.
Ctrl-C releases it. After SIGKILL or a machine crash, remove that run's lock only
after checking that its process is no longer running.

## What is frozen

- FANNS-medium: 100,000 vectors, 4,096 dimensions, original L2 representation.
  All five input files use the existing revision/size/SHA-256 locks.
- Groups: **identical sorted sets of `sub_categories`**. This is a natural metadata
  partition, not an assertion of shared authorship, dependence, or reputation.
- Filters: original per-query category-count equality (EM), plus update-date
  `<=` thresholds at database quantiles 0.1 and 0.5. Tied dates mean the actual
  eligible fraction need not be exactly 10%/50%; per-query counts are reported.
  No assumption about the date field's calendar epoch is needed.
- Conditions: groups alone; EM filter alone; EM+groups; each date filter alone
  and with groups. Category count is group-aligned when metadata is consistent;
  date filters can split groups. Actual numbers of split groups are recorded.
- Seed 29017 freezes 32 development and 32 disjoint held-out query IDs. The pilot
  uses the first four development queries; the screen uses the first 16.
  **Held-out query vectors are not included in the prepared workload and never
  evaluated by this runner.** Confirmation requires a subsequent frozen plan.
- k=10/100, budgets 256/1024, two timed repeats after one warmup, one search thread.
  These are screening settings, not a promise of reaching 95% recall. Larger
  budgets/k=50 can be added in a new named run if the observed frontier justifies it.

The original FANNS EM top-100 is replayed for each development query before its
cells. ID-set differences are reported separately from distance agreement at
rtol=1e-6, atol=1e-9. A disagreement beyond that tolerance aborts. Numeric agreement
does **not** establish that discrepant IDs are exact mathematical ties.
New grouped/date ground truth is exhaustive native float64-accumulated squared L2,
with smaller IDs breaking ties. The new conditions do not have published numerical
results; their purpose is a controlled extension of a benchmark with existing
results. Original published wall-clock numbers are not directly comparable to
this runner's measurements.

## Baselines and exploratory mechanisms

| Family | What is actually run |
|---|---|
| Exact | Existing compiled exhaustive selection of best eligible representative per group |
| FAISS HNSW post | Full HNSW search, overfetch, then eligibility and exact shortlist group selection |
| FAISS HNSW filter | Native IDSelectorBitmap for result eligibility, overfetch, then group selection |
| HNSW level 0 | Extracted level-zero graph, using the new common diagnostic traversal |
| Ordinary / diverse DiskANN construction | Actual pinned upstream graph builders, each using the same new diagnostic traversal |

Optional DiskANN construction is pinned at
`659041e1b12e3ab41c70ad12d11a29a883f17e15`: R=64, Lbuild=200,
alpha=1.2; diverse construction additionally uses NumDiverse=16 and natural groups.
The graph reader follows `src/in_mem_graph_store.cpp` at that revision, rejects
frozen-point/dynamic formats, and validates row counts and IDs. The common traversal
on a diverse-built graph is **not** the original Diverse Search baseline. A winning
construction/traversal combination must later face that published search too.
No ACORN/RACORN integration into this new grouped objective is claimed here.

Seven policy settings, rather than a full Cartesian product:

| Policy | Change from ordinary diagnostic traversal |
|---|---|
| ordinary | Single graph entry; no group cap; 128 pending nodes maximum |
| cap1 | At most one pending main-queue node per group |
| cap4 | At most four pending main-queue nodes per group |
| reserve | 96 main slots + 32 reserve slots; no group cap |
| cap1_reserve | Same reserve, main queue cap one per group |
| random4 | Four fixed uniformly sampled items as entry points |
| balanced4 | Four fixed uniformly sampled groups, then one item within each |

The reserve stores nearest evicted/rejected nodes, and supplies every fourth
expansion when available (or when the main queue empties). It is a routing
hypothesis, not a bridge detector. It shares the **same total pending capacity**,
rather than receiving free extra slots. Seen nodes are never reinserted.
Pending caps do not cap all discovered candidates or all expansions per group.
Eligible discoveries always update the best group representative even if they
are not retained for routing. Ineligible nodes can route in every diagnostic policy.

Entry sets are query-independent and prepared once; initial distance computations
count against the query budget. The single entry is each graph's stored entry.
The `random4` and `balanced4` sets are identical across graph constructions, making
them useful controls for construction comparisons. No upper-layer descent is
performed by the diagnostic kernel. This does not test rebuilding a hierarchy
with data-dependent promotion; that remains a later branch.

## Interpretation and resource bounds

The diagnostic budget is a **strict count of unique vector-distance evaluations**,
including entries; it can stop in the middle of a neighbor list. FAISS's budget
column instead means shortlist size and efSearch. Those are not equivalent units.
Compare diagnostic policies at matched distance budgets, and compare families at
matched measured recall/latency. Also inspect actual degree, neighbor-group count,
and stored bytes: equal R/M settings do not guarantee identical graph memory.

Query timing includes Python/native boundary costs, input validation, allocations,
bitmap packing for FAISS filtering, traversal, and final selection. Predicate
masks are precomputed for everyone; predicate evaluation/lookup cost is excluded.
FAISS candidate reranking uses float64. The native diagnostic kernel uses float64
distance accumulation; FAISS traversal/upstream graph construction use their own
arithmetic. Counters are enabled in timed diagnostic runs. This is not yet an
optimized production implementation, and wall-clock losses can reflect overhead.

Report item recall, group recall, fill, and representative distance regret together.
Finding the correct group but the wrong representative is not full correctness.
For infeasible k, recall/fill denominators use the available exact result length;
the summary prints the feasible-query count. Do not interpret these as success at
the requested k. Empty references have null recall. No tail-latency conclusion
should be drawn from four or 16 queries.

The pilot and screen run the same per-query configuration. **After preparation**,
screen search time should be approximately four times pilot search time, before
allowing for query difficulty and system noise. This is a calibration estimate,
not a bound. Preparation/build time is separate and printed per graph. Do not
extrapolate high-dimensional FANNS build time from SIFT or the 1,000-item smoke.
The three-graph path stores several full vector copies on disk: allow roughly
12–15 GB of free space including source data and upstream outputs. Query execution
should fit the user's 30 GiB machine; upstream build peak RAM is not hard-limited.
Four build threads are the default convenience setting; search stays single-threaded
for attribution. Multithreaded construction is not guaranteed bit-reproducible;
saved graph hashes identify the actual shared construction. Use BUILD_THREADS=1
if deterministic build scheduling matters more than build time.

## Decisions after this screen

1. First check original EM replay, zero constraint violations, feasible-query
   counts, and whether date filters actually split groups. Failure here means fix
   the workload/harness before interpreting algorithm rankings.
2. Check whether cap/reserve changes improve recall at matched distance budget
   on more than an isolated query, and whether the change survives representative
   correctness and measured latency. Inspect expansion and group-discovery counters.
3. Check whether a graph-construction effect persists with shared entry sets.
   The ordinary/diverse construction x ordinary/capped traversal cells are the
   controlled mechanism comparison, not a reproduction of the paper's 2x2 table.
4. A useful screening signal would be a repeatable >=10% work/latency improvement
   near a useful recall target without substantially worse failures elsewhere.
   This is a prospective triage threshold, not a significance test or universal
   novelty criterion. Smaller coherent effects can still motivate investigation.
5. Promote at most two mechanisms into a frozen confirmation plan with untouched
   queries, a second construction/entry seed, stronger published search baselines,
   and another dataset. Only then consider claims of general improvement.
6. If only existing graph construction or exact fallback wins, retain the negative
   finding and favor the comparative/engineering diploma scope. If no cap/entry
   mechanism helps, do not immediately launch a larger blind Cartesian sweep.

The D-005 1M exact-reference discrepancy is not resolved by a successful FANNS
replay: it belongs to a different dataset/harness and still needs its own audit.
Prior research notes and results remain in place.
