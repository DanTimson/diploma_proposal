# D-007: baseline and instrumentation controls

This additive patch requires D-006. It reuses its vectors, development query IDs,
natural groups, predicates, and all saved graph constructions. It changes no D-006
files, upstream tracked files, original upstream executables, or saved indexes.
All expensive experiments are intended for the user's machine.

The D-006 screen and budget check established that the tested caps/reserves mostly
lost recall or time, and random multistart's aggregate advantage largely vanished
at 4,096 distances. This round tests whether apparent opportunities survive stronger
baselines. It is not another search for new mechanisms and is not held-out confirmation.

## Apply and run

After extracting the delivery ZIP into `.local/d007-delivery`:

```bash
source .venv/bin/activate
git apply --check .local/d007-delivery/D-007-baseline-controls.patch
git apply .local/d007-delivery/D-007-baseline-controls.patch
python -m unittest discover -s tests -v
bash scripts/run_d007.sh upstream-smoke
```

The upstream smoke uses **two of the same development queries**, the existing
100k indexes, one warmup and one timed repetition, and a reduced parameter grid.
It builds only a logging executable, not the indexes. It validates the logging
copy against the original search executable on both graph constructions with
ordinary, cap-one and cap-four search. If it fails, send the error and referenced
build/search log before attempting the full screen.

Then:

```bash
bash scripts/run_d007.sh screen
bash scripts/run_d007.sh package
```

Send **D-007-review.zip**. An optional four-query full-grid calibration is available:
`bash scripts/run_d007.sh pilot`. Its search component should scale roughly fourfold
to the 16-query screen, but executable startup, full-file verification, and loading
indexes are fixed costs and do not scale that way. Do not estimate this round by
multiplying the D-006 runtime: the methods and timing instrumentation have changed.

`bash scripts/run_d007.sh smoke` runs the local-only FANNS-small smoke if the
previous `local-runs/D-006-smoke` prepared workload is present. It exercises no
upstream executable. The complete user test suite remains necessary because the
public base did not include the user's local D-005 helper changes.

The wrapper defaults to `local-runs/D-006-fanns` and `external/DiskANN-diversity`.
Override `D006_PREPARED` / `D007_UPSTREAM` if needed. A direct module invocation
without `--upstream` runs only local controls, but cannot answer the upstream
baseline question. Do not reuse its output directory for a different plan.

## What changes in the comparison

### 1. Independent FAISS exploration and shortlist size

The full grid is `(efSearch, returned shortlist)`:

```
(256,256), (1024,256), (1024,1024), (4096,256), (4096,1024)
```

Each is run with post-filtering and the native bitmap selector. There is no wrapper
`max(ef,pool)` coupling. All tested pools are <= ef, avoiding a larger requested
output forcing a larger effective exploration floor. Reported FAISS distance
counts provide an additional check on actual work; ef itself is not a work count.

Mask packing, search, and float64 shortlist selection have separate recorded
timings. The integrated query timing includes their wrapper overhead too. This
lets us distinguish more traversal from more materialized candidates and more
reranking. Per-query HNSW counters are reset for each search; search is single-threaded.

### 2. SIMD exhaustive baseline

`exact_reference` is the existing scalar float64 correctness reference.
`exact_simd` performs the same full eligible scan and one-per-group selection,
using an explicit OpenMP SIMD reduction over double-precision squared differences.
Compilation uses `-O3 -fopenmp-simd -march=native`, **not fast-math** and not
multithreaded execution. The binary is compiled locally, never distributed.

The SIMD reduction may change summation order. Every selected result is checked
against the reference. Different IDs are recorded separately and only allowed
when sorted distances agree at rtol=1e-10/atol=1e-10; that is a numerical-equivalence
check, not proof of exact ties. Larger discrepancies abort. Constraint checks still
apply independently. The baseline is an improved compiled scan, not a claim to
the fastest possible exact implementation or an optimal BLAS/GPU implementation.

### 3. Actual FAISS hierarchical entry

The level-zero diagnostic now has three entry controls:

- The stored single entry, as in D-006.
- The same fixed four random items, as in D-006.
- FAISS 1.12.0's **actual `greedy_update_nearest`** descent from max_level down to
  level one, using its storage distance computer and native SIMD distance routines.

The descent's initial distance and reported distance calls are charged against
the total budget (1,024 or 4,096). The remaining budget goes to the unchanged
D-006 native level-zero kernel. Its reevaluation of the descended entry is also
charged. Repeated upper-layer evaluations count as work, not just unique IDs.
An upper descent that exhausts the requested budget causes an explicit error;
the runner never labels an over-budget query as budget-compliant.

Timing includes descent and its Python/native boundary calls. Upper-layer
distances use FAISS arithmetic, and level-zero distances use the diagnostic's
float64 arithmetic. This is a controlled composition, **not** a claim that the
entire diagnostic traversal is now the FAISS HNSW algorithm. No hierarchy is rebuilt.

### 4. Published upstream traversal, both graph constructions

The pinned Diverse DiskANN revision remains
`659041e1b12e3ab41c70ad12d11a29a883f17e15`. Both ordinary and diverse-built graphs
are searched using the upstream ordinary search and upstream diverse search with
per-seller caps one and four. This supplies the construction x traversal comparison
that our common diagnostic kernel alone could not provide.

Upstream `(L, returned shortlist)` settings are:

```
(256,256), (1024,256), (1024,1024), (4096,1024)
```

The upstream algorithm is unchanged. The runner compiles a copy of its CLI with
one logging insertion **outside the timed queries**, before the original CLI sorts
latencies. The insertion exports query-ordered search time and distance comparisons.
The exact insertion, source/binary hashes, compiler command and link command are
recorded in the plan. The copy must return byte-identical candidate files to the
original binary on the first grid setting in all six construction/search combinations.

Compilation reuses the existing Unix Makefiles `flags.make` and `link.txt`, including
the original library flags. It writes exclusively into `.local/d007-native/upstream`.
Other CMake generator layouts are rejected explicitly. No system packages or BLAS
alternatives are changed. The original build must contain `search_memory_index`.

Index staging uses symlinks to original static graph/data files and a freshly written
seller file from the frozen natural groups. Before searching, the runner validates
the graph checkpoint and compares the upstream stored vectors with the prepared
vectors. Unsupported padded/dynamic storage is rejected rather than guessed.

**Filter composition matters:** upstream diverse search selects candidates using
groups, after which this harness applies each predicate and selects final eligible
representatives. It does not implement native filtered diverse search. A group whose
closest unfiltered item is ineligible may lose its eligible alternative. Cap-four
helps test this limitation, but cannot guarantee its removal.

Each row also records how many exact target groups have an ineligible global-best
member. This offline diagnostic is never supplied to search. It explains a possible
composition failure; it is not an upper bound on what approximate search can return.

## Timing, scope, and decisions

- Four grouped conditions: groups alone, EM+groups, and the two date+group conditions.
  k=10/100; same first 16 development queries; two timed repetitions after warmup.
  The existing filter-only EM ground truth is replayed before evaluating each query.
- Local methods are interleaved in deterministic randomized order. Predicate masks
  are precomputed for every method; evaluating/indexing the predicate itself is excluded.
- Upstream search runs separately, with one warmup query batch followed by two timed
  batches. Its reported `ms` is **separately measured search plus downstream selection**.
  Process startup, index loading, output I/O, and transfer between stages are excluded.
  `timing_scope=separate_search_plus_selection` marks these rows. Do not present this
  sum as measured integrated end-to-end latency or as a decisive marginal speed win.
- Summaries report mean recall, minimum query recall, and the number of queries at
  >=95% recall. Sixteen development queries do not establish tail behavior or significance.
- Results are feasible only when enough eligible groups exist. Small-smoke infeasibility
  is reported explicitly. Representative recall and distance regret remain distinct
  from merely recovering group IDs.
- A complete manifest identifies code, inputs, graph hashes, settings, and upstream
  probe. Completed cells and candidate batches resume only with matching plan and
  content checksums. Changed code/settings require a new output directory. Run locks
  prevent concurrent use; after a hard kill, verify the process is gone before removing
  that run's RUNNING.lock. Never edit prepared D-006 artifacts to make a check pass.
- No original graph rebuild, held-out evaluation, ACORN/RACORN grouped adaptation,
  or promotion to a novel algorithm occurs in this round.

The next decision is whether any apparent diagnostic advantage survives independently
tuned FAISS, faster exhaustive scanning, real hierarchy descent, and published diverse
traversal. If the latter explains the benefit, keep it as a baseline result. If a
gap remains, examine per-query distance counts, representative failures, and the
filter/group composition diagnostic before designing a new search policy.
