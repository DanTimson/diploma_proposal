# Round 3 adjudication and next decision — 16 November 2026

## Decision

Change the immediate next experiment from larger blind over-fetch budgets to
**same-graph filtered-search calibration**, and advance a small application
pilot alongside it. The patch includes a pinned RACORN smoke-tested runner,
a measured FANNS metadata audit, an opt-in sparse exact kernel with paired
checks, and an executed saved-prediction B copy pilot. Larger user-machine
runs are prepared, not reported as completed.

Do not announce a new HNSW algorithm, a useful planner, or source-independence
estimation from any of this. A defensible supervisor discussion currently has
two options: B's evidence-selection question with a retrieval component if
needed; or D as a carefully scoped constrained-search measurement study.
A remains the closest match to the original reputation problem and is not
refuted by ANN experiments. None is yet evidence of a large commercial project.

## How the reviews change our judgment

| Review claim | Assessment and consequence |
|---|---|
| Small, low-dimensional eligible subsets favor exact scans | Correct scope criticism. Our 100k result never established an algorithm-wide ANN loss. Add larger N and structured/correlated filters. |
| Exact cost is N*s*d while graph cost is roughly constant; derive a high-d crossover | Too strong. Graph distance work also depends on d, geometry, selectivity, recall target and construction. Exact also pays mask enumeration, aggregation and top-k costs. The 58k/2k crossover values are estimates from an inadequate model, not thresholds to implement. |
| Replace N-sized exact scratch with a hash map | The overhead is real; the proposed replacement is not uniformly faster. Retain the old kernel and measure the new one separately. |
| Independent masks and iid groups cover too little | Correct. They isolate mechanisms but do not establish natural-workload robustness. The claim that this is always ACORN's easiest case needs qualification by graph construction and data geometry. |
| Cap-before-filter loss is operator ordering | Correct for our custom shortlist control. Official diverse-search queues are a different mechanism. Do not use the former as evidence of the latter's limitations. |
| Resumption proves a contribution | Neither our original control nor the review supports that. Traversal work at the same final budget matches a one-shot traversal; checkpoint selection adds overhead. Savings against repeated restarts are engineering evidence, not novelty. Add one-shot budgeting if developing this further. |
| A selectivity router is a defensible new method | Existing systems and SIEVE already make related planning decisions. Any new claim needs an identified additional signal, competing planners, and held-out cost/recall validation. |
| FANNS author/category metadata solves natural grouping | Metadata is real, but semantics must be specified. Category combinations form a partition; shared coauthors do not. Counts measured below. |
| Require every named baseline immediately | Scope-dependent. RACORN is directly relevant to a predicate-agnostic traversal claim. SIEVE is necessary prior art for workload-aware indexing/planning; matching its workload knowledge and memory budget is essential before comparison. Adding every system before defining the task is not a useful gate. |
| Add a second attribute to make the problem hard | Not a research justification by itself. The exact formulation matters; some combinations of partition constraints remain tractable. Do not add arbitrary fairness constraints to manufacture difficulty. |

The first review read D-004 in detail but predated our uploaded expanded
DiskANN results. The second includes valuable literature pointers but describes
an older repository state in several places: ACORN and diverse-search artifacts
have already been run, and D-003/D-004 do not use per-item synthetic groups.
Its per-item criticism still applies to the earlier real-data transfers.

The newest upstream run remains part of the evidence: at group-only L=200,
cap16, diverse construction improves item recall 95.97% -> 99.94%; at 10%
eligibility, cap64/L3200 improves 99.84% -> 100%. At 1%, cap4/L3200 both give
about 30.8% recall/fill. Expected passing candidates from an independent 1%
mask are about 32 out of 3200 before group collapse. This is candidate
starvation, not a general negative result about diverse construction.

## Literature/artifact verification

**RACORN: verified primary paper and artifact.** arXiv 2607.00768v2 was revised
6 November 2026. Section II-B distinguishes the original ACORN construction
from ACORN-style search on existing pruned HNSW graphs. The RACORN artifact
compares its ACORN-1 and RACORN-1 on the latter. This is a consequential
explanation for why our original ACORN artifact can perform well while a
different paper reports ACORN-1 collapse. The paper's reported speedups are
not transferred to our workloads. We pin its actual current artifact and
use its unchanged driver for the next filter-only control.

**HCBGen: primary abstract and venue statement verified; full implementation
not audited here.** The work argues that selectivity/correlation alone are
not reliable difficulty proxies and proposes execution-conditioned hardness.
We adopt the caution, not its numerical claims or the assertion that our two
alpha values reproduce HCBGen.

**SIEVE: paper and official artifact verified.** Its workload-trained index
collection under a memory/recall budget makes a simple scan/index router an
insufficient novelty claim. It is not an interchangeable fixed-graph baseline.
No SIEVE results were run or inferred here.

**FANNS: official dataset schema and actual medium metadata checked.** The
paper first appeared in July 2025 and was revised in April 2026; describing
it simply as a new 2026 dataset misses that history. Its published ground
truth is for ordinary filtered retrieval, not our new grouped objective.

**hnswlib: relevant C++ API exists, but the review overstates drop-in support.**
The inspected `stop_condition.h` contains `MultiVectorSearchStopCondition`
and document-ID-aware spaces. This is not evidence of a standard Python
`grouped-search` switch that already enforces our exact output semantics.
A native adapter still needs to handle result representatives and eligibility.

**Qdrant: primary docs confirm existing planning.** Its documented paths
include exact scoring for small matching sets and graph/ACORN choices for
larger ones. This is sufficient to reject novelty for a bare selectivity
threshold; the production thresholds are not our experimental parameters.

Other review-only leads (FAVOR, JAG, Curator, the NeurIPS color-bucketing
poster, and several additional arXiv IDs) remain unverified here and are not
promoted into the bibliography as established support. No claim of a complete
literature search is made.

## Measured FANNS grouping feasibility

Pinned medium `database_attributes.jsonl`, SHA-256
`6d09f45c631821b0bc979152af373d6f02bf32d3a4c5b197fb947f5d8da8b4e9`:

| Definition | Groups among 100k papers | Papers in non-singleton groups | Largest group |
|---|---:|---:|---:|
| Exact set of subcategories | 8,397 | 94.767% | 3,171 |
| Exact set of author strings | 96,024 | 6.973% | 53 |

81,289 papers have multiple authors. There are 261,182 distinct author strings,
of which 74,170 recur. Choosing the first author would force a partition by
changing the meaning; grouping by entire author set misses partial overlap.
Author strings also do not constitute a disambiguated identity registry.

The existing filter `number_of_sub_categories=1` selects 52,817 papers in 149
category-set groups; value 2 selects 29,941 papers in 2,053 groups. These are
real, strongly structured group/filter combinations with sufficient groups
for k=100. Values 7–10 have fewer than 100 eligible groups, so their reference
cardinality must shrink rather than count unavoidable underfill as failure.

This makes medium FANNS with category-set grouping a viable **systems** test,
not a claim about independent sources or recommendation utility. It has the
ordinary FANNS published comparator as an anchor, but grouped results would
be a new protocol requiring exact-reference validation. Large FANNS vectors
alone are about 44.8 GB, exceeding the user's reported RAM: start with medium,
not a casual full-corpus in-memory run.

## Exact-baseline check

The original kernel allocates/scans N `Hit` records when grouped. A new sparse
kernel uses a map over encountered eligible groups and preserves float64 L2,
stable ID ties, and the objective. All-mask tiny tests and 48 paired synthetic
query/cell comparisons return identical IDs. The old default is retained.

Sequential synthetic check: N=100k, d=128, eight queries, warmup plus three
repetitions. At 1% eligibility median latency improves from approximately
0.289 to 0.240 ms for unique groups and 0.310 to 0.263 ms for skewed groups.
At 50% with unique groups it worsens from 9.583 to 13.094 ms. These are local
mechanism timings, not an ANN crossover or a guarantee of a universal kernel
improvement. The separate JSON receipt retains every measured query.

The defensible cost decomposition is predicate/ID preparation + distances +
group aggregation + top-k selection, with fixed and variable costs measured.
The current wrapper's `flatnonzero(mask)` is still O(N); a provided posting
list and a full predicate scan must not be charged as the same operation.

## Executed B-001 pilot: application connection and its limits

MultiVerS-20, 279 claims, 50 released candidates per claim. Baseline saved
prediction replay again yields 187 true positives / 254 predicted positives /
460 gold pairs, F1=0.5238095, matching the earlier replay. No training or
full-corpus retrieval occurred.

We now distinguish two ceilings:

- Gold evidence documents present among candidates: **438/460 = 95.22%**.
- Correctly signed pairs with the frozen predicted labels: **187/460 = 40.65%**.

At k=10, confidence ranking retrieves 331 gold documents (71.96%) but only
185 correctly signed pairs (40.22%). All 15 both-stance claims have both types
of gold documents somewhere among the candidates; frozen labels correctly
identify both sides for only two of those claims. The top-10 list contains
both kinds of gold documents for nine claims and correctly signed evidence
for both sides for two.

| Copies of top-confidence candidate | Score-only gold documents | Text-dedup gold documents | Score-only correctly signed pairs | Text-dedup correctly signed pairs |
|---:|---:|---:|---:|---:|
| 0 | 331 | 331 | 185 | 185 |
| 1 | 328 | 331 | 184 | 185 |
| 5 | 287 | 331 | 179 | 185 |
| 20 | 179 | 331 | 144 | 185 |

These copy origins are known by construction, while the selector uses only
observable text hashes. The result supports a copy-crowding mechanism and a
simple dedup baseline, not a new robust method. In particular, the signed
selection problem is almost saturated (185 of 187 available hits at k=10).
An ANN speedup cannot fix these stance errors. Evidence-document selection
has more headroom; its evaluation must keep contradiction/document recall
separate from the verifier's predicted stance and report both.

Next B work, if selected: MMR and predicted-stance quota on the same candidates,
then near-duplicate perturbations and an independently grounded second corpus.
Do not add HNSW to a 50-candidate selection stage simply to merge D into B.
HNSW becomes relevant only if corpus-scale candidate generation is a measured
bottleneck or changes evidence availability.

## Bound the remaining work before the selection window

Run the prepared 100k/1M **filter-only** RACORN control sequentially on the
user's machine. It compares existing algorithms on one shared graph, at 1%,
10%, 50% eligibility and random/adverse masks, while leaving the D-004 test
queries untouched. It is diagnostic, not a new grouped-search benchmark.

Then choose one topic rather than expanding indefinitely:

1. B has an application question and a reproduced published anchor. Proceed
   only if meaningful document-level gains survive MMR/quota/dedup baselines;
   do not sell frozen-label signed recall as the available opportunity.
2. D is a systems-study option if the supervisor accepts that scope. The next
   natural grouped test is medium FANNS/category-set grouping, with the usual
   filtered protocol reproduced separately. A new-method claim requires a
   residual gap beyond existing techniques, not merely an exact/ANN crossover.
3. A remains available for the original reputation motivation. Its aggregation
   experiment must stand on its own; do not infer its prospects from D results.

Larger blind over-fetch remains a useful later control, but it is no longer
the highest-information immediate run. Do not consume held-out queries while
still selecting objectives, datasets, implementations or hyperparameters.

## Primary sources checked

- RACORN v2: https://arxiv.org/html/2607.00768v2 ; artifact https://github.com/naver/racorn
- HCBGen abstract: https://doi.org/10.48550/arXiv.2606.14193
- SIEVE: https://www.vldb.org/pvldb/vol18/p4723-li.pdf ; https://github.com/BillyZhaohengLi/SIEVE-vldb25
- FANNS v3: https://arxiv.org/abs/2507.21989 ; https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium
- hnswlib C++ stop conditions: https://github.com/nmslib/hnswlib/blob/master/hnswlib/stop_condition.h
- Qdrant planning: https://qdrant.tech/documentation/search/search/
