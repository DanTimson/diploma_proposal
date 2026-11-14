# D: definition, comparison contract and stop/go rules

## Three different meanings of “HNSW for this context”

1. **D0 — approximate an existing retrieval operation.** Keep representations, similarity, eligibility and final scoring fixed. Replace exhaustive candidate retrieval by HNSW. Ask whether final quality remains acceptable while total query cost falls.
2. **D1 — make retrieval honor the application.** Keep the graph intact; vary candidate budgeting, result eligibility, and observable duplicate handling. Routing through an ineligible item remains permitted. A result exclusion must not silently sever the navigation graph. D1 includes an exact prefiltered fallback; restrictive filters can make exhaustive search cheap.
3. **D2 — change construction/traversal.** Only after D0/D1 isolate a persistent failure, retain a geometric routing backbone and test alternate neighbor diversification, reserved bridge edges, or a separate group-level graph. Match edge/storage budgets and construction cost. No new topology is implemented in D-001; the implementation uses FAISS HNSW.

The original socially meaningful hierarchy, where connector accounts inhabit upper layers, is a fourth interpretation. It changes which entities route influence, not just search. Random HNSW levels have a navigation role; connector centrality is not calibrated trust. That model needs a separate aggregation experiment against exact dependence calculations, kNN, random sparsification and propagation. D-001 neither establishes nor refutes its usefulness.

## Fixed objective

For query q, vectors x_i, eligibility E(q), declared group g_i and budget k, choose a maximum-cardinality set S (up to k) subject to i in E(q) and at most one i per group. Among those sets maximize sum s(q,x_i). With a single partition constraint this is solved exactly by retaining the best eligible item per group and taking the best k groups. Negative inner products do not reduce the required cardinality. This objective is not MMR or arbitrary graph dependence.

Groups are explicit copy roots in the synthetic duplication test. In SciFact and MovieLens each item has its own group. Thus the real-data transfers do not yet test natural source independence. No gold stance, attacker identity, test interaction or annotation provenance enters the retrieval algorithm. A missing gold item stays in the task-recall denominator.

## Implemented controls

- Exact prefiltered NumPy search and compiled FAISS Flat search, sharing the selection rule.
- HNSW with a fixed candidate pool, bounded pool doubling, and FAISS's native result selector. Native filtering is not ACORN or PathSeer.
- An identical-group collapse control when all group members have bitwise-identical vectors and identical query eligibility. Both exact and approximate engines receive this benefit; heterogeneous groups are explicitly refused.
- FAISS `M=16`, `efConstruction=100`, construction seed 11; query ef 32/128, initial pool 40, maximum pool 1280, k=10. Effective ef is max(requested ef, pool); record it per query. Pool is capped by the database size.
- Adaptive search accumulates candidates and stops when it finds k eligible groups or exhausts its cap. Finding k is **not** a certificate that they are the correct k. No hidden exact fallback rescues reported ANN quality.

## Workloads and what they establish

| Workload | Representation and task | Valid interpretation |
|---|---|---|
| Synthetic clean, copies, 1% eligibility, adverse filter | Fixed 64-dimensional normalized vectors; independently declared masks/copy groups | Controlled mechanism diagnostics, not a Community Notes attack study |
| SciFact-Open pooled corpus | Corpus-only TF-IDF/SVD; normalized vectors; claim queries | Effect of index approximation on a fixed weak retrieval model; pooled corpus is not full 500k retrieval |
| RecAD processed ML1M | Binary train likes >=4, truncated SVD; unnormalized inner product; all train-seen items excluded | Candidate retrieval transfer with held-out positive recall; not RecAD's victim/attack reproduction |
| arxiv-for-fanns small EM | Original 4096-dimensional vectors; L2; number_of_sub_categories exact-match filter | Adapter verification against supplied nearest-neighbor IDs; authors designate 1k set as debugging only |

The FANNS author repository supplies medium/large results and parameters; a specific medium ACORN result has been inspected and pinned. This meets the availability requirement, but directly comparable timing requires rerunning the medium protocol on the same hardware. There are no claimed published numbers for our new LSA/SVD transfer protocols. Preserve the earlier MultiVerS replay as a separate result.

## Measurement and anti-confounding rules

One CPU thread for FAISS and BLAS, one untimed warm-up, three measured repetitions; rotate method order. Time search plus bitmap creation, candidate rescoring and group selection. Do not time model fitting as query latency; record preparation/build separately. Report p50/p95 and per-query receipts. Serialized index size includes its stored vectors and is not peak resident memory. Python wrappers matter at sub-millisecond scales; final systems conclusions need more repetitions and a native/batched control.

Compare against the faster of the tested exact baselines, including collapsed exact when available. Do not call a faster method beneficial without its quality, fill, group violations and application metric alongside it. Count empty-reference queries separately; undefined recall is null, not a perfect score. Exact and FAISS scores can have floating-point ties; group recall and score gap should be inspected together.

The smoke fixes seed and first queries without tuning to their outcomes. Final work must separate tuning and held-out queries, repeat index seeds and use paired query/group bootstrap. Repeated timings are not independent relevance samples. Audit corpus/claim overlap before learning any retrieval scorer. Sparse-filter gains must include a cheap exact-subset option. Any learned edge policy requires held-out query workloads and drift tests.

## Proposed decision gates, not observed claims

**Continue from smoke:** one plausible useful regime, a verified reference, and an identifiable bottleneck. A single favorable timing is enough to motivate measurement, not to establish a thesis contribution.

**Scale pilot:** on at least two relevant regimes, including one real workload at a meaningful scale, seek either >=1.5x median speed at >=0.99 exact group recall with <=1 percentage-point absolute task-recall loss, or a predeclared build/memory benefit at matched quality. Require no eligibility/group violations. Examine p95 and smaller subgroups separately. These are project-management thresholds; choose application tolerances before final runs.

**Algorithmic thesis gate:** ordinary HNSW, exact-subset search, deduplication and existing filtered methods do not already dominate the proposed change. A special-purpose wrapper can be useful software without being sufficient algorithmic novelty.

**Stop/revert:** after the bounded scale pilot and one justified D1 treatment, no stable useful frontier remains across the chosen relevant workloads, or the only gains come from a weak exact baseline/changed representation. Then prioritize A/B. Failure on the tiny MovieLens/debug corpus alone is not a scale-independent rejection of HNSW. Failure of retrieval D does not disprove the separate social-hierarchy model.

## Next bounded iteration before November 19

Choose whether D's principal contribution should be retrieval systems or influence-graph sparsification. For the current retrieval interpretation: run the full 279 SciFact claims, three seeds and a fixed representation; execute an arxiv-for-fanns medium subset using original vectors/filters and the author protocol; add ACORN as an external baseline; measure index construction, peak memory and amortization. Only then choose one D2 mechanism. The medium database vectors alone are about 1.64 GB; do not accidentally launch the full multi-algorithm suite, which authors estimate in days. Keep A/B ready if this bounded investigation fails.
