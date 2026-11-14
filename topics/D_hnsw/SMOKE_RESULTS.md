# D-001 smoke results — 14 November 2026

**Decision: continue one bounded D pilot; do not yet commit to a custom HNSW design.** Ordinary HNSW has a useful-looking retrieval regime in this smoke, while sparse eligibility, duplicate saturation and a small recommender show clear limitations. These are local observations, not superiority over published systems.

The index is FAISS CPU 1.12.0, M=16, efConstruction=100, seed 11; k=10; one warm-up and three measured repetitions; one CPU thread on an AMD EPYC 9V74 host. These are not timings from your desktop. Query latency includes filtering/selection wrappers but excludes representation fitting and index building. All rows use the final code, with hashes in the receipts.

## Representative comparisons

The table deliberately includes unsuccessful regimes. Recall is group top-10 agreement with the exact constrained score, not evidence correctness or user satisfaction. Exact latency is the faster tested exact baseline for that workload. Settings are illustrative smoke configurations, not a tuned held-out frontier.

| Case | Database / queries | HNSW variant | Exact p50 ms | HNSW p50 ms | Exact-group recall | Speed ratio |
|---|---:|---|---:|---:|---:|---:|
| clean | 6,000 / 48 | hnsw_adaptive_ef128 | 0.129 | 0.148 | 99.37% | 0.87x |
| copies | 7,920 / 48 | hnsw_adaptive_ef128 | 0.142 | 0.465 | 49.17% | 0.30x |
| copies | 7,920 / 48 | hnsw_collapsed_ef128 | 0.142 | 0.218 | 99.37% | 0.65x |
| sparse_filter | 6,000 / 48 | hnsw_adaptive_ef128 | 0.052 | 2.866 | 96.88% | 0.02x |
| adverse_filter | 6,000 / 48 | hnsw_filtered_ef128 | 0.096 | 0.194 | 11.67% | 0.50x |
| scifact | 12,236 / 96 | hnsw_fixed_ef32 | 0.216 | 0.111 | 100.00% | 1.94x |
| recad | 3,659 / 96 | hnsw_adaptive_ef128 | 0.107 | 0.147 | 100.00% | 0.73x |
| fanns | 1,000 / 64 | hnsw_adaptive_ef128 | 0.928 | 0.455 | 100.00% | 2.04x |

A ratio above 1 means faster; below 1 means slower. p95, fill, objective gaps, all configurations and task metrics are in `results/ann_smoke_summary.json`. Full query receipts are supplied separately to avoid bloating the repository patch. The two copy rows share the same best exact baseline, which already collapses identical groups.

## Interpretation by application

**SciFact transfer:** on 12,236 pooled documents and 96 claims, the shown HNSW setting preserved every exact top-10 and reduced median lookup from about 0.216 to 0.111 ms (1.94x). The representation is corpus-only TF-IDF/SVD, not a modern pretrained retriever. Macro evidence Recall@10 is 46.33% for both exact and HNSW, excluding claims without annotated evidence from that mean. Matching exact retrieval does not imply 100% evidence recall. This does not reproduce MultiVerS F1, and it is not full 500k-document retrieval. Index build was around 0.7 seconds: thousands of queries would be needed to amortize even this small absolute per-query saving, before update costs.

**Recommendation transfer:** 3,659 train-catalogue items are too few to establish a scaling advantage. Adaptive HNSW reaches exact top-10 agreement at ef=128 but is slower. A fixed candidate pool loses many eligible recommendations because seen items occupy the pool. Some approximate variants have slightly higher held-out positive recall than exact scoring; this is a small-sample scoring perturbation, not demonstrated recommendation improvement.

**FANNS:** exact reference recovered the supplied top-10 for all 64 checked queries. Adaptive HNSW at ef=128 matched them and was about twice as fast in this run. The original vectors have 4096 dimensions, explaining why even 1k objects can involve meaningful distance work. Nevertheless the authors explicitly designate this size for debugging. It does not substitute for their medium/large experiments or an ACORN comparison.

**Copies:** ordinary HNSW loses many groups under exact-duplicate saturation. Pool doubling alone does not restore them. Collapsing identical groups restores roughly 99.4% ANN recall, but collapsed exact search is faster at this scale. Therefore these results support deduplication as a control, not a claim that a new HNSW topology is needed.

**Filters:** exact search over a small eligible subset wins decisively in the 1% case. In the deliberately adverse case, permitted items are the farthest 10% under the query geometry; global candidate search is the wrong access path. These examples justify a planner/fallback baseline and stronger filtered-ANN competitors before graph design.

## Verification and limitations

- 25 tests passed: original 11 and 14 ANN tests, including exhaustive checks of the group-constrained reference, native filtering, negative/unnormalized IP, gold isolation, copy-collapse preconditions and binary readers.
- Zero eligibility or group-cap violations in all seven final workloads. Reference and ANN quality are reported separately.
- Only one index seed, small query subsets, synthetic masks and weak transfer representations. No statistical significance claim. Same-behavior wrapper variants have timing differences, demonstrating sub-millisecond measurement noise.
- No ACORN/PathSeer/REWEAVE execution, custom graph construction, full-corpus retrieval, peak-RSS measurement or reputation propagation has been performed.
- The pinned FANNS medium ACORN artifact contains results and parameters, but our small-set run must not be numerically compared against its throughput.

## What to do next

Keep D first. Decide whether the intended graph is a retrieval index (current executable interpretation) or an influence/aggregation graph (different objective). For retrieval, run a bounded medium-scale pilot with held-out queries, multiple seeds, a stronger fixed representation and an ACORN baseline. Choose one topology change only if that comparison isolates a failure that simple preprocessing or exact-subset routing does not solve. If no useful frontier survives, return to A/B without expanding D indefinitely.
