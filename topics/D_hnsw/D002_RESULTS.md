# D-002: completed research batch — 14 November 2026

**Verdict:** D has a demonstrated applicability regime and should not be rejected as uniformly unhelpful. The evidence establishes a useful retrieval trade-off, not a new HNSW algorithm or improved reputation propagation. The next step is a specific D2 hypothesis and a strong filtered-search comparison, rather than another generic scaling sweep.

All **21 cases** completed, with **three index seeds** for each of seven workloads, in approximately **24 minutes of compute**. Source hashes and all result receipts were checked. Every workload fingerprint remained identical across its index seeds. The source/representation seed was fixed at 11. There were **zero eligibility or group-cap violations**. The patch has **34 passing tests**.

## Comparable observed points

Recall is mean top-10 group agreement with the exact constrained objective, averaged over the three index seeds. Speed ratios compare median query latency against the faster tested exact baseline in that workload. Ratios below one mean slower. These are per-query, single-thread measurements; not batched throughput. The representative setting is fixed across seeds, rather than selected separately for each seed.

| Workload | N / dimensions / queries | Method | Exact-group recall | Speed ratio | Median latency, ms |
|---|---|---|---:|---:|---:|
| fanns-medium-q1000 | 100,000 / 4096 / 64 | hnsw_adaptive_ef128 | 99.22% | 40.91x | 1.277 |
| fanns-medium-q1000 | 100,000 / 4096 / 64 | hnsw_filtered_ef256 | 99.48% | 23.82x | 2.181 |
| scifact-all | 12,236 / 96 / 279 | hnsw_adaptive_ef128 | 99.95% | 1.75x | 0.115 |
| recad-256 | 3,659 / 64 / 256 | hnsw_adaptive_ef128 | 100.00% | 0.51x | 0.212 |
| synthetic-clean-100000 | 100,000 / 64 / 128 | hnsw_adaptive_ef256 | 89.45% | 1.61x | 0.678 |
| synthetic-copies-100000 | 105,120 / 64 / 128 | hnsw_collapsed_ef256 | 89.45% | 1.09x | 1.577 |
| synthetic-sparse_filter-100000 | 100,000 / 64 / 128 | hnsw_adaptive_ef128 | 95.34% | 0.03x | 6.190 |
| synthetic-adverse_filter-100000 | 100,000 / 64 / 128 | hnsw_filtered_ef256 | 0.47% | 1.68x | 0.762 |

## The useful regime: medium FANNS

The actual author-provided 100,000-vector, 4096-dimensional dataset was downloaded and checksum-verified. We used the unchanged exact-match predicate and query indices 1000–1063, a 64-query pilot slice distinct from the earlier debugging queries. Each of the three runs independently matched all supplied exact top-10 reference sets before benchmarking.

Adaptive HNSW at ef=128 gave **99.22% mean exact-group recall** with about **41x lower median lookup latency**: approximately **1.28 ms versus 52.17 ms** for FAISS Flat. At ef=256, native filtered HNSW gave **99.48% mean recall** at about **24x speed**, with median latency around **2.18 ms** and a lower tail than the repeated adaptive-search variant.

The first number must not be turned into a guarantee of >=99% recall. Adaptive ef=128 varied from 98.75% to 99.53% across seeds; the query-bootstrap interval for the seed-averaged mean was **98.54%–99.74%**. For the filtered ef=256 point it was **98.96%–99.84%**. These intervals resample the 64 queries, conditional on these fixed vectors and observed seeds; they do not account for all possible workloads or model changes.

Construction took a median **109 seconds**. Based on the observed median per-query savings and additional construction time relative to Flat, break-even was roughly **2,100–2,200 queries**. This is an amortization estimate for a static index, excluding embedding generation, updates and service overhead. The full comparison process peaked near **6.24 GiB RSS**; it holds multiple controls and preparation buffers. The serialized HNSW index was approximately **1.54 GiB**, so process peak is not deployable index memory.

These timings do not beat a published ACORN result: ACORN was not run, the query subset/hardware differ from the paper, and a batched exact baseline has not been measured. The direct result is superiority over the two exact controls in this online-style lookup experiment at a small recall cost.

## Application transfers

**SciFact:** all 279 claims, 12,236 pooled documents, corpus-only 96-dimensional LSA. Adaptive ef=128 retained **99.95% exact-neighbor agreement** at approximately **1.75x** median speed. Macro annotated evidence Recall@10 stayed **43.28%**, measured on the 206 claims with annotated evidence. Contradicting-evidence recall also remained unchanged. The index accelerated the same weak representation; it did not improve evidence reasoning or reproduce MultiVerS F1. A 500k-corpus experiment remains separate.

**Recommendation:** on 3,659 training-catalogue items and the same 256 users, adaptive ef=128 matched every exact top-10 but took approximately **0.212 ms versus 0.110 ms**. Macro held-out positive recall remained **5.19%**. This small recommender does not need HNSW at the tested operating point. It is not a RecAD poisoning/victim-model reproduction.

## Negative and boundary cases

**Uniform random vectors:** at 100k items, ef=256 reached only 89.45% recall for roughly 1.6x speed. This intentionally difficult synthetic geometry does not meet the predeclared near-exact quality target. It also shows why ambient vector dimension alone does not predict HNSW performance.

**Exact copies:** ordinary adaptive HNSW at ef=256 retained about 70.36% of exact groups with only 83.6% average fill. Collapsing identical observable groups recovered the clean-case 89.45% recall and full fill, but had little speed advantage over collapsed exact search. The data support the deduplication control; they do not establish novel source-independence inference.

**One-percent eligibility:** exact prefiltered search won. Adaptive HNSW still lost roughly 4.7% of reference neighbors and was about thirty times slower than the faster exact control.

**Adverse filter:** permitted items were deliberately in the farthest 10% under the query geometry. Bounded global-candidate search returned no eligible results. Native filtered HNSW filled the list but recovered only about **0.47%** of the true top-10. Thus “found k eligible items” is not a quality certificate. This is a concrete reason to include exact-subset routing and filter-aware graph traversal, not a reason to treat every query with one global geometric search policy.

## What this changes for topic selection

Keep D first for one targeted research step. The applicability question has a positive answer in a substantial real-vector regime and a smaller evidence-transfer regime. No custom construction algorithm is justified merely by those gains. The strongest next retrieval hypothesis concerns nonidentical provenance-linked candidates and query-specific eligibility, compared with exact-subset search, larger candidate pools, a simple group index and ACORN.

If the intended main idea is instead an HNSW-like **influence graph**, the next experiment must explicitly evaluate aggregation and minority retention under sparsification. The present results neither establish nor refute that idea. `NEXT_DECISION.md` separates the two interpretations and the necessary controls. A/B remain viable alternatives if neither interpretation yields a defensible contribution.

## Reproducibility and deliverables

The patch is based on `03aff5696401b86ff61c754956b862adf0aac1e3`. Existing A/B/C and D-001 results are preserved. It adds a resumable runner, resource limits, atomic receipts, independent data/index seeds, lower-memory exact scoring, shuffled measurement order, analysis code, source pins and this report.

The shipped default plan has 33 benchmark cases plus eight preparation jobs. **The completed local pilot has 21 cases**, omitting the 10k synthetic tier and using 64 rather than 128 medium-FANNS queries. The precise pilot plan is included separately. No missing cases are represented as completed.

Use `scripts/analyze_sweep.py` to reproduce the aggregate and query-bootstrap diagnostics. The full receipt archive contains the pilot plan, manifest, state, logs and per-query JSON; raw third-party datasets are excluded. The scope is a screening study with limited queries, three index seeds, one host and no strong filtered-search competitor yet.
