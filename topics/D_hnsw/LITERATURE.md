# D literature and artifact audit — 14 November 2026

This is a focused prior-art check, not a proof of novelty. Implementation documentation, author repositories and source/data records were inspected. The latest preprints are context, not reproduced baselines.

| Source | Why it changes D | Artifact status here |
|---|---|---|
| [HNSW original paper](https://arxiv.org/abs/1603.09320), Malkov & Yashunin | Hierarchical navigable proximity graphs and diversified neighbor selection already exist; naming a new objective does not make the index new | Use an established implementation before rewriting construction |
| [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) | Same library supplies exhaustive and HNSW controls; avoid attributing Python-versus-native speed differences to the algorithm | FAISS CPU 1.12.0 installed and executed |
| [ACORN](https://arxiv.org/abs/2403.04871), SIGMOD 2024; [author code](https://github.com/guestrin-lab/ACORN) | Predicate-agnostic filtered HNSW is established. A new filter policy must beat this family, not only postfiltering | Repository located/inspected; algorithm not built or run |
| [FANNS benchmark](https://arxiv.org/abs/2507.21989), preprint 2025, revised April 2026; [code/results](https://github.com/spcl/fanns-benchmark) | Real transformer vectors, multiple filter semantics, scale/selectivity variation and released results provide a concrete systems comparison route | Original small vectors/EM labels/ground truth downloaded; exact-reference match checked. Medium ACORN result JSON inspected; timings not reproduced |
| [PathSeer](https://doi.org/10.1145/3802098), 2026; [maintainer implementation explanation](https://weaviate.io/blog/pathseer-filtered-vector-search) | Adaptive neighbor handling is already researched; fixed-versus-adaptive filter scheduling alone is not a novelty claim | Implementation explanation inspected; no paper-method reproduction. Weaviate's adaptation and original paper method must be distinguished |
| [REWEAVE](https://arxiv.org/abs/2611.16060), 11 November 2026 preprint | Learns edge utility and rewires within degree limits from query feedback. This constrains a future claim about workload-adaptive bridges | Primary abstract checked; full reproduction/guarantees not audited. Do not rely on its reported gains as our expected outcome |
| [VIBE](https://github.com/vector-index-bench/vibe), 2026 benchmark | Useful follow-on for standard ANN controls and modern embeddings | Author repository/results site inspected; not run |

## Application overlaps

[SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/) and [RecAD](https://arxiv.org/abs/2309.04884) supply real data already used in A/B/C preparation. D changes their representation/scoring protocol and therefore compares its index variants to its own exact reference, not to the published MultiVerS F1 or RecAD attack HR. [+VeriRel](https://arxiv.org/abs/2508.11122) remains relevant if we later learn the evidence scorer: changes in scoring and changes in index approximation must be ablated separately.

## Availability prerequisite

The direct ANN anchor is FANNS: vectors, filters, exact neighbors, baseline implementations and stored medium/large results exist. Pinned source revisions and artifact hashes are in `evidence/sources.lock.json`. Its author-designated 1k debugging set is not the published medium/large experiment. Before presenting a comparable speedup, rerun the chosen medium baseline and match k, filter semantics, vector metric, query IDs, threads, tuning and hardware. Do not divide our local milliseconds by another machine's reported throughput.

In the initial small set, exact-match filtering means `number_of_sub_categories == label`, not topic category membership. Query filter values are generated metadata in the benchmark, not trust labels. The source card describes GPT-4-generated search terms; this is not an organic production query log.

## What could remain a contribution

The defensible opening is the interaction of eligibility, repeated provenance and the final retrieval objective, supported by a matched cost/quality study and, if needed, one targeted construction/traversal change. A general HNSW application wrapper, ordinary oversampling, exact duplicate collapse, or adding a trust scalar to distance does not independently establish novelty. A negative applicability study may support a diploma only if a supervisor accepts that scope; the current plan is to fall back to A/B if D lacks a useful regime.
