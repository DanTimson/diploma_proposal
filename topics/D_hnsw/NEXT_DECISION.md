# After the D-002 screen: which HNSW hypothesis?

The current runner tests **approximate constrained retrieval**. Its answers must not be generalized to **reputation propagation through an HNSW-like graph**. Those are different algorithms and different dependent variables.

If ordinary HNSW offers a useful regime, that establishes engineering feasibility. It does not yet establish a diploma contribution: the usefulness of ANN itself is already known. The next step is to identify a concrete constraint interaction that strong baselines do not handle, rather than accumulate more positive ANN timing plots.

## Retrieval interpretation

Candidate question: can a navigation graph preserve high-value, nonredundant candidates under query-specific eligibility at lower cost than exact-subset search, deduplication and established filtered search?

Required next comparison: ACORN under the same vectors, masks, queries, k and hardware. PathSeer is relevant if its artifact can be reproduced. A group-aware neighbor rule should retain a geometric backbone and be compared at the same edge/memory budget; its group features must be observable without gold stance/attacker identity. Standard HNSW already diversifies geometric neighbors, so “diversify neighbors” alone is not new.

The exact-copy saturation case is insufficient novelty evidence. [FAISS's own FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ#searching-duplicate-vectors-is-slow) documents difficulties with identical/near-identical vectors and recommends deduplication. To motivate a new method, include nonidentical but provenance-linked documents where one-representative collapse is not exact, and ensure that a larger ordinary candidate pool or a secondary group index does not solve the problem just as well.

## Influence/aggregation interpretation

Candidate question: does HNSW-like hierarchical sparsification preserve or improve useful reputation aggregation compared with exact scoring, kNN sparsification, random edges and ordinary trust propagation at a matched edge/compute budget?

Here, nodes would be accounts or evidence sources, edges would retain a clearly defined influence/similarity meaning, and upper-layer membership would be an experimental variable. It must not silently encode trusted status. Start with exact aggregation on the same historical features; then vary only the retained graph and propagation operator. Report the aggregation error and manipulation/minority-retention trade-off, not just nearest-neighbor recall.

An elementary check is important: if the exact aggregation rule itself cannot distinguish an honest specialist group from a coordinated coalition under the available observations, changing its routing topology does not create the missing information. A graph may still improve cost, coverage or stability under stated assumptions; those are separate claims.

This branch reconnects D to topic A. The current benchmark code supplies reusable infrastructure, but does not test that branch. A failed retrieval regime cannot be used as evidence that influence-graph sparsification fails, and a successful ANN speedup cannot be used as evidence that it produces better reputations.

## Scope decision

Choose one of these interpretations before introducing a custom constructor. Keep A/B as alternatives if neither yields a defensible and reproducible contribution within the diploma window. The immediate deliverable remains a specific research question, a runnable baseline and an honest limitation—not a claim of a new universal reputation architecture.
