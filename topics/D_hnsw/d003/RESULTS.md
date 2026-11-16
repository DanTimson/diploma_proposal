# D-003 measured results — 2026-11-15

## Outcome

The upstream diverse-search implementation is runnable. The four-way experiment
finds an eligibility/group interaction on real SIFT vectors with synthetic metadata.
A simple relaxation of the existing method's candidate-group cap repairs much of
the 10% case; this is evidence to strengthen baselines, not evidence for a new graph.
At 1%, exact subset search remains the practical winner in the measured screen.

## Completed work

- Pinned Microsoft DiskANN diversity branch compiled without source changes. Added
  cstdint via a compiler flag for modern GCC. Private build dependencies were used
  on the execution host; user Ubuntu prerequisite commands are separate.
- Upstream smoke: two graph constructions, ten search configurations, zero final
  constraint violations. Upstream SIFT adaptation: two constructions and fifteen
  search configurations, zero final violations.
- Main factorial: 100k randomly sampled SIFT vectors, 64 randomly sampled queries,
  k=100, four kinds of conditions, three filter fractions, six retrieval settings.
- SIFT source files pinned and verified; first eight original queries agree 100%
  with released top100 ground truth under exact FAISS over the full million vectors.
- No result claims numerical reproduction of the original paper's figures.

## Primary HNSW screen

The table uses the 32 queries labelled confirmation. All configurations are reported
as an exploratory screen; we have now inspected this partition. A future learned or
selected policy requires new untouched queries. One index seed only; no confidence
claim or hardware-independent speed guarantee. Item recall is agreement with exact
constrained representatives, and can penalize distance ties. All rows had zero
eligibility/group/invalid-ID/duplicate-ID violations.

| Condition | Method | Item recall | Fill | p50 ms |
|---|---|---:|---:|---:|
| neither | exact_cpp | 100.00% | 100.00% | 11.021 |
| neither | exact_faiss | 100.00% | 100.00% | 2.444 |
| neither | hnsw_native ef512 | 100.00% | 100.00% | 1.241 |
| group | exact_cpp | 100.00% | 100.00% | 11.155 |
| group | exact_faiss | 100.00% | 100.00% | 7.995 |
| group | hnsw_native ef512 | 99.97% | 100.00% | 3.886 |
| both-0.5 | exact_cpp | 100.00% | 100.00% | 8.238 |
| both-0.5 | exact_faiss | 100.00% | 100.00% | 14.450 |
| both-0.5 | hnsw_native ef512 | 99.88% | 100.00% | 4.232 |
| both-0.1 | exact_cpp | 100.00% | 100.00% | 3.005 |
| both-0.1 | exact_faiss | 100.00% | 100.00% | 5.418 |
| both-0.1 | hnsw_native ef512 | 87.31% | 100.00% | 3.962 |
| both-0.01 | exact_cpp | 100.00% | 100.00% | 0.520 |
| both-0.01 | exact_faiss | 100.00% | 100.00% | 2.195 |
| both-0.01 | hnsw_native ef512 | 44.47% | 44.53% | 19.623 |

These p50 values include mask/ID construction and final selection in the measured
method. The C++ scan uses float64 accumulation and is not universally fastest:
FAISS wins the unrestricted case. Selection uses the faster exact control where
making a practical comparison. No GPU or batched-throughput claim is implied.
The final screen was run after the upstream builds completed; initial preliminary
screen timings overlapped compilation and are deliberately excluded from this table.

## Existing diverse search: effect of a relaxed candidate cap

Same exported vectors, queries, groups, and masks. Upstream standard graph, diverse
search, L=3200. Candidate cap is internal, while final one-per-group remains fixed.
The filter is applied after upstream retrieval; no native joint traversal is claimed.

| Condition | Candidate cap | Item recall | Fill |
|---|---:|---:|---:|
| group | 1 | 99.38% | 100.00% |
| group | 4 | 99.72% | 100.00% |
| both-0.5 | 1 | 93.94% | 100.00% |
| both-0.5 | 4 | 99.69% | 100.00% |
| both-0.1 | 1 | 61.97% | 93.78% |
| both-0.1 | 4 | 98.47% | 100.00% |
| both-0.01 | 1 | 9.06% | 9.06% |
| both-0.01 | 4 | 31.94% | 31.94% |

This is a concrete reason to avoid rushing into custom graph construction: retaining
multiple representatives per group substantially repairs the 10% composition failure.
It does not establish a 99% guarantee, and it still fails badly at 1%. Raising the
cap further, resumable traversal, ACORN, and exact fallback remain untested controls.

The diversity-enabled construction used upstream NumDiverse=1, its weak default.
Its results are similar to standard construction here. This is NOT a negative
result about a properly tuned diversity-aware graph. The delivered adapter exposes
--num-diverse for subsequent 8/16/32 comparisons; those are not run in this milestone.
The delivered adapter differs from the measured version only by exposing this knob
(default unchanged); the measured source snapshot and its verified SHA are included.

Upstream CLI mean search latency excludes downstream mask/group selection. It is
retained in raw logs but is not compared to end-to-end p50 as though interchangeable.

## Mechanism and limits

The best representative of a group before filtering need not be its best eligible
representative. Discarding within-group alternatives can therefore harm a later
filter. Conversely, ordinary candidate retrieval can exhaust its budget on the
dominant group. This expected noncommutativity is a mechanism to measure, not a
novel theorem. Real-vector geometry is present; the groups and filters are synthetic.

Group recall alone can conceal a wrong group member. D-003 adds item recall and
squared-L2 regret to prevent that. Geometry preservation is still not evidence
quality, factual correctness, recommendation utility, or reputation robustness.

## Reproduction and dataset gate

SIFT is a shared corpus with prior diverse-search results, but this is a subsampled,
newly labelled adaptation. Natural-group Automotive files are presently inaccessible
through their original release URLs. SQID is another route, requiring metadata and
judgment joins. The first serious application-level milestone remains uncompleted.

## Additional control and validation

A separate clustered 10k-vector, 32-query screen uses an eight-dimensional latent
space embedded in 64D. All tested HNSW settings reach 100% unconstrained item recall
on its 16-query confirmation partition. This removes the earlier uniform-random
clean-case recall ceiling for this control. It is a synthetic diagnostic, not an
additional natural corpus. All final selections across both screens and upstream
adaptations have zero recorded constraint violations.

The test suite contains 40 passing tests, including exhaustive small-set comparison
of native exact search, tie/empty handling, unsafe shape rejection, nested masks,
and detection of wrong representatives despite perfect group recall.

## Runtime and handoff

Final 100k factorial screen: 138.7 seconds on the execution host, excluding initial SIFT download.
Upstream build/search commands: 167.0 seconds total, excluding independent Python evaluation and initial compilation.
Do not use these as exact predictions for the user's machine. First setup requires
~525 MB SIFT data plus platform-specific compiler dependencies. No large rerun is
required to inspect the delivered results. See README.md for commands and DECISION.md
for the proposed diploma question and stop rules.
