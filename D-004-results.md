# D-004 development results — 2026-11-16

## Decision

**These results do not justify presenting a new HNSW-like graph algorithm as
the diploma's established contribution.** Existing filtered search resolves
most of the low-recall case, and compiled exact prefiltering is faster on the
tested 100k workload. Preserving a frontier saves repeated work, but is an
existing technique. No commercial-scale opportunity or reputation benefit
has been demonstrated.

A narrower systems study remains plausible: **Constrained vector retrieval
under simultaneous eligibility and group uniqueness: reproducible comparison
and query-plan selection.** Its scientific content would be the conditions
under which exact prefiltering, filtered ANN, and diverse search trade off,
plus an independently evaluated selection policy if a nontrivial crossover
exists. A policy that merely chooses exact search everywhere is not evidence
of a useful new method. Whether a comparative/engineering contribution meets
the diploma standard requires the prospective supervisor's judgment.

## Experimental scope

- SIFT, 100,000 vectors sampled as in D-003; dimension 128; k=100.
- Synthetic group labels: approximately 80% in one group, remaining points
  spread over up to 999 other groups; query masks independent of geometry.
- Replaying the old D-003 workload reproduced its recorded SHA-256 exactly
  (`f2654d7afd0c2634ad19298fa29e7439772bfe01ebfa1132f5b4ee22ddd7c4a6`).
  All four database/metadata arrays match; only query-dependent arrays change.
- 32 new development queries, excluding every D-003 query ID; a separate
  32-query test split was prepared but **not searched**.
- One index seed, M=32 and efConstruction=200. One CPU thread, one warmup,
  three timed repetitions, median per query. Exact distances use float64
  accumulation; upstream ANN candidate discovery uses its own arithmetic.
- The full exploratory grid and ACORN pilot overlapped. Their quality
  measurements are retained, but latency conclusions below use subsequent
  **sequential** runs. No statistical significance claim is made.
- 47 tests passed locally, including the opt-in compiled ACORN boundary test.
  Default tests skip that one test until the external binary is supplied.

## Quality: the missing baseline changes the interpretation

With only 1% of the database eligible:

| Method/configuration | Item recall | Fill |
|---|---:|---:|
| Exact prefilter | 100% | 100% |
| FAISS native selector, initial pool 5000, cap 5000 | 42.75% | 42.97% |
| Actual ACORN-1, pool 800 | 99.47% | 100% |
| Actual ACORN-1, pool 3200 | 99.59% | 100% |
| Custom resumed level-0 traversal, 2048 expansions, predicate before shortlist cap | 27.50% | 27.59% |

The ACORN result rules out treating the FAISS wrapper's failure as an unmet
filtered-retrieval problem. The custom traversal is a diagnostic control;
its poor result is not a general negative result about resumable ANN.

With 10% eligibility, the custom traversal at the same 2048 expansions gives:

| Shortlist rule | Item recall | Fill |
|---|---:|---:|
| Group cap 1 before filtering | 65.34% | 92.41% |
| Group cap 4 before filtering | 98.41% | 100% |
| Group cap 16 before filtering | 99.09% | 100% |
| Unlimited group cap before filtering, finite pool | 95.28% | 95.53% |
| Filtering before the group cap | 99.22% | 100% |

This isolates a composition problem: the closest group representative may be
ineligible, while a slightly farther representative is usable. Unlimited caps
also allow the dominant group to consume the finite shortlist. Filtering
first avoids both effects in this control. These caps affect outputs, not
graph traversal; they are not an implementation of the ICML paper's queues.

## Resumption: real savings, not novelty

For the sequential schedule with checkpoints 512 and 2048, retained state
performs a mean **13,592.875** distance evaluations per query; restarting
performs **19,535.1875**, a **30.42% reduction**. Final selections are identical
between the two. At 10% eligibility with predicate-first cap=1, median
cumulative latency is 10.70 ms resumed versus 14.94 ms restarted (about 1.40x).
Both include selection at each checkpoint; neither is a one-shot last-budget
measurement. Retained state uses additional per-query memory.

All 1,152 selections shared between the full exploratory grid and focused
sequential rerun matched exactly, including references. All final-output
eligibility, uniqueness, duplicate-ID and ID-range violation counts were zero.
The empty-mask ACORN integration test exercises the upstream seed-candidate
edge case; final outputs remain valid because eligibility is rechecked.

## Limitations that affect topic selection

1. **Only one real vector corpus, synthetic constraints.** We still lack a
   runnable natural-data joint-filter/group benchmark with its complete
   published protocol. Dataset overlap is not numerical result reproduction.
   The original prerequisite of comparable published results is only partly
   met by these adapted experiments.
2. **No held-out confirmation yet.** Development queries informed this report;
   do not describe the numbers as a final independent evaluation.
3. **ACORN-1 is one baseline, not all ACORN.** Gamma variants and parameters
   are not tuned. Generic upstream build versus pip FAISS is also not a
   controlled comparison of algorithmic implementation quality.
4. **No expanded official DiskANN measurements in D-004.** The adapter and
   development-only export support caps 1/4/16/64 and NumDiverse=16, but that
   optional run remains unexecuted. D-003's NumDiverse=1 result does not settle
   the performance of tuned diverse construction.
5. **No scaling conclusion.** At 100k, a 1% exact scan touches only 1000 vectors.
   Larger databases or different dimensions/query predicates could change
   the crossover. The present numbers cannot dismiss ANN in general.
6. **No reputation semantics.** Fixed declared groups do not model uncertain
   identity, independence, malicious coordination, or truth; those need their
   own task, metrics, data and baselines.

## Bounded next decision

Keep the held-out queries unused for now. Before investing in a new graph,
make at most one further comparison round: expanded official diverse search,
then a larger-N exact/ACORN comparison if resources allow. Freeze all choices
before using the held-out split. A credible positive result must show a
recall/latency/memory tradeoff against the best available baseline, not merely
against the weak earlier wrapper.

If there is no useful crossover or no natural benchmark with published
comparators, promote A/B for the diploma proposal and retain D as a personal
research/benchmarking project. Given the November 19–24 selection window,
do not spend the remaining preparation time on a bespoke graph implementation
before resolving those two prerequisites.

The adjacent TIMINGS.md records the sequential timing table and observed run
durations. All evidence is development-only and machine-specific.
