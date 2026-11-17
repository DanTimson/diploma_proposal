# D-008 prospective stop/go rule — fixed before inspecting D-008 results

## What D is being asked to justify

D remains the primary proposal only if it offers a credible algorithmic improvement
for filtered, one-per-group nearest-neighbor retrieval. Reproducible benchmarking
and a useful implementation are fallback academic outcomes, not automatic reasons
to retain D as the primary project. A comparative-only diploma requires a separate,
explicit decision with the user/supervisor. Failure of this gate defaults to A/B.

This is a project-selection threshold, not proof that all possible D-related ideas
are unviable or that a method is novel. A positive benchmark is not sufficient to
claim novelty; the surviving mechanism must also be distinguished from prior work.

## Final development round

One candidate: alternate nearest eligible pending-node expansion with ordinary
nearest-node expansion. Ineligible and same-group nodes remain routable; eligible
items are considered before group-answer retention. This is different from merely
filtering before grouping: D-006 already did that. Ordinary routing is the paired
control. Filtering after group retention is a diagnostic ablation, not a candidate.

Frozen settings: existing FANNS-medium vectors/groups/date cuts, same first 16
development queries; beam 128, alternating period two; actual FAISS upper descent;
total distance budgets 1024, 4096, 16384. SIMD distances and compiled candidate
selection are used for the ordinary control too. No graph rebuild or GPU change.

Primary regimes: EM+groups, 10%-date+groups, 50%-date+groups at k=100. Groups-only
and k=10 results are descriptive controls and cannot rescue a failed primary gate.
One common candidate budget must pass at least two of the three primary regimes.
No new thresholds, extra policies, larger grids, or post-hoc favorable query subsets.

Per regime, quality requires:
- Mean representative/item recall >=95%, mean fill >=99%.
- At least 15 of the 16 queries have item recall >=90%; worst query >=80%.
- All queries feasible; zero constraint violations.

Compare against the fastest qualifying tested baseline in that regime: SIMD exact,
independently tuned FAISS, published upstream compositions, and the corrected
ordinary native control. Select by geometric-mean per-query latency. This compares
methods satisfying a fixed quality requirement; it does not assert identical recall
when exact search is the baseline. Larger shortlists outside the frozen grid cannot
be used to inflate or dismiss the observed gate after seeing results.

Candidate must reduce geometric-mean paired latency by >=20% (baseline/candidate
speedup >=1.25), be faster on >=12/16 queries, and have the paired-query bootstrap
95% lower speedup bound >1. Bootstrap: 2000 resamples, seed 80017. Upstream cost is
still search+selection measured separately, so a pass against it is provisional.
The auxiliary eligible queue indexes the same 128 slots but consumes additional
bookkeeping memory; it is not claimed to have identical scratch-memory usage.

These are deliberately material engineering thresholds, not a conventional
significance standard. Testing several budgets makes development estimates optimistic;
bootstrap on development queries does not remove selection bias.

## Outcomes we commit to

1. **No candidate passes two primary regimes:** stop developing D as the primary
   algorithmic proposal. Archive the negative result, then reopen A/B using the
   original criteria (available datasets, published comparisons, achievable scope).
   A better reranker or a baseline tuning win alone does not count as D's contribution.
2. **A candidate passes:** freeze exactly one policy/budget and the two passing
   regimes before any hold-out access. Tie-break: most passing regimes, then lower
   budget. At most one confirmation round, with the existing untouched 32 query IDs
   and a second HNSW construction seed; tune neither candidate nor baselines on it.
   Confirm representative recall >=95%, fill >=99%, >=30/32 queries at recall >=90%,
   worst >=80%; >=20% paired latency reduction, >=24/32 latency wins, and bootstrap
   lower bound >1 in the two frozen regimes on both HNSW constructions. Use integrated
   upstream timing if it is the decisive baseline. A failure switches the main effort
   to A/B. Ambiguity does not earn an unlimited third round.
3. A positive development signal also gets a bounded prior-art check before
   confirmation: one focused work session, at most two hours. If the proposed
   distinction reduces to an existing method/composition, do not label it novel;
   switch to A/B unless the user explicitly chooses a reproduction-oriented diploma.
4. Invalid measurements are not scientific negative evidence. Allow one repair-only
   rerun for a demonstrated correctness/measurement bug, preserving this policy/grid.
   If it still cannot be evaluated reliably within that bound, park D and prioritize
   A/B rather than repeatedly postponing topic selection.

No promise of a novel algorithm should be made to a supervisor from the development
screen. If confirmation cannot be completed in time for the selection window, D
remains a backup proposal; A/B preparation takes precedence.

This document is included in the run's source hashes. Changing it after a run begins
invalidates that run's decision plan rather than silently moving the goalposts.
