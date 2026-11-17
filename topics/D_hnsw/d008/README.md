# D-008 — last bounded development round

Requires D-006, D-007 and D-007a. Adds files only; preserves previous code/results.
Read DECISION.md before running. Its thresholds and implementation are included
in the plan hashes; do not revise them after seeing this round's outcome.

## Commands

Extract the delivery ZIP into `.local/d008-delivery`, then:

```bash
source .venv/bin/activate
git apply --check .local/d008-delivery/D-008-final-screen.patch
git apply .local/d008-delivery/D-008-final-screen.patch
python -m unittest discover -s tests -v
bash scripts/run_d008.sh screen &&
bash scripts/run_d008.sh decide &&
bash scripts/run_d008.sh package
```

Send D-008-review.zip including DECISION.json, even when the verdict is STOP.
The same 16 development queries and saved indexes are reused. Three timed repeats
follow warmup. The held-out queries are untouched. The screen has more diagnostic
work than D-007a because its fixed budget grid includes 16,384 distances; the prior
9.6-minute runtime is not a reliable prediction. No index rebuild is required.

`bash scripts/run_d008.sh smoke` uses the previous FANNS-small prepared directory
with two queries and reduced settings, without upstream binaries. It cannot pass
the development decision gate. Full plan changes require a new output directory;
identical-command resume verifies cached cells. The D-007a monotonic-clock logging
probe is reused; upstream measurements are re-collected in this run with compiled
downstream selection. No original upstream files or indexes are changed.

## Actual hypothesis and controls

D-006 already filtered discoveries before updating group answers. D-008 does not
present this ordering as a new algorithm. It explicitly compares before/after
retention on the same routing procedure. After modes keep the best unfiltered
member of each discovered group and then apply the predicate at output; this is
an ablation, not a claim to reproduce published traversal internally.

There is one new policy: every second expansion takes the nearest eligible pending
node, if one exists; other expansions take the nearest pending node regardless of
eligibility. Ineligible and same-group nodes remain routing candidates. All modes
use nearest-distance eviction with a shared bound of 128 pending nodes. An auxiliary
eligible index references the same slots; its extra bookkeeping memory is not free.
No group cap or reserve is applied. Four cells isolate routing and retention:
ordinary/before, ordinary/after, alternating/before, alternating/after.

All four use real FAISS upper-layer descent and SIMD float64 level-zero distances.
The upper-level calls are charged to the total distance budget. This remains a
diagnostic traversal over an extracted graph, not a new faithful HNSW implementation.
Retained-group discovery counters in after modes include ineligible group winners;
they must not be interpreted as eligible coverage. Routing counters remain comparable.

Candidate reranking now calls the compiled SIMD exact selector on the explicit
eligible candidate IDs: it avoids materializing float64 candidate-vector matrices
in Python. FAISS and upstream candidates both receive this correction. Warmup
outputs are independently checked against the previous NumPy selector outside
measured intervals, using the existing tight numerical-equivalence audit.
Ordinary native controls receive the same SIMD improvement as the candidate.

The independent FAISS ef/pool grid and published upstream construction/search grid
are unchanged from D-007. Original EM replay, exact-reference checks, hashes,
constraint validation, per-query stage timings and monotonic probe timing are retained.
Upstream costs are still separately measured search plus selection, not integrated
end-to-end timing. A marginal win against these sums is not proof of deployment benefit.

## Stop/go interpretation

The decision evaluator reads full per-query cells, not just aggregate summaries.
It permits only alternating/before to qualify as the candidate. Faster reranking,
ordinary routing, or better baseline parameters cannot qualify as novel contributions.
It selects one common budget for two primary regimes, with thresholds fixed in
DECISION.md. Development bootstrap intervals do not correct for trying multiple
budgets. A pass authorizes only one frozen confirmation round, not a novelty claim.

No mechanism passes: archive D and switch the primary proposal effort to A/B.
Do not quietly reinterpret the outcome as permission to keep tuning a comparative
diploma. That is a separate academic option requiring an explicit choice.
