# Research decision after D-003

Working diploma title: **Approximate vector retrieval with eligibility and group
constraints: reproduction, failure analysis, and adaptive candidate selection**.
This is a provisional systems research topic, not a claim of a new reputation system.

## Hypothesis worth testing

Diversity and eligibility need not commute. If the globally best member of group G
is ineligible, retaining only that representative can discard G's best eligible
member. A larger number of groups in a diverse candidate pool does not necessarily
restore the missing within-group alternatives. A simple cap relaxation may repair
this without any new graph construction. Test it before proposing a method.

The reverse ordering has its own cost: finding enough eligible points before group
selection can spend most of the candidate budget on one dominant group. The relevant
variables are local group concentration, eligible group count, within-group filter
variation, geometric difficulty, and selectivity, not selectivity alone.

## Minimum next study

1. Keep k=100 and item recall comparable with the diverse-NN paper. Keep a separate
   k=10 application track. Compare score regret and fill, not only group overlap.
2. Expand upstream queue caps 1/4 to 1/2/4/8/16; measure memory and comparisons.
   Tune only on designated tuning queries; freeze the policy and assess new queries
   and index constructions. Repeating fixed-query seeds is not more independent data.
3. Vary diverse construction threshold beyond this milestone's default1 and report
   build cost/edge budget. The present default must not stand in for a tuned method.
4. Add ACORN + output group cap and a truly resumable ordinary traversal. The current
   repeated searches restart; they are not sufficient to rule out that engineering fix.
5. Compare independent, group-correlated, and geometry-correlated masks. Start from
   real BRAND/category attributes if Automotive access can be restored. Do not label
   synthetic group/filter assignments as natural provenance.
6. Only if these baselines leave a robust gap, test a native joint traversal or a
   query-budget rule based on observable eligible-group concentration. Exact fallback
   remains a primary control, especially for 1% subsets.

## Rejection criteria

No algorithmic novelty claim if relaxed caps/larger pools/compiled exact/ACORN remove
the advantage at matched quality and cost. Preserve the negative finding as a study
of composition limits. Do not change the recall threshold after seeing failures;
add geometric utility measures with a separate declared acceptance contract.

If natural groups and an external utility task are available, the alternative is a
hard-cap versus soft-welfare study (D/B), not necessarily another graph algorithm.
If the intended objective is reputation propagation, return to A's aggregation
baseline: none of the current ANN results measures that objective.

## Diploma readiness

Completed: reproducible runner, direct upstream artifact, correctness controls,
adapted shared-corpus experiments, documented failure mechanism, auditable outputs.
Pending: faithful numerical paper reproduction, natural grouped dataset, stronger
filtering/iterative baselines, held-out policy evaluation, and any original method.
This is suitable as an honest preliminary proposal; supervisor acceptance determines
whether a comparative study plus extension is an adequate diploma scope.
