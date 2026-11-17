# D-008 validation — 2026-11-17

- Combined available suite: 73 tests; 72 passed, one pre-existing optional ACORN
  integration test skipped. All six new retention/decision tests passed.
- FANNS-small smoke: two frozen development queries, four grouped conditions,
  k=10/100, four routing/retention cells, independent FAISS settings, two exact
  baselines. 160 per-query rows / 80 aggregate rows; zero constraint violations.
- Hierarchical descent plus native traversal respected total distance budgets.
  Compiled candidate reductions passed independent NumPy-reference audits.
- Completed-cell resume preserved all 16 cell files. A changed plan was rejected.
- Synthetic gate tests verify positive qualification, the STOP verdict for an
  insufficient speedup, deterministic budget tie-breaking, and rejection of
  incomplete data instead of calling it a negative research result.
- The smoke correctly reports CALIBRATION_ONLY rather than qualifying D.
- Shell syntax passed; the patch is checked on a D-006/D-007/D-007a checkout before
  delivery. The user should still run their full suite including local D-005 changes.

Only smoke/correctness work was run here. No medium dataset, new graph construction,
held-out confirmation, or upstream search experiment was executed. Upstream search
uses the D-007a adapter already exercised on the user's machine; D-008 reruns it
with corrected downstream selection. Performance and the final gate remain untested
until the user's development run. No novelty or improvement is claimed.
