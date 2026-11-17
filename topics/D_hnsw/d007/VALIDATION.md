# D-007 validation — 2026-11-17

- Available combined D-004 + D-006 + D-007 suite: **67 tests**, 66 passed and
  one pre-existing optional ACORN integration test skipped (D004_ACORN_BINARY absent).
  All nine new D-007 tests passed.
- FANNS-small smoke: 1,000 vectors, two frozen development queries, four grouped
  conditions, k=10/100, two independent FAISS parameter pairs, three entry controls
  at 1,024 total distances, and scalar/SIMD exact baselines.
- 144 per-query method rows / 72 aggregate rows; zero retrieval constraint violations.
- SIMD exact selections matched scalar reference IDs in every smoke cell. Both
  sampled original FANNS EM reference sets matched. Small-workload infeasible k
  values remained explicitly marked.
- Real FAISS upper descent matched an independent neighbor-walk check. The smoke's
  hierarchical control respected its total distance budget in all measured calls.
- Compiler vectorization diagnostics confirmed the explicit distance reduction
  was vectorized on this machine. This is not a target-machine speed claim.
- Identical-command resume preserved all 16 completed cell files. Changing the
  repeat count under the same output directory was rejected.
- Logging-only upstream insertion, build-command rewriting, per-query file parsing,
  missing-result sentinel handling, and stored-vector identity checks are unit-tested.
- Shell syntax passed. The distributed patch is checked against a checkout with
  D-006 applied before delivery.

The upstream logging executable was **not compiled or executed here**, because
the upstream MKL/Boost/libaio build dependencies are absent. The user-side
`upstream-smoke` is the outstanding end-to-end integration check. It validates
returned IDs against the original executable in all six construction/search
combinations. No medium-dataset experiment or held-out evaluation was run here.

The patch adds paths only and leaves D-005/D-006 sources intact. The exact user's
local D-005 helper source was not in the public base, so the local full test suite
remains part of application validation.
