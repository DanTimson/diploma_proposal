# D-006 validation receipt — 2026-11-17

- Base: 0c3ed5e15748547aa190141cedb49f05a33e1d2b (public D-004 checkout).
- Full available suite: 58 tests, 57 passed, one skipped. The skipped pre-existing
  ACORN integration test requires D004_ACORN_BINARY. All 11 new D-006 tests passed.
- FANNS-small end-to-end smoke: 1,000 database vectors, two development queries,
  k=10/100, budgets 64/128, seven conditions, seven diagnostic policies plus
  exact and FAISS HNSW baselines. One timed repeat after warmup.
- 182 aggregate rows / 364 per-query method rows; zero constraint violations.
- Both sampled original FANNS EM top-100 reference sets matched exactly by IDs.
- Resume preserved the modification timestamps of all 28 completed cell files.
  Attempting to change the repeat count under the same run name was rejected.
- Unit tests include a routing counterexample: group cap loses a necessary route,
  reserve restores it; ineligible-node routing; representative improvement; strict
  distance budget including entries; empty filters; deterministic ties; graph-format
  corruption; checkpoint corruption; disjoint splits; frontier feasibility checks.
- Shell syntax check passed. Patch application is separately checked on a clean
  checkout before distribution.

No medium-dataset sweep, large build, held-out evaluation, or algorithm-performance
claim was made here. The optional actual DiskANN builder path was reviewed against
the pinned upstream source; the binary graph reader was fixture-tested. The upstream
builder was **not compiled or executed here** because MKL/Boost/libaio build
dependencies are absent. Its first real run should remain visible in the user's
local preparation logs. This also means the three-construction path has not yet
received an end-to-end integration test.

The exact final state of the user's local D-005 changes was unavailable in the
public checkout. D-006 adds new paths and intentionally avoids altering those files;
the user's full local test suite is still required.
