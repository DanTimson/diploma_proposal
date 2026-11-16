# D-001: HNSW applicability smoke

**D-002 update:** `SWEEP.md` documents the bounded, resumable batch, medium FANNS inputs, independent data/index seeds, lower-memory exact search and shuffled measurement order. The D-001 results below remain historical; do not mix their timings with D-002 without rerunning the controls.

Read `PROPOSAL_RU.md` for the supervisor-facing proposal, `PROTOCOL.md` for the exact question and gates, `LITERATURE.md` for prior-art limits, and `SMOKE_RESULTS.md` for measured diagnostics.

This adds an executable D0/D1 runner, not a custom HNSW constructor. It uses fixed vector representations, an exact constrained reference, FAISS Flat/HNSW, candidate-budget variants and an exact-copy collapse control. Every run writes a new JSON file; an existing output is never overwritten.

From the repository root, in WSL/Linux with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ann]'
python -m unittest discover -s tests -v
diploma-ann --out local-runs/d-clean.json synthetic --mode clean
diploma-ann --out local-runs/d-copies.json synthetic --mode copies
diploma-ann --out local-runs/d-sparse.json synthetic --mode sparse_filter
diploma-ann --out local-runs/d-adverse.json synthetic --mode adverse_filter
```

The base A/B package remains standard-library-only. D requires the `ann` extra. Its native FAISS version is pinned; CPU tests avoid depending on ROCm. Test discovery skips D tests if the extra is absent. Query timing claims require these tests to execute, not skip.

Reuse the earlier data acquisition commands:

```bash
diploma-starters fetch scifact_open
diploma-starters extract-scifact external/scifact_open.tar.gz external/scifact-open
diploma-ann --out local-runs/d-scifact.json --queries 96 scifact external/scifact-open
diploma-starters fetch recad_ml1m
diploma-ann --out local-runs/d-recad.json --queries 96 recad external/recad_ml1m.zip
```

Skip extraction if those four files already exist: the extractor intentionally refuses to overwrite them. SciFact fitting uses only the document corpus, not claims/labels. RecAD fitting reads only training ratings; test positives are evaluation-only. These are new transfer protocols, not reproductions of published model scores.

Download FANNS's pinned debugging release (about 185 MB for these five files):

```bash
diploma-starters fetch fanns_small_database_vectors --destination external/fanns-small
diploma-starters fetch fanns_small_query_vectors --destination external/fanns-small
diploma-starters fetch fanns_small_database_attributes --destination external/fanns-small
diploma-starters fetch fanns_small_em_query_attributes --destination external/fanns-small
diploma-starters fetch fanns_small_ground_truth_em --destination external/fanns-small
diploma-ann --out local-runs/d-fanns.json --queries 64 fanns external/fanns-small
```

The adapter refuses to benchmark if the local exact top-k does not match the supplied ground truth. It preserves the original vectors, L2 metric and exact-match predicate. This is a debugging set; use medium for scientific timing comparisons.

CLI flags precede the workload subcommand. Example of a larger synthetic diagnostic:

```bash
diploma-ann --out local-runs/d-100k-seed29.json --queries 128 --seed 29 --efs 64 128 256 --pool 80 --cap 2560 synthetic --n 100000 --mode sparse_filter
```

This harness keeps a dense query-by-item eligibility mask, suitable for bounded diagnostics. Do not use it for millions of items and thousands of queries without replacing masks with per-query bitmap/sparse predicate providers. Large raw data, generated models and runs stay under ignored `external/` and `local-runs/`. No third-party datasets or source trees are included in the patch.

## D-003: diverse-search baseline and joint-constraint screen

See [d003/README.md](d003/README.md) for runnable commands,
[d003/DATA_AUDIT.md](d003/DATA_AUDIT.md) for dataset/reproduction status,
[d003/RESULTS.md](d003/RESULTS.md) for the measured adaptation, and
[d003/DECISION.md](d003/DECISION.md) for the next decision.
D-001/D-002 results are preserved; D-003 changes the protocol and adds item recall.
