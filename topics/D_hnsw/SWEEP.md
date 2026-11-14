# D-002: bounded research batch

This extends D-001 with a resumable CPU runner and larger comparisons. It does not implement a new graph topology or claim reproduction of ACORN. The current priority remains D, with A/B available if the measured cost/quality trade-off is not useful.

## Apply and run

The patch targets commit `03aff5696401b86ff61c754956b862adf0aac1e3`. No remote writes are involved.

```bash
git apply --check D-002-research-sweep.patch
git apply D-002-research-sweep.patch
source .venv/bin/activate
python -m unittest discover -s tests -v
```

If ANN dependencies are missing, first run `bash scripts/setup_env.sh all`. The expected test count is 34 with ANN dependencies installed on Linux/WSL.

Optional fast check, no downloads:

```bash
python -m diploma_starters.sweep --plan topics/D_hnsw/sweep-smoke.json --run-dir local-runs/D-002-check
```

Run the default batch in the background:

```bash
bash scripts/run_sweep.sh start
tail -f local-runs/D-002-sweep/launcher.log
```

Ctrl-C stops `tail`, not the sweep. For foreground operation, use `python -m diploma_starters.sweep` instead. Do not run both at once. A run-directory lock prevents two runners sharing one output directory, but does not prohibit separate directories; do not run separate performance sweeps concurrently.

```bash
bash scripts/run_sweep.sh status
bash scripts/run_sweep.sh stop
bash scripts/run_sweep.sh resume
# Only when intentionally retrying failed/timed-out cases:
bash scripts/run_sweep.sh retry
```

`stop` validates the runner's PID start time before signalling it. The runner terminates its current child process group, saves state and retains completed results. Resume verifies existing benchmark receipts and their checksums. Failures/timeouts are not automatically retried; dependent cases are blocked, independent cases continue. Interrupted/pending cases resume with the normal command. Each launch gets a fresh six-hour wall-clock budget; completed benchmarks are not repeated. Normal terminal closure is supported through nohup; Windows sleep/reboot still interrupts execution.

## Default workload

The plan in `sweep.json` contains **33 benchmark jobs plus 8 preparation jobs**:

| Family | Cases | Fixed inputs |
|---|---:|---|
| Synthetic | 24 | 10k/100k base items × clean/copies/1% filter/adverse filter × 3 index seeds; 128 queries |
| SciFact transfer | 3 | All 279 claims; same corpus-only 96-dimensional LSA representation |
| Recommendation transfer | 3 | Same 256 users and same train-only SVD factors |
| FANNS medium | 3 | Original 100k × 4096 vectors; EM; query indices 1000–1127 inclusive |

Index seeds: 11, 29, 47. **Data/representation seed stays 11.** Query efs: 32, 128, 256; pool 40, cap 2560; k=10; M=16, efConstruction=100. For a given query pool, effective ef remains max(requested ef, pool). There is one warm-up and three measured repetitions. Method order is shuffled per query/repetition with a separate fixed measurement seed, avoiding the earlier consistent wrapper-order/cache advantage.

The benchmark family and source-result availability remain as in D-001. The medium FANNS query subset and local hardware do not reproduce the paper's entire timing protocol. Original labels/vectors and the exact predicate are preserved; exact top-k must agree with supplied ground truth before each case proceeds. SciFact and RecAD are fixed-representation transfer tests, not MultiVerS or RecAD attack reproductions.

## Resource envelope

- One child at a time; one FAISS/BLAS CPU thread; lower scheduling priority (nice +5); no GPU.
- Six-hour budget per launch, 30-minute per-job timeout, 12 GiB virtual address-space cap per child. These are bounds, not runtime predictions. The address-space cap is not an RSS measurement or a claim that 12 GiB will be used.
- At least 8 GiB free disk required. About **2.14 GB** of pinned source files download if none are present. Already present files are checksum-verified and reused. Partial downloads restart from scratch; completed benchmarks resume independently.
- Exact scoring is blocked to avoid whole-corpus distance temporaries. Hashing no longer copies the full vector matrix. Peak RSS is recorded for each benchmark child on Linux; it includes preparation and all simultaneously resident exact/ANN controls, not just one deployable index.
- `.venv`, datasets and logs stay local. No model/data files enter the patch. A/B/C artifacts and earlier D-001 results remain unchanged.

## What the runner records

`manifest.json` fixes the plan, Python/package versions, code hashes and source lock. Resuming with changed code, plan, interpreter location or package versions is rejected; choose a new run directory after changes. Benchmark jobs also check for changes before launching. The summary and state are written atomically after each job. Benchmark result JSON is committed atomically without overwriting an existing receipt. Damaged receipts are retained with an `.invalid-...` suffix.

`SUMMARY.md` lists every job and the fastest tested ANN setting meeting >=99% exact-group recall, >=99% fill and zero constraint violations. This is exploratory selection within the sweep, not held-out tuning. A “qualifying” row can still have worse evidence/recommendation recall, p95 latency, memory or construction cost. Read the complete JSON before drawing conclusions. No qualifying row is a useful result, not a runner failure.

`results/` inside the run directory contains all query-level results; `logs/` contains individual stdout/stderr. The top-level repository `results/` remains for deliberately reviewed summaries.

Recompute the multi-seed aggregate and conditional query-bootstrap intervals after completion:

```bash
python scripts/analyze_sweep.py --run-dir local-runs/D-002-sweep --out .local/D-002-analysis.json
```

The output must be new. Incomplete runs require an explicit `--allow-partial`. The aggregation verifies identical workload fingerprints across index seeds and rejects observed eligibility/group violations.

## Review after completion

1. Verify that FANNS ground truth agreed and no constraint violations occurred.
2. Compare the same method/ef across all three index seeds; do not pool repeated timings as independent relevance observations.
3. Inspect task recall, fill, p95, index construction and memory alongside median latency. Confirm unchanged workload fingerprints across index seeds.
4. Keep the exact-prefilter and duplicate-collapse controls. Gains against a weak exact implementation do not establish a benefit.
5. If a useful regime survives, choose a stronger representation/ACORN comparison and one specific D2 hypothesis. If it disappears, resume A/B. This batch tests retrieval; it does not judge the separate influence-propagation interpretation of HNSW-like layers.

The timers measure **one query at a time**, including per-query eligibility/selection. They are not a batched-throughput benchmark. A final systems claim needs a batched exact control where batching is allowed, and stronger filter-aware competitors. Different services can legitimately have different latency/throughput requirements.

To return the complete run for review:

```bash
mkdir -p .local
tar -czf .local/D-002-review.tar.gz -C local-runs D-002-sweep
```

Do this after stopping/completing the sweep. Inspect logs for any local paths you do not wish to share; no credentials are intentionally recorded. The archive excludes the external datasets because they live outside the run directory.
