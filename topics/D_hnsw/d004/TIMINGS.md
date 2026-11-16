# D-004 sequential timings

32 development queries; one CPU thread; milliseconds, median per query then percentile across queries. Separate executions ran sequentially. No confidence intervals or cross-machine claim.

| Eligibility | Method | Recall | Fill | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|
| 0.01 | acorn1_pool3200_ef512 | 99.59% | 100.00% | 19.356 | 21.661 |
| 0.01 | acorn1_pool800_ef512 | 99.47% | 100.00% | 18.907 | 25.987 |
| 0.01 | exact_cpp | 100.00% | 100.00% | 0.535 | 0.723 |
| 0.01 | faiss_native_restart_ef512 | 42.75% | 42.97% | 11.909 | 14.771 |
| 0.01 | graph_restart_pre_cap1_b2048 | 27.50% | 27.59% | 14.105 | 18.221 |
| 0.01 | graph_resume_pre_cap1_b2048 | 27.50% | 27.59% | 10.273 | 13.337 |
| 0.1 | acorn1_pool3200_ef512 | 99.91% | 100.00% | 85.250 | 98.758 |
| 0.1 | acorn1_pool800_ef512 | 99.53% | 100.00% | 22.569 | 25.592 |
| 0.1 | exact_cpp | 100.00% | 100.00% | 3.252 | 3.746 |
| 0.1 | faiss_native_restart_ef512 | 99.97% | 100.00% | 14.307 | 16.310 |
| 0.1 | graph_restart_pre_cap1_b2048 | 99.22% | 100.00% | 14.938 | 19.355 |
| 0.1 | graph_resume_pre_cap1_b2048 | 99.22% | 100.00% | 10.697 | 15.048 |

Graph rows include cumulative work at checkpoints 512 and 2048. Other methods use their own stated candidate budgets. Equal graph expansion counts are not equal distance-computation budgets across algorithms. ACORN includes timed final filtering, exact candidate reranking, and group selection.

## Observed wall-clock run durations

Elapsed between config and summary file writes (data loading before the config write and initial SIFT download excluded):

- D-004-smoke: 1.4 seconds.
- D-004-pilot: 103.5 seconds.
- D-004-timing: 72.3 seconds.
- D-004-acorn800-timing: 28.0 seconds.
- D-004-acorn-timing: 34.6 seconds.

The full pilot is approximately a two-minute job on this environment; allow several minutes on another CPU. The 800/3200 ACORN runs each rebuild twice. First-time external compilation and the approximately 525 MB download are additional costs. Timing shifts between separate ACORN runs illustrate why these numbers need repeated runs and uncertainty estimates before fine-grained claims.
