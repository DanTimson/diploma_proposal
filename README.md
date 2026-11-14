Diploma proposal starters — updated 14 November 2026

**Current priority: D first.** Start with `topics/D_hnsw/README.md` and its `PROTOCOL.md`. D tests whether HNSW offers a useful application-level quality/cost trade-off before committing to graph modifications. A and B remain the fallback research proposals; C remains conditional. The original A/B audits below remain valid and unchanged. Topic D is delivered through a review patch, not a remote repository write.

Two eligible proposals and one conditional reserve for the November 19–24 selection window. Weekend review: November 14–15. These are inception-stage packages corresponding to the requested “5–10%”: question, prior results, accessible artifacts, evaluation contract and small tested tools. They are not completed diplomas and contain no new scientific performance claim.

**Read first:** `BRIEFING.md`, then `topics/A_notes/PROPOSAL_RU.md` and `topics/B_evidence/PROPOSAL_RU.md`. The recommendation is to discuss both with prospective supervisors: A best preserves the reputation objective; B has the strongest immediately verified experimental foundation. C is a reserve because exact paper-table comparability is unresolved.

**Already done.** Pinned upstream code revisions; downloaded actual source artifacts; recomputed 3,380 Community Notes summary rows from counts; downloaded SciFact-Open and replayed its saved model predictions with both the upstream evaluator and an independent P/R/F1 implementation; audited processed RecAD data; implemented mechanism probes and tested metric edge cases. Detailed evidence is in `results/` and `evidence/`.

**Not done.** No new model trained, no Community Notes simulator/scorer rerun, no RecAD training, no proposed method evaluated on a real benchmark, no HSE supervisor availability checked. The JSON plans declare workloads; they are not full experiment executors. No raw third-party datasets or upstream source trees are redistributed in this package.

**Quick start (Python 3.10+).** From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[audit]'
python -m unittest discover -s tests -v
diploma-starters smoke
diploma-starters plan topics/A_notes/experiment.json
diploma-starters plan topics/B_evidence/experiment.json
```

The base package and tests use only the standard library. The audit extra adds pandas and pyarrow for released Parquet predictions. Upstream projects should have separate environments; do not force their old training dependencies into this environment.

**Recheck the two evidence anchors.** Network downloads are explicit and checked against SHA-256. SciFact-Open is about 287 MB compressed. Its URL contains `latest`, but the downloader rejects changed bytes rather than silently changing the benchmark.

```bash
diploma-starters fetch cn_summary
diploma-starters audit-notes external/cn_summary.csv
diploma-starters fetch scifact_open
diploma-starters extract-scifact external/scifact_open.tar.gz external/scifact-open
diploma-starters audit-scifact external/scifact-open/data/claims.jsonl external/scifact-open/prediction/model_predictions.parqet
```

The extractor only writes four selected data/prediction files. The full 500k abstract corpus remains inside the archive; using `corpus_candidates.jsonl` is NOT full-corpus retrieval. See topic B's protocol before running retrieval experiments.

**Optional reserve-data audit.**

```bash
diploma-starters fetch recad_ml1m
diploma-starters audit-recad external/recad_ml1m.zip
```

**Directory guide.** `topics/` contains Russian supervisor proposals, protocols and declared workloads. `src/` contains original small tools. `tests/` checks metric semantics, known failure cases and archive handling. `evidence/` records pinned sources and comparability decisions. `results/` contains audits and smoke results, each labeled by what was actually run. Prior strategic analysis is retained separately; this package narrows it for the selection deadline.

Do not present the synthetic smoke example as an improvement over Community Notes or MultiVerS. Do not describe a saved-prediction replay as retraining reproduction. New experimental settings must compare all baselines locally under the same protocol.
