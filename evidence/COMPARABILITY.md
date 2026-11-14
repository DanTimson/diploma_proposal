# Evidence and comparison ledger

Checked 13 November 2026. An available corpus is necessary but insufficient: exact splits, candidate pools, labels, preprocessing and metric denominators determine whether published numbers are comparable.

| Source | Status and role | What has been verified | What remains |
|---|---|---|---|
| [Community Notes are Vulnerable to Rater Bias and Manipulation](https://arxiv.org/abs/2511.02615), 2025 preprint | A's reproduction anchor | Author code, synthetic generators, reduced scorer and stored counts; 3,380 summary rows recomputed | Run one generator/scorer cell; confirm historical environment |
| [Gaming Consensus](https://proceedings.mlr.press/v306/selvam26a.html), ICML 2026 | A's important recent prior art | Paper explicitly studies strategic manipulation of core matrix factorization and reports deployed mitigations | Compare mitigation assumptions; no locally reproduced result |
| [QS-MF](https://arxiv.org/abs/2604.11224), 2026 preprint | A's related rater-quality model | Relevant alternative to geometry-only trust | No ready implementation adapter or reproduced result claimed |
| [FoolsGold](https://www.usenix.org/conference/raid2020/presentation/fung), RAID 2020 | A's conceptual dependence-defense baseline | Similarity-based Sybil defense already exists in federated learning | Adapt carefully: gradients are not rating histories |
| [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/), Findings EMNLP 2022 | B's reproduction anchor, Table 5 | Real archive, labels and predictions downloaded; official and independent P/R/F1 replay | New selection/retrieval experiment, full-corpus run |
| [MultiVerS](https://aclanthology.org/2022.findings-naacl.6/), Findings NAACL 2022 | B's released baseline | SciFact-Open saved predictions replayed | No retraining; use correct model provenance |
| [+VeriRel](https://arxiv.org/abs/2508.11122), accepted CIKM 2025 | B's recent novelty constraint | Already combines verification feedback with retrieval; evaluates SciFact-Open among other datasets | Implementation and matched-protocol comparison not yet established |
| [How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation](https://aclanthology.org/2026.emnlp-main.1915/), EMNLP 2026 | B's recent evaluation context | Publication and abstract checked | Full methodological review before freezing contribution |
| [RecAD](https://arxiv.org/abs/2309.04884), RecSys 2023 | C's conditional anchor, Table 2 | Processed ML1M downloaded; splits disjoint by user/item pair; current code inspected | Recover exact recipe; correct current validation default; reproduce one baseline |
| [TrustGRS code](https://github.com/IIE-MLY/Trsut-GRS), AAAI 2025 related work | C's recent related work | Inspected repository contains partial modules | Complete runnable reproduction not established; cannot use as readiness evidence |

The newest paper is not automatically the best reproduction anchor. Older anchors are used here because actual data and outputs were verified; recent work determines which novelty claims remain defensible.

## Results already available

`results/scifact_prediction_replay.json`: independently computed point estimates, not paper bootstrap means. MultiVerS F1 52.380952; ParagraphJoint 50.293772; Vert5Erini 36.417207. All five models' P/R/F1 agree with the upstream evaluator. Only a deprecated NumPy alias was replaced in the upstream local replay (`astype(np.int)` to `astype(int)`). The upstream evaluator is fetched separately rather than redistributed.

`results/notes_summary_audit.json`: four rates reconstructed from raw counts in all 3,380 rows. The exact fractions take precedence over ambiguous prose: suppression is helpful-but-unpublished / all helpful; waste is helpful-but-unpublished / all unpublished. Checking these arithmetic identities is not rerunning an attack.

`results/recad_data_audit.json`: 795,917 training, 93,516 validation and 96,531 test ratings, with zero pair overlap across splits. Explicit-rating defaults currently point validation at test; this does not establish what the paper's experiment used.

## Scope and reuse

Third-party code and datasets are not bundled. Download URLs, revision pins and SHA-256 hashes are provided. Licenses must be checked before redistribution or incorporation into a public product. Inspection did not find explicit licenses for several artifact repositories; public availability alone is not a redistribution license.

The implemented exact-history cap is deliberately weak: diversified fake histories and empty histories bypass it, while identical honest specialists are also downweighted. Tests cover those failures. It is a mechanism probe, not the proposed research contribution.
