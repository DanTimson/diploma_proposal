# Dataset and prior-art audit — D-003

Audit date: 2026-11-15. "Available" below is not synonymous with "paper reproduced".

| Artifact | What was actually checked | Status for this milestone |
|---|---|---|
| ICML 2025 Graph-Based Algorithms for Diverse Similarity Search | Full paper, colored objective, graph algorithms, experimental design; actual Microsoft diversity branch cloned and compiled | Algorithm artifact runs; paper's numerical plots not reproduced |
| SIFT1M | Three pinned mirror files downloaded and SHA256 verified; 1M x128 base, 10k queries, canonical top100 truth | Usable shared vector corpus; exact Flat matches canonical top100 on first eight queries |
| Paper's SIFT-skewed groups | Paper specifies 80% dominant group, remainder over 999 groups; no exact released assignment recovered | Newly seeded adaptation, explicitly not byte-identical reproduction |
| Amazon Automotive | Original Big ANN PR311 README recovered: 2,000,640 x384 vectors, 7,119 queries, BRAND/CAT/RATING metadata | All four original Azure blob URLs returned HTTP409; alternative static-site query-label URL returned404. Not ready |
| SQID / NashANN Amazon | Pinned HF dataset metadata checked; embeddings and IDs available | Color/title/description and relevance require joining upstream ESCI; not the Automotive dataset, not yet prepared |
| NashANN | Repository and paper checked; pinned source revision recorded | Candidate objective baseline, not compiled/run here |
| ACORN / FANNBench | Primary study and artifact identified | Necessary next baseline for stronger filtering claims; no adapter claimed |

## Exact sources

- ICML paper: https://proceedings.mlr.press/v267/anand25a.html
- Paper full text: https://arxiv.org/html/2502.13336v1
- Algorithm source: https://github.com/microsoft/DiskANN/tree/diversity
  commit `659041e1b12e3ab41c70ad12d11a29a883f17e15`.
- SIFT mirror: https://huggingface.co/datasets/qbo-odp/sift1m
  revision `bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b`; hashes in sources.json.
  Mirror README attributes the original TexMex release. The original HTTPS
  download returned503 in this environment and ann-benchmarks mirror returned403.
  Canonical neighbor check supports consistency, not independent proof of every byte.
- Automotive release description:
  https://github.com/harsha-simhadri/big-ann-benchmarks/pull/311
  `dataset_preparation/amazon_products/readme.md`. The four original paths are
  `https://comp21storage.blob.core.windows.net/$web/amazon-automotive/` plus
  `amazon_base.fbin`, `amazon_base_labels.txt`, `amazon_query.fbin`, `amazon_query_labels.txt`.
  HTTP failures do not establish permanent unavailability. No substitute labels invented.
- Welfare paper: https://arxiv.org/abs/2602.08742
- NashANN: https://github.com/shivi98g/NashANN commit
  `9b56053cd2ba6c95ee20d37ec096de3322e67a7b`.
- SQID: https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset
  revision `d8b4aa89ea8a4238b4c2fa30d6da1fdde4af82ba`. The published files contain
  product/query IDs and CLIP features, not the full metadata required by NashANN's
  color-cleaning procedure. Product features ~915 MB; query features ~26 MB.
- Metadata/judgments source: https://github.com/amazon-science/esci-data
- Filtering study: https://arxiv.org/abs/2508.16263
- Filtering implementation collection: https://github.com/lmccccc/FANNBench

## Meaning of "reproduction"

The present evidence verifies that the upstream algorithm implementation is runnable
and produces valid selections on an adapted corpus with prior published diverse-search
experiments. It does **not** validate the numerical paper claims, recover the original
color assignments, match its 48-thread hardware protocol, or establish superiority
against ACORN. Those distinctions should appear on a supervisor proposal slide.

The shared-corpus/published-results prerequisite is met at the *SIFT benchmark-family*
level. A claim of natural provenance-constrained application improvement still lacks
an accepted real grouped dataset. Automotive is the best direct lead but currently
blocked; SQID is a different, potentially useful objective-comparison route.

## Correcting the literature interpretation

Colored/group-capped retrieval is explicitly part of the ICML paper, not just geometric
diversity. Multi-attribute demographic fairness is a different problem from one fixed
partition cap. Group identity is neither independence nor trust. NashANN supplies a
reason to compare hard caps against soft objectives, but its own welfare/entropy scores
are not external evidence utility. Any application study needs independent judgments.
