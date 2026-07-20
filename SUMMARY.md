# Summary — Final Rotation-Robust Information-Extraction Pre-Model

**Project:** CSX4201 vision-info-extraction
**Verified through:** 2026-07-21

## Outcome

The workspace now contains a complete working public-trained pre-model, not
only a smoke lifecycle. Images and multipage PDFs pass through independent OCR
orientation and fine-deskew selection, exact general/Thai PaddleOCR models, a
calibrated multi-task LayoutXLM text-and-2D-layout encoder, learned entities,
document type, canonical evidence, typed relations, evidence/arithmetic
validation, table and generic key/value fallbacks, and a versioned JSON output
contract.

The preserved K-Means quadrant experiment remains an auxiliary display branch.
It never controls OCR or extraction. Its weak mapped accuracy and failed
exact-angle estimator are reported, not hidden.

## Final evidence snapshot

| Surface | Result |
|---|---|
| Raw integrity | 128,793 files; 35,459,126,772 bytes; public/private separation retained |
| Normalized public population | 12,433 pages; zero Gmail fit rows; zero leakage across 29,886 identities |
| Final model data | 11,684 examples; 7,782 train; 1,243 dev-select; 763 calibration; 1,896 test; 1,261 CORU pages held wholly unseen |
| Final training | Four epochs; 7,812 optimizer steps; epoch 4 robustness-aware selection score 0.824160; reload max difference 0.0 |
| Locked layout test | 1,760 examples; calibrated entity/canonical/relation F1 0.9813/0.9814/0.4632; 97.56% document coverage at 100% selective accuracy |
| Layout rotation | 18 angles; minimum entity/canonical/relation F1 0.7491/0.9360/0.3434; entity retention at least 95.30% |
| End-to-end rotation | 72/72 nonempty; bounded public OCR coverage 0.4026–0.4368; entity F1 0.1314–0.1830; synthetic Thai 18/18 |
| Unseen CORU | 100/100 pages; 78.53% QA-answer text recall; 15.68% canonical exact match |
| Private operation | 26/26 documents and 203/203 pages; public aggregate only; no private filename/text/image/per-document output |
| Verification | Final report compilation passed; complete IE verifier 46/46; exact OCR and hash-bound integration passed |
| Automated tests | Host suite: 243 passed, 2 environment-dependent skips; OCR-runtime partition: 122 passed; CUDA-layout partition: 2 passed |

## Portable product and publication

The final model is available through a one-command CLI and repaired local GUI.
The GUI previews images and first-page PDF renders, uses one progress surface,
and keeps long OCR and run-log output independently scrollable. Extraction is
local and requires no OpenAI API key.

The public `v1.0.0-build-week` Release contains a privacy-audited
weights-included archive for Windows and a Docker-backed macOS route. The
1,153,305,967-byte ZIP has SHA-256
`e8fc8229235c42436a487d687f75888c5f7713a69cd94c95459b55fb0d046dc6`
and was built from clean commit
`20739bfb3d6ff4b3b03f973aa8040ca152353a64`. ZIP integrity, model manifests,
portable doctor probes, and a full CPU sample extraction pass. Windows native
GPU and Docker Linux/AMD64 CPU output parity was verified earlier; physical
Apple hardware remains untested.

OpenAI Build Week submission `1102544` is `Submitted` in the Work &
Productivity track. It uses the public 2:54 demo, public repository, Thailand
as the owner-confirmed country, and `/feedback` Session ID
`019f7669-11fd-7923-ad68-ea1a09bd7d74`.

## Model and runtime

The final checkpoint uses `microsoft/layoutxlm-base` multilingual text and
normalized 2D-layout embeddings with entity, document, canonical-evidence, and
real relation heads. It omits the Detectron2 visual backbone. Dynamic training
uses 60% upright and 40% arbitrary-angle examples; checkpoint selection also
includes a fixed 37° slice.

Checkpoint:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final
```

Model SHA-256:

```text
34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189
```

Paddle GPU and CUDA PyTorch run in separate Python 3.10 processes to avoid a
Windows cuDNN DLL collision. All large environments, caches, datasets,
checkpoints, generated outputs, and private operational results remain on D:.
The source and derived checkpoint license is CC-BY-NC-SA-4.0.

## What the scores mean

The layout heads are accurate when evaluated on reference tokens and boxes,
but the real OCR pipeline substantially limits end-to-end extraction. On the
three-page angle sample, SROIE OCR is strong while the selected FATURA/FUNSD
examples are weak; aggregate text coverage stays near 0.4. Sparse FUNSD-only
relation labels further constrain learned relations. These are current model
limitations, not verifier failures.

CORU contributes no fit or selection row. Its 100-page result measures whether
known answer strings appear in OCR and whether canonical values match exactly;
it does not invent token-level entity/relation ground truth. Private Gmail
results prove local operation only and are never accuracy evidence.

## Complete versus open

Complete:

- public normalization, leakage-safe splits, final multi-stream data build;
- four-epoch public-only training, resume state, reload check, calibration;
- exact general/Thai OCR verification and automatic arbitrary-angle deskew;
- image, PDF, multipage, rotated, unknown-type, and Thai inference;
- locked in-domain, 18-angle layout, 18-angle end-to-end, and 100-page unseen
  evaluation;
- schema validation, private path/cache gates, aggregate-only private testing;
- cryptographically bound integration evidence and final report bundle;
- preserved, failure-isolated K-Means display baseline.

Still open research/product decisions:

- professor-approved canonical fields, document types, and official quality
  thresholds;
- a compatible labeled public Thai benchmark;
- broader OCR/domain adaptation and stronger relation supervision;
- an approved orientation/zone method if the professor requires more than the
  preserved K-Means diagnostic plus independent OCR correction;
- any future commercial redistribution path, because the inherited
  LayoutXLM-derived checkpoint is CC BY-NC-SA 4.0.

The result is a complete working academic pre-model with measured limitations,
not a claim of production readiness.
