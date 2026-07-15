# Summary — Rotation-Robust Information Extraction

**Project:** vision-info-extraction, CSX4201  
**Verified through:** 2026-07-15

## Outcome

The repository now implements the complete inference and smoke-training
lifecycle for a rotation-robust document information-extraction pre-model.
Images and single- or multipage PDFs flow through independent OCR orientation
selection, PaddleOCR general/Thai routing, layout-aware entity inference,
key-value relation extraction, canonical field rules, and a versioned JSON
contract.

The existing K-Means rotation experiment remains preserved as an auxiliary
quadrant display. Its output never controls OCR or extraction. The unreliable
exact-angle experiment remains disabled by default.

Engineering verification is complete for the bounded smoke scope. Model
quality is not final. The three-example LayoutXLM training run proves GPU
training, save/reload, subprocess inference, and relation-head lifecycle, but
its public scores are too low for a production-quality claim.

Two bounded independent reviews were completed. The final review confirmed
that reproducible integration evidence, real rotated phrase recovery, and the
required OCR/detection metrics are closed, with no reproducible completion
blocker remaining.

## Evidence snapshot

| Surface | Verified result |
|---|---|
| Raw integrity | 128,793 files; 35,459,126,772 bytes; deterministic sample hashes unchanged |
| Preserved rotation run | 8,332 rotations, 20/20 checks, about 38% public held-out zone accuracy |
| Annotation normalization | 12,433 normalized public pages; 135 source defects classified; Gmail fit rows 0 |
| Model dataset | 9 usable smoke examples; 3 train, 2 validation, 4 test |
| Required OCR models | All three exact model artifacts hash-verified and GPU-initialized |
| OCR smoke | General confidence 0.9998; Thai 0.9549; rotated phrase recovered at orientation 270 with confidence 0.9999 |
| Layout smoke training | Validation loss 2.3490; token accuracy 0.1007; checkpoint/relation reload max difference 0.0 |
| Public angle smoke | 16/16 runs completed; recognized-text coverage 0.2503; text-detection P/R/F1 0.5483/0.3333/0.4146 |
| Extraction quality | Entity F1 0.0096; relation F1 0.0; canonical-field accuracy 0.05 |
| Private operations | 2/2 pages; aggregate only; no private content or per-document public output |
| Automated tests | 158 passed, 1 skipped; 53 OCR-runtime tests; 3 CUDA-layout tests |

## Runtime design

Large assets live under:

```text
D:\CSX4201\vision-info-extraction-assets
```

Two Python 3.10 environments prevent a Windows CUDA DLL collision:

- `environments\ie-ocr`: PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, and
  CPU-only PyTorch required by PaddleX;
- `environments\ie-layout`: PyTorch 2.8.0 + CUDA 12.8 and Transformers
  4.57.6.

The executed hardware was an RTX 5050 Laptop GPU with 8,151 MiB. The final
recorded storage gate had about 45.22 GiB free on C: and 386.12 GiB on D:, so
the required 15 GiB reserves passed.

## Data and privacy

Public annotation adapters cover SROIE, FUNSD, FATURA, and full-document CORU
components. They preserve source provenance, clip source boxes to image bounds,
support Windows-1252 fallback for SROIE, and classify source omissions and
malformed CORU JSON instead of fabricating labels.

Gmail stayed private-test only. The training manifest contains zero private fit
rows. The committed private evaluation report contains counts and timings only:
no filename, path, OCR text, image, identifier, or per-document prediction.

## Smoke model and evaluation

The model uses `microsoft/layoutxlm-base` multilingual embeddings and its 2D
layout-aware encoder without the Detectron2 visual backbone. Dynamic training
rotations use expanded white canvases and transform polygons/boxes with the
same homogeneous matrix. The source checkpoint license is
CC-BY-NC-SA-4.0.

The final smoke evaluation selected one public page per dataset and ran four
angles. OCR reference coverage was 12/16 runs. Aggregate CER was 0.7497 and WER
0.8856 only over those referenced runs. Recognized-text coverage was 0.2503;
polygon-IoU text-detection precision/recall/F1 was 0.5483/0.3333/0.4146.
Entity F1 was 0.0096; upright and
rotated entity F1 were 0.0097 and 0.0096, giving rotation retention 0.9856.
That ratio reflects two equally weak numbers and must not be read as high
absolute quality.

CORU was a natural unseen-dataset check because it had zero model fit rows.
All 4 angle runs produced nonempty generic OCR, averaging 29.25 words, 24.25
entities, and 6.75 key-value pairs. CORU has no usable OCR token reference in
this smoke protocol, so CER/WER are intentionally `null`.

## What is complete and what is not

Complete:

- reproducible OCR/layout environments and D:-backed caches;
- exact OCR model download, hashing, initialization, and smoke tests;
- public annotation schema, adapters, manifests, and verification;
- dynamic geometry transforms and model data alignment;
- CUDA smoke training and exact checkpoint reload checks;
- image/PDF/multipage/Thai/rotated/unknown-type inference;
- hash-bound, independently revalidated synthetic integration evidence;
- schema validation, error handling, privacy gates, and bounded evaluation;
- preserved rotation baseline with display-only failure isolation.

Not complete as final research evidence:

- full public OCR alignment and final multi-epoch model training;
- production-quality entity, relation, field, or orientation accuracy;
- labeled public Thai benchmark evaluation;
- separately retrained leave-one-dataset-out studies;
- professor-approved target fields, thresholds, and final test protocol.

Use the repository as a verified implementation and smoke baseline, not as a
claim that the research model is finished or accurate.
