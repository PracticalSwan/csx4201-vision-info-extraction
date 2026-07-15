# Information-Extraction Requirements

Acceptance criteria and current evidence for the rotation-robust pre-model.

## Scope

The main result is structured information extraction from images and PDFs,
including rotated, multilingual, multipage, and unknown-type inputs. The
existing four-cluster K-Means model is a display-only diagnostic. Its weak
quadrant prediction and the failed exact-angle experiment cannot control or
block the main pipeline.

## Functional requirements

| ID | Requirement | Validated state |
|---|---|---|
| IE-001 | Accept supported images plus single- and multipage PDFs. | Pass in unit tests and local image/two-page PDF integration. |
| IE-002 | Handle EXIF, grayscale, RGB/RGBA, transparent, and CMYK inputs without losing page geometry. | Pass in synthetic tests; transparency flattens on white. |
| IE-003 | Detect text with `PP-OCRv6_medium_det`. | Pass; exact artifact and runtime model identity verified. |
| IE-004 | Recognize general text with `PP-OCRv6_medium_rec`. | Pass for English/Turkish-compatible general route smoke. |
| IE-005 | Recognize Thai/mixed text with `th_PP-OCRv5_mobile_rec`. | Pass for Thai model and mixed-language multipage smoke. |
| IE-006 | Select OCR orientation independently of K-Means. | Pass; cardinal candidates and optional supplied deskew use OCR evidence only. |
| IE-007 | Preserve polygons and boxes when rotating and mapping results back. | Pass in homogeneous-transform and clipping tests. |
| IE-008 | Produce layout entities, key-value pairs, canonical fields, and evidence. | Pass functionally; bounded quality metrics remain low. |
| IE-009 | Return generic OCR/entities/key-value output for unknown document types. | Pass in unit and real CLI synthetic integration. |
| IE-010 | Validate all output against versioned JSON Schema. | Pass; missing canonical fields are explicit `null`. |
| IE-011 | Keep pages separate in multipage output. | Pass; page number, geometry, OCR, entities, and relations are page-specific. |
| IE-012 | Make K-Means display-only and failure-isolated. | Pass; disabled, wrong, or missing artifacts cannot block extraction. |

## Data and training requirements

| ID | Requirement | Validated state |
|---|---|---|
| DT-001 | Treat `data/raw` as read-only. | Pass; raw verifier unchanged. |
| DT-002 | Normalize SROIE, FUNSD, FATURA, and supported CORU annotations without changing sources. | Pass: 12,433 records; source defects reported separately. |
| DT-003 | Reuse document-safe public splits. | Pass; model manifests retain existing project split IDs. |
| DT-004 | Fit only public data and keep Gmail private-test only. | Pass; `gmail_fit_rows` is 0 in model, training, and evaluation reports. |
| DT-005 | Align OCR tokens to labels and exclude insufficient alignments. | Pass; 0.70 coverage gate and explicit exclusion reasons. |
| DT-006 | Apply continuous training rotations with aligned polygon/box transforms. | Pass in data loader and model tests. |
| DT-007 | Save and reload model/tokenizer/label map/relation head. | Pass; maximum reloaded logit difference is 0.0. |
| DT-008 | Report training scope honestly. | Pass; only the smoke profile was trained and no final-quality claim is made. |

## Safety and reproducibility requirements

| ID | Requirement | Validated state |
|---|---|---|
| SF-001 | Preserve at least 15 GiB free on C: and D: before downloads/training. | Pass; latest report records about 45.25/386.12 GiB free. |
| SF-002 | Place model/cache/checkpoint/data assets on D:. | Pass; setup and runtime reports resolve all large roots below the external asset root. |
| SF-003 | Detect incomplete or hash-mismatched OCR artifacts. | Pass in model registry and download verification. |
| SF-004 | Isolate Paddle CUDA and PyTorch CUDA processes on Windows. | Pass; two environment probes and end-to-end subprocess inference. |
| SF-005 | Cache OCR only with source, model, preprocessing, route, and transform provenance. | Pass in cache tests. |
| SF-006 | Return actionable input/model/storage errors without fabricated output. | Pass for missing/corrupt/unsupported/encrypted/oversize inputs and checkpoint failures. |
| SF-007 | Keep private output under ignored private roots. | Pass; public path rejection and aggregate-only private report tests. |
| SF-008 | Preserve the previous rotation verifier and artifacts. | Pass; 20/20 checks and artifact reload remain intact. |
| SF-009 | Make synthetic integration evidence reproducible and tamper-evident. | Pass; tracked runner plus independent source/model/config/checkpoint/fixture/output SHA-256 and semantic verification. |

## Measured quality

The following are observations, not acceptance thresholds:

- OCR-only aligned smoke subset: CER 0.5448, WER 0.6629, 0% empty output.
- Rule baseline: 4/37 applicable fields correct (10.81%).
- Layout smoke validation: loss 2.3490, token accuracy 10.07%.
- Public angle smoke: 16/16 completed, CER 0.7497 and WER 0.8856
  across the 12 runs with OCR reference text.
- Recognized-text coverage: 0.2503 on referenced runs.
- Text-detection precision/recall/F1: 0.5483/0.3333/0.4146 at polygon IoU 0.5.
- Entity precision/recall/F1: 0.0062/0.0217/0.0096.
- Relation precision/recall/F1: 0/0/0.
- Canonical-field accuracy: 0.05.
- Orientation exact accuracy: 0.375 overall and 0 at 45 degrees.
- Rotation-retention entity-F1 ratio: 0.9856, based on two very weak scores.

These results do not satisfy any plausible final-quality target. A final model
requires larger public dataset preparation, training, and an agreed benchmark.

## Open acceptance decisions

- professor-approved field/document scope and minimum quality thresholds;
- official held-out protocol and whether a natural CORU holdout is sufficient;
- whether cardinal OCR candidates plus optional fine-angle evidence satisfy
  “any angle,” or an approved supervised orientation model is required;
- zone boundary semantics and whether K-Means is academically mandatory;
- final deliverable format and approval to redistribute the noncommercial
  LayoutXLM-derived checkpoint.
