# Information-Extraction Requirements

Acceptance criteria and executed evidence for the final working academic
pre-model. Passing a functional requirement does not imply production-quality
accuracy; measured quality is reported separately.

## Scope

The controlling result is structured extraction from raster images and
single/multipage PDFs, including rotated, multilingual, and unfamiliar
documents. The preserved four-cluster K-Means model is a display-only
diagnostic. Its cluster, zone, confidence, and failed exact-angle experiment
cannot control or block OCR/extraction.

## Functional requirements

| ID | Requirement | Final validated state |
|---|---|---|
| IE-001 | Accept supported raster images and single/multipage PDFs. | Pass in unit tests and executable image/PDF integration; PNG/JPEG/TIFF/BMP/WebP paths covered. |
| IE-002 | Normalize EXIF, grayscale, RGB/RGBA, transparent, and CMYK inputs while preserving geometry. | Pass; transparency flattens on white and geometry transforms are tested. |
| IE-003 | Use the exact `PP-OCRv6_medium_det` artifact. | Pass; inventory SHA-256 and GPU initialization verified. |
| IE-004 | Use exact `PP-OCRv6_medium_rec` for the general route. | Pass; model identity and known-phrase recovery verified. |
| IE-005 | Use exact `th_PP-OCRv5_mobile_rec` for Thai. | Pass; Thai Unicode and mixed-language multipage routing verified. |
| IE-006 | Select OCR correction independently of K-Means. | Pass; cardinal candidates plus reliable automatic/supplied fine deskew use OCR evidence only. |
| IE-007 | Rotate pixels, polygons, boxes, entities, and relations with one transform. | Pass in homogeneous-transform, clipping, and arbitrary-angle tests. |
| IE-008 | Produce learned entities, document type, typed relations, canonical evidence, fields, and tables. | Pass functionally with calibrated abstention; quality limitations remain explicit. |
| IE-009 | Preserve useful generic output for unfamiliar types. | Pass; OCR, entities, generic key/value pairs, and schema-valid unknown type remain available. |
| IE-010 | Validate every result against a versioned JSON Schema before write. | Pass; unsupported/conflicted fields are explicit `null`. |
| IE-011 | Keep multipage geometry/output isolated by page. | Pass, including continued output after a configured page-level failure. |
| IE-012 | Make K-Means display-only and failure-isolated. | Pass; disabled/missing/wrong artifacts cannot block extraction. |
| IE-013 | Fail fast when a required final checkpoint/calibration is missing or mismatched. | Pass in CLI, worker, unit, integration, and verifier checks. |

## Data, training, and evaluation requirements

| ID | Requirement | Final validated state |
|---|---|---|
| DT-001 | Treat `data/raw` as read-only. | Pass; raw verifier remains unchanged. |
| DT-002 | Normalize supported FATURA/SROIE/FUNSD/CORU records without fabricating labels. | Pass: 12,433 public records; exclusions/source defects categorized. |
| DT-003 | Use document/duplicate-safe split identities. | Pass: 29,886 identities, zero cross-split violations. |
| DT-004 | Reserve CORU as wholly unseen domain. | Pass: all 1,261 pages are `unseen_domain_test`. |
| DT-005 | Keep Gmail private-test only. | Pass: zero private/Gmail rows in model data, training, calibration, selection, and public evaluation. |
| DT-006 | Build a full public labeled final profile with multiple OCR streams. | Pass: 11,684 examples, including 11,172 ground-truth and 512 OCR/hybrid variants. |
| DT-007 | Apply continuous rotations with aligned targets. | Pass: final training uses 60% upright/40% arbitrary-angle geometry. |
| DT-008 | Train real entity/document/canonical/relation targets. | Pass: 514,220 entity tokens, 152,875 canonical tokens, 40,954 relation pairs, 4,545 positives. |
| DT-009 | Select checkpoints without private or test data. | Pass: four bounded dev trials; final selection uses public dev-select upright/37° only. |
| DT-010 | Calibrate without private or test data and bind the result. | Pass: 708 public dev-calibration examples; exact checkpoint/build/manifest hashes. |
| DT-011 | Save/reload model, tokenizer, labels, heads, and resumable state. | Pass; checkpoint reload maximum difference 0.0; final resume state retained on D:. |
| DT-012 | Execute a locked in-domain test once, without tuning from it. | Pass: 1,760 public ground-truth examples; report is hash-bound. |

## Safety and reproducibility requirements

| ID | Requirement | Final validated state |
|---|---|---|
| SF-001 | Preserve at least 15 GiB free on C: and D: at materialization/training gates. | Pass; final complete verifier records more than 43/362 GiB free. |
| SF-002 | Keep large assets below the configured D: root. | Pass for environments, caches, examples, checkpoint, generated and private output. |
| SF-003 | Detect incomplete/hash-mismatched OCR artifacts. | Pass in registry, downloader, verifier, and tests. |
| SF-004 | Isolate Paddle CUDA from CUDA PyTorch on Windows. | Pass with persistent subprocess inference and separate environment partitions. |
| SF-005 | Bind public OCR caches to source/model/profile/route/transform provenance and exclude private data. | Pass in cache and privacy tests. |
| SF-006 | Return actionable input/model/storage/protocol errors without fabricated output. | Pass for missing, corrupt, encrypted, oversized, invalid-checkpoint, and worker failures. |
| SF-007 | Constrain detailed private outputs to ignored private roots. | Pass; public path rejection, anonymous IDs, no public visualization, aggregate-only report. |
| SF-008 | Preserve historical rotation evidence. | Pass: 20/20 rotation verifier checks and reload evidence. |
| SF-009 | Make integration evidence executable and tamper-evident. | Pass: 11 source/model/config/checkpoint/fixture/output artifacts independently re-hashed and semantically checked. |
| SF-010 | Reject large/unexpected/publication-risk Git candidates. | Pass in the complete IE verifier; final staged audit is still mandatory before push. |

## Measured quality

### Locked reference-token layout test

On 1,760 public `test_in_domain` ground-truth examples (1,761 windows):

| Head | Raw micro-F1 | Calibrated/abstained micro-F1 | Raw macro-F1 |
|---|---:|---:|---:|
| Entity | 0.9807 | 0.9813 | 0.7290 |
| Canonical evidence | 0.9792 | 0.9814 | 0.9749 |
| Relation | 0.4668 | 0.4632 | 0.5620 |

Document accuracy is 1.0. Calibrated document coverage is 0.9756 with 1.0
selective accuracy. The micro/macro gap and per-dataset slices expose class and
dataset imbalance; B-HEADER and QUESTION_ANSWER are the weakest supported
entity/relation classes.

### Rotation robustness

The 18-angle layout-only grid uses 30 dataset-balanced test pages and rotates
reference geometry with the page. Minimum calibrated scores are entity 0.7491,
canonical 0.9360, relation 0.3434, and composite 0.7227. Minimum entity and
canonical retention versus upright are 95.30% and 98.66%.

The bounded 18-angle end-to-end grid uses one real page from each labeled
dataset plus one synthetic Thai page at each angle. All 72 cases are nonempty.
Across real pages, OCR text coverage is 0.4026–0.4368, detection F1
0.3330–0.3592, entity F1 0.1314–0.1830, relation F1 0–0.0205, and field
accuracy 0.2222–0.5556. Synthetic Thai text/routing succeeds 18/18. This gap
shows that OCR, not only the layout heads, limits usable extraction.

### Unseen and private operation

All 100 deterministic CORU pages succeed with nonempty OCR. The model finds
78.53% of 4,001 QA answer strings and exactly matches 15.68% of 523 applicable
canonical fields. CORU lacks compatible token polygons, so entity/relation F1
is undefined.

Two anonymous Gmail documents succeed locally with zero failures. There is no
private ground truth, so the aggregate is an operation check—not accuracy.

## Open acceptance decisions

- professor-approved canonical fields/document types and minimum thresholds;
- whether the required four zones specifically demand K-Means, a supervised
  angle model, or only quadrant display after orientation estimation;
- exact 90/180/270 boundary ownership;
- approved labeled Thai benchmark and broader OCR/domain adaptation plan;
- final deliverable format and whether/how to redistribute the
  CC-BY-NC-SA-4.0 checkpoint.
