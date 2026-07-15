# Rotation-Robust Information-Extraction Design

## Separation of responsibilities

The pipeline has one controlling path and one diagnostic branch:

```text
controlling: input -> pages -> OCR candidates -> language route -> entities
                   -> relations/fields -> schema validation -> atomic JSON

diagnostic:  page -> preserved handcrafted features/PCA/K-Means -> display only
```

The controlling path never consumes a K-Means cluster, zone, or exact-angle
estimate. `safe_kmeans_display` catches every diagnostic failure and returns
nullable display fields. This preserves the historical baseline without
letting its roughly 38% zone accuracy reduce extraction availability.

## Input and geometry

`src/inference/document_io.py` loads supported images and PDFs with explicit
limits. Images are EXIF-transposed, flattened to RGB on white, and checked for
minimum/maximum dimensions and pixels. PyMuPDF renders each unencrypted PDF
page independently and enforces a page cap.

`src/information_extraction/geometry.py` represents expansion rotations with a
3x3 homogeneous matrix. Pixels, polygons, and bounding boxes share that exact
transform. Coordinates are clipped only after transformation. The inverse
matrix maps selected OCR evidence to the original page coordinate system.

Training draws a deterministic continuous angle in [0, 360) while retaining an
upright fraction. Validation uses configured fixed angles. The inference
candidate set is 0/90/180/270 plus an optional user/evidence-supplied deskew
angle. The old zone-guided exact-angle estimator is disabled.

## OCR environments and process boundary

PaddlePaddle GPU 3.3.0 and PyTorch CUDA 2.8.0 bundle incompatible cuDNN DLLs on
this Windows host. Importing both CUDA frameworks in one process is unsafe.
The design therefore uses:

- `ie-ocr`: Paddle GPU, PaddleOCR 3.7.0, CPU-only PyTorch required by PaddleX;
- `ie-layout`: CUDA PyTorch, Transformers 4.57.6, SentencePiece, Accelerate.

The main process owns Paddle models. `SubprocessLayoutEntityExtractor` launches
`scripts/layout_entity_worker.py` with the layout interpreter and exchanges
one JSON object per line. Startup, request, protocol, checkpoint, and worker
termination failures are explicit. The main pipeline may fall back to generic
entities with a warning when no learned checkpoint is configured.

All large roots and cache variables resolve below
`D:\CSX4201\vision-info-extraction-assets`. Storage gates run before setup and
training.

## OCR models and routing

`ModelRegistry` reads `reports/ocr/model_setup.json`, resolves exact local
artifacts, and verifies SHA-256 inventories. The adapter constructs two routes:

| Route | Detector | Recognizer |
|---|---|---|
| general | `PP-OCRv6_medium_det` | `PP-OCRv6_medium_rec` |
| thai | `PP-OCRv6_medium_det` | `th_PP-OCRv5_mobile_rec` |

Auto mode evaluates the general route first unless explicit mode forces Thai.
Thai is also evaluated for Thai metadata/hints, Thai Unicode evidence, weak or
empty general output, low valid-character ratio, or general mean confidence
below the calibrated 0.75 center. Candidate scores combine
confidence, text length, detection coverage, character validity, script
consistency, line alignment, and garbage/duplicate penalties. Route and
orientation choices are recorded in output provenance.

OCR caching uses a digest of source bytes, model artifact hashes, route,
preprocessing version, candidate transforms, and relevant settings. Private
results bypass public caches.

## Annotation and model data

Dataset adapters normalize source annotations to a shared schema containing
stable page/document IDs, source provenance, image dimensions, tokens,
polygons, boxes, BIO labels, entities, relations, canonical fields, split, and
privacy status.

- SROIE: quadrilateral OCR rows and canonical receipt fields; Windows-1252
  fallback; malformed/degenerate rows skipped with provenance.
- FUNSD: words, entity categories, and entity links.
- FATURA: invoice boxes/fields with source coordinates clipped to image bounds.
- CORU: supported full-document KIE/QA components; line crops and text-only
  components excluded; missing/malformed source annotations classified.

Model preparation runs PaddleOCR on a bounded public selection and aligns OCR
tokens to annotated geometry/text. Coverage below 0.70 is excluded. Private
rows cannot be emitted as training examples.

## Layout model

`LayoutXLMTextLayoutForTokenClassification` uses the
`microsoft/layoutxlm-base` tokenizer, token embeddings, normalized 2D box
embeddings, and LayoutLMv2 encoder. It omits the visual backbone and Detectron2
dependency. This is a layout-aware text model, not full multimodal LayoutXLM.

Training uses batch size 1, gradient accumulation, mixed precision, gradient
clipping, validation selection, early stopping, atomic state records, and a
reload comparison. The relation head scores entity pairs from label and
geometry features. Current learned relation training is a lifecycle smoke;
production inference also applies deterministic geometry-aware key/value
relations and canonical rules.

The smoke checkpoint license inherits CC-BY-NC-SA-4.0 constraints from the
source model.

## Output contract

`schemas/inference_output.schema.json` requires:

- document/source metadata;
- document type and language decision;
- nullable display-only rotation output;
- page-specific size, orientation, full text, OCR words/lines, polygons and
  boxes, entities, relations, tables, warnings, and transforms;
- all canonical fields, with `null` for unsupported values;
- evidence, confidence, method, and page number for every non-null field;
- processing duration, device, version, and privacy mode.

Validation precedes atomic write. A private result must target the configured
ignored private root.

## Evaluation design

Smoke evaluation selects one usable public page per dataset and runs 0, 45,
90, and 270-degree inputs. It reports OCR, entity, relation, field, document,
orientation, timing, dataset, language, text-detection, recognized-text
coverage, and rotation-retention metrics. OCR
error denominators exclude samples that have no reference token text and
report reference coverage explicitly.

CORU has zero model fit rows and serves as a natural unseen-dataset check. Its
generic output counts are meaningful, but CER/WER are unavailable because the
selected annotation has no source token geometry. This is not equivalent to a
separately retrained leave-one-dataset-out benchmark.

Private Gmail evaluation runs local inference and writes aggregate counts only.
Synthetic cross-surface integration runs on D: and commits only a hash-bound,
text-free evidence summary; the complete verifier re-hashes and independently
inspects every schema-valid output.

## Failure boundaries

The workflow stops or returns explicit errors for raw-data drift, insufficient
disk reserve, missing/hashing-failed models, unsupported/corrupt/encrypted
inputs, oversized documents, invalid geometry/schema, missing/incompatible
checkpoints, worker protocol failure, private-path violations, or detected
private publication candidates. K-Means failure is the deliberate exception:
it becomes a warning because the branch is non-controlling.
