# Rotation-Robust Information-Extraction Design

## Separation of responsibilities

The system has one controlling path and one diagnostic branch:

```text
controlling: input -> pages -> OCR orientation/profile/route -> calibrated
             layout heads -> relations/tables/fields -> schema -> atomic JSON

diagnostic:  page -> handcrafted features/PCA/K-Means -> display only
```

The controlling path never consumes a K-Means cluster, zone, or estimated
angle. `safe_kmeans_display` catches diagnostic failures and returns nullable
display fields. The preserved baseline therefore cannot reduce OCR or
extraction availability.

## Input and geometry

`src/inference/document_io.py` accepts supported raster formats and PDFs with
explicit size/page limits. Images are EXIF-transposed, flattened to RGB on
white, and normalized without changing the source. Each unencrypted PDF page
is rendered independently.

`src/information_extraction/geometry.py` uses one 3x3 homogeneous transform for
pixels, polygons, boxes, entities, and relations. Expanded rotations keep the
whole page on a white canvas; coordinates are clipped only after the transform
and mapped back with its inverse.

Training retains an upright fraction and otherwise draws deterministic angles
from [0, 360). The selected final run used 60% upright and 40% arbitrary-angle
examples. Inference scores 0/90/180/270-degree candidates and can add an
automatic fine-deskew candidate when line evidence is reliable. K-Means and
the failed historical exact-angle estimator never supply a correction.

## Process boundary and storage

PaddlePaddle GPU and CUDA PyTorch bundle incompatible cuDNN DLLs on this
Windows host. The implementation uses two Python 3.10 environments:

- `ie-ocr`: Paddle GPU, PaddleOCR/PaddleX, and CPU-only PyTorch required by
  PaddleX;
- `ie-layout`: CUDA PyTorch 2.8.0, Transformers 4.57.6, SentencePiece, and
  Accelerate.

The OCR process owns Paddle models. `SubprocessLayoutEntityExtractor` keeps a
persistent `scripts/layout_entity_worker.py` process and exchanges one JSON
object per line. Startup, protocol, request, timeout, and checkpoint errors are
explicit. Final-profile evaluation and private testing require the learned
checkpoint; they cannot silently fall back to rules.

Environments, caches, aligned examples, checkpoints, generated documents, and
private output live under
`D:\CSX4201\vision-info-extraction-assets`. Storage gates preserve at least
15 GiB on both C: and D: before materialization or training.

## OCR models, preprocessing, and routing

`ModelRegistry` verifies exact local artifact inventories from
`reports/ocr/model_setup.json` before model construction:

| Route | Detector | Recognizer |
|---|---|---|
| general | `PP-OCRv6_medium_det` | `PP-OCRv6_medium_rec` |
| Thai | `PP-OCRv6_medium_det` | `th_PP-OCRv5_mobile_rec` |

The final dev-only ablation compared original, grayscale, contrast, denoise,
sharpen, background normalization, quality-auto, and optional Paddle
orientation/unwarping modules. `original` retained the best selection score;
the locked test and private set were not used. Raster-to-PDF DPI ties resolved
to 200 DPI for lower cost.

Auto routing uses trusted metadata/hints, Unicode evidence, valid-character
ratio, confidence, coverage, and duplicate/garbage penalties. Weak general
output triggers a Thai comparison. Output records the exact detector,
recognizer, preprocessing version, route, candidate scores, selected
orientation, and fine-deskew evidence.

Public OCR caching binds source bytes, model hashes, route, preprocessing
profile/version, and transform settings. Private inference bypasses public
caches.

## Public data and leakage controls

Adapters normalize source annotations into stable page/document IDs, source
provenance, dimensions, tokens, polygons, boxes, entities, relations,
canonical fields, document type, language, split, and privacy status.

- SROIE supplies receipt OCR quadrilaterals and canonical fields.
- FUNSD supplies words, entity labels, and linked entity pairs.
- FATURA supplies Turkish invoice text boxes and fields.
- CORU supplies supported full-document KIE/QA records; malformed, empty,
  line-crop, and text-only components are excluded with reasons.

The authoritative normalized public population is 12,433 pages. Leakage-safe
identity grouping assigns FATURA/SROIE/FUNSD to train, `dev_select`,
`dev_calibration`, and `test_in_domain`; all 1,261 CORU pages remain
`unseen_domain_test`. The final model build contains 11,684 examples: 11,172
ground-truth examples plus 256 PaddleOCR and 256 hybrid variants. Gmail rows
are structurally ineligible for model data, training, calibration, selection,
or public evaluation.

## Final multi-task model

`MultiTaskTextLayoutModel` starts from `microsoft/layoutxlm-base` multilingual
token and normalized 2D-layout embeddings. It intentionally omits the visual
Detectron2 backbone, which has no compatible verified Windows runtime here.
The shared encoder feeds four learned heads:

- BIO entity labels;
- document type;
- canonical-field evidence labels;
- geometry-aware typed entity-pair relations with real positives and hard
  negatives.

Inference merges overlapping windows, applies learned calibrated thresholds,
resolves conflicting evidence by abstaining, then combines learned output with
deterministic evidence-backed field validation, arithmetic checks, generic
key/value fallback, and structured table output.

The final public run trained four epochs using mixed precision, batch size 1,
gradient accumulation 4, gradient clipping, and a 0.7 upright/0.3 fixed-37°
checkpoint-selection score. It completed 7,812 optimizer steps over 7,782
training examples. The checkpoint and tokenizer reload exactly (maximum logit
difference 0.0). The source and derived checkpoint are
CC-BY-NC-SA-4.0 and are not committed to Git.

Calibration uses only `dev_calibration`. Temperature scaling and abstention
thresholds are bound to the exact checkpoint, final build ID, and manifest
hash. Every final evaluator checks that binding before reading a held-out row.

## Output contract

`schemas/inference_output.schema.json` requires document/source metadata,
document/language decisions, nullable display-only rotation output,
page-specific OCR/layout geometry, entities, typed relations, tables,
canonical fields, validation status, extraction source, warnings, transforms,
and processing provenance. Unsupported or conflicted canonical values are
explicitly `null`; every emitted value carries evidence and confidence.

Schema validation precedes atomic write. Private results must target the
configured ignored D: private root.

## Evaluation boundaries

The protocol separates evidence so no result is overstated:

- `dev_select`: bounded hyperparameter and OCR-profile selection only;
- `dev_calibration`: temperatures and abstention thresholds only;
- `test_in_domain`: one locked ground-truth-token evaluation plus fixed-angle
  layout and bounded real-OCR grids; never used for tuning;
- `unseen_domain_test`: deterministic CORU QA/OCR coverage only;
- private Gmail: local operational aggregates only, with no accuracy claim;
- synthetic integration: executable image, rotation, Thai, multipage, schema,
  and hash-binding proof.

The reference-token layout scores isolate the learned heads. End-to-end scores
include OCR errors and therefore measure the usable pipeline. These two levels
must be reported separately.

## Failure boundaries

The workflow fails closed for raw-data drift, storage-reserve violations,
missing or hash-mismatched models, invalid manifests/build bindings,
unsupported/corrupt/encrypted/oversized inputs, invalid geometry or schema,
missing/incompatible final checkpoints, worker protocol failure, private-path
violations, or detected publication candidates. K-Means failure is the sole
deliberate exception because it is non-controlling and becomes a warning.
