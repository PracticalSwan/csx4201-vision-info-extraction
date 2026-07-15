# vision-info-extraction

CSX4201 document information-extraction pre-model with rotation-robust OCR.
The repository now contains two deliberately separate systems:

1. The main pre-model accepts images and PDFs, selects an OCR orientation,
   routes general or Thai recognition, extracts layout entities and key-value
   relations, normalizes canonical fields, and writes schema-validated JSON.
2. The preserved K-Means rotation baseline reports a quadrant for display
   only. It never controls OCR, correction, extraction, or success.

The implementation and smoke lifecycle are verified. Model quality is not
final: the trained LayoutXLM smoke checkpoint used three public training
examples and is proof of training/checkpoint integration, not a production
model.

## Verified status (2026-07-15)

| Area | Current evidence |
|---|---|
| Existing rotation baseline | 8,332 rotations; 20/20 verifier checks; public held-out zone accuracy about 38% |
| Exact-angle experiment | About 90-degree median error and 0% reliable estimates; disabled for inference by default |
| Public annotation normalization | 12,433 normalized pages from SROIE, FUNSD, FATURA, and CORU; Gmail fit rows 0 |
| OCR models | `PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`, and `th_PP-OCRv5_mobile_rec`; hash and GPU smoke verification pass, including exact phrase recovery from a 90-degree fixture |
| Layout model | Detectron2-free LayoutXLM text + normalized 2D layout encoder; smoke trained on CUDA; checkpoint and relation-head reload differences 0.0 |
| Public smoke evaluation | 16/16 successful runs across four datasets and 0/45/90/270-degree inputs; text-detection P/R/F1 0.5483/0.3333/0.4146, entity F1 0.0096, relation F1 0.0 |
| Private operational test | 2/2 pages completed locally; aggregate-only report; no private filenames, OCR text, images, or per-document output |
| Tests | 158 passed, 1 skipped in the development runtime; 53 OCR-runtime tests and 3 CUDA-layout tests pass |

The public smoke metrics are intentionally reported even though they are poor.
They show that the end-to-end path works, not that the model is accurate.

## Architecture

```text
image or PDF
  -> EXIF/color/PDF page normalization
  -> independent OCR candidates (0, 90, 180, 270; optional supplied deskew)
  -> PP-OCRv6 detector
  -> general or Thai recognizer selected from metadata and OCR evidence
  -> LayoutXLM text + 2D-layout entity worker
  -> geometry-aware key-value relations and evidence-backed field rules
  -> JSON Schema validation and atomic output

same page -> preserved K-Means quadrant -> display field only
```

Paddle and CUDA PyTorch load conflicting cuDNN DLLs on this Windows host, so
they run in separate Python 3.10 processes. The OCR process uses Paddle GPU and
CPU-only PyTorch (required transitively by PaddleX). The layout worker uses
PyTorch 2.8.0 + CUDA 12.8. Both environments, caches, datasets, checkpoints,
and generated outputs live under `D:\CSX4201\vision-info-extraction-assets`.

The layout implementation uses the official `microsoft/layoutxlm-base`
embeddings and encoder with multilingual text and normalized bounding boxes.
It intentionally omits the visual Detectron2 backbone because a compatible
Windows wheel is unavailable. The source checkpoint is
CC-BY-NC-SA-4.0; review that noncommercial license before redistribution or
commercial use.

## Setup

Requirements: Windows, Python 3.10, D: with at least 15 GiB free, and C: with
at least 15 GiB free. GPU mode was verified on an NVIDIA GeForce RTX 5050
Laptop GPU (8,151 MiB).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ie_environment.ps1
& 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe' scripts/download_ocr_models.py
& 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe' scripts/verify_ocr_models.py --device gpu:0
```

The setup script checks storage first and configures `PADDLE_PDX_CACHE_HOME`,
`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`,
`PIP_CACHE_DIR`, `TMP`, and `TEMP` on D:. Details are in
[docs/ocr_setup.md](docs/ocr_setup.md).

## Inference

Run the CLI with the OCR environment:

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/extract_document.py `
  --input 'path\to\document.pdf' `
  --output 'D:\CSX4201\vision-info-extraction-assets\generated\run-001' `
  --language auto --device gpu:0
```

Useful options:

- `--language auto|general|thai|en|tr|th`
- `--language-hint th` when trusted metadata identifies Thai
- `--deskew-angle <degrees>` to add an evidence-backed fine-angle candidate
- `--disable-kmeans-display` without changing extraction
- `--private-output` to require an ignored private destination
- `--max-pages`, `--continue-on-page-error`, and `--dry-run`

Outputs validate against
[schemas/inference_output.schema.json](schemas/inference_output.schema.json).
Every canonical value includes evidence and confidence; unsupported fields
remain `null`. Unknown document types retain OCR, layout entities, and generic
`key: value` relations.

## Data and training

Raw inputs are read-only. Normalized annotations are derived from public data
without modifying source annotations:

```powershell
& $ocr scripts/normalize_ie_annotations.py --force
& $ocr scripts/verify_ie_annotations.py
& $ocr scripts/prepare_model_dataset.py --profile smoke --device gpu:0 --force
```

The executed smoke model dataset has 9 usable examples: FATURA 4, FUNSD 2,
and SROIE 3; splits are 3 train, 2 validation, and 4 test. Seven candidates
were excluded for missing source token geometry or insufficient OCR alignment.
No CORU or Gmail row entered training.

```powershell
$layout = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
& $layout scripts/train_layout_model.py --profile smoke --device cuda
```

The smoke run completed two micro-steps and one optimizer step. Validation loss
was 2.3490 and token accuracy 0.1007 over 139 tokens. These values are not a
quality claim. Full-corpus dataset preparation and final training were not run;
they require a separately bounded capacity plan and an agreed official
evaluation protocol.

## Evaluation and verification

```powershell
& $ocr scripts/evaluate_rule_baselines.py
& $ocr scripts/evaluate_information_extraction.py --profile smoke --device gpu:0
& $ocr scripts/evaluate_private_gmail.py --limit 2 --device gpu:0
& $ocr scripts/run_integration_smoke.py --device gpu:0
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/verify_data.py
python scripts/verify_rotation_data.py --profile full --complete
python scripts/verify_information_extraction.py --complete
```

The public smoke evaluation uses one page from each public dataset and four
input angles. It is bounded and not statistically representative. CER/WER are
computed only where reference OCR tokens exist (12/16 runs); CORU has no usable
OCR reference in this protocol. On the same 12 referenced runs, recognized-text
coverage is 0.2503 and text-detection precision/recall/F1 is
0.5483/0.3333/0.4146 at polygon IoU 0.5. CORU nevertheless provides a natural unseen
dataset check because it contributed zero fit rows: 4/4 runs returned nonempty
OCR, with mean 29.25 OCR words, 24.25 predicted entities, and 6.75 key-value
pairs.

The synthetic image/45-degree/Thai/two-page-PDF integration report is produced
by a tracked runner rather than edited by hand. Complete verification re-hashes
the runner, core pipeline sources, config, schema, exact model setup, 1.1 GB
layout checkpoint, relation head, fixtures, and schema-valid outputs before
independently checking their semantics.

## Privacy

`data/raw/private/gmail/` contains real financial and legal documents. Never
commit or upload it. Gmail is private-test only and must never influence OCR
routing thresholds, model fitting, checkpoint selection, or hyperparameters.
Public reports may contain only aggregate private counts. See
[docs/privacy.md](docs/privacy.md).

Large and private assets are ignored. Before any publish action, inspect
`git status --ignored`, the staged diff, file sizes, secret matches, and the
repository visibility.

## Preserved rotation baseline

The original bounded full-angle experiment remains intact: 400 public pages
plus 203 private pages, 8,332 generated rotations, 1,957-value handcrafted
features, train-only StandardScaler/PCA/K-Means, and a training-only Hungarian
cluster-to-zone mapping. Mapped validation/test accuracy is about 38%. The
exact-angle experiment has 0% reliable estimates and is not used by the main
pipeline.

The OCR runtime is Python 3.10 and therefore uses scikit-learn 1.7.2, while the
preserved K-Means artifacts were serialized with 1.8.0 (Python 3.14). A live
compatibility check produced the same cluster and zone and a confidence delta
below 1e-10 on the checked public sample. The wrapper remains failure-isolated;
any load or prediction error returns a display warning and cannot stop OCR.

To reproduce the preserved bounded rotation baseline in dependency order:

```powershell
python scripts/prepare_page_images.py
python scripts/create_rotation_splits.py
python scripts/generate_rotation_data.py --profile smoke
python scripts/verify_rotation_data.py --profile smoke
python scripts/generate_rotation_data.py --profile full
python scripts/verify_rotation_data.py --profile full
python scripts/extract_rotation_features.py --profile full
python scripts/fit_rotation_preprocessing.py
python scripts/train_kmeans_rotation.py
python scripts/evaluate_kmeans_rotation.py
python scripts/evaluate_angle_estimation.py --profile full
python scripts/run_rotation_experiment.py --profile full
```

Here `full` means the complete configured angle profile over the documented
bounded selection, not the full 22,086-page public corpus.

## Known limitations

- The LayoutXLM checkpoint is a smoke checkpoint, not a final trained model.
- Public smoke text-detection F1 is 0.4146, entity F1 is 0.0096, relation F1 is
  0.0, and field accuracy is 0.05; do not describe the system as accurate.
- The default orientation set is cardinal. Arbitrary-angle inputs complete,
  but the 45-degree exact-orientation score was 0% in the bounded smoke run.
  A supplied deskew candidate is supported; automatic fine-angle correction is
  disabled because the preserved estimator is unreliable.
- Thai routing and mixed English/Thai multipage inference are verified on
  synthetic smoke fixtures, not a labeled public Thai benchmark.
- The natural CORU holdout demonstrates generic output only; it lacks usable
  OCR reference tokens and is not a separately retrained leave-one-out study.
- LayoutXLM visual features are omitted, and the checkpoint license is
  noncommercial.
- Official target fields, quality thresholds, and final delivery protocol
  still require professor confirmation.

More detail: [requirements](docs/requirements.md),
[design](docs/design.md), [information extraction](docs/information_extraction.md),
and [evaluation](docs/evaluation.md).
