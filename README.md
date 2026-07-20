# vision-info-extraction

CSX4201 final working vision information-extraction pre-model for images and
PDFs, including arbitrary rotation handling, general/Thai OCR routing,
layout-aware entities, typed relations, canonical fields, tables, calibrated
abstention, and schema-validated output.

The repository contains two deliberately separate systems:

1. The controlling pre-model selects OCR preprocessing, orientation, and
   language route, then runs the calibrated multi-task layout model and
   evidence validators.
2. The preserved four-cluster K-Means rotation baseline produces a quadrant
   for display/diagnostics only. It never controls OCR or extraction.

The implementation, final public training lifecycle, held-out evaluation, and
local operational checks are complete. This remains an academic pre-model,
not a production or high-stakes decision system: reference-token layout scores
are strong, but real OCR is the main end-to-end bottleneck and relation quality
is limited by sparse supervision.

## One-step local use

The July 19 Build Week extension adds a local GUI, one-command CLI, bundled
model release, Windows setup, Docker-backed macOS path, and an optional
consent-gated GPT-5.6 review skill. Extraction itself is fully local and uses
no OpenAI API key. The GUI previews uploaded images and the first page of
PDFs, uses one loading indicator, clears stale results when the input changes,
and keeps overflowing OCR text and run logs independently scrollable.

Current development machine:

```powershell
python run_ocr.py "path\to\document.pdf"
python app.py
```

Shareable release:

```text
D:\OCR_Model
D:\OCR_Model.zip
SHA-256: f6a057e5c37c6036bd1d4ad6c247aa0895e893d87fe17f997fd011e0c5064f9e
Size: 1,153,302,135 bytes
Private Release: https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week
```

On Windows, recipients run `setup_windows.bat` once and then
`launch_windows.bat` or `run_cli.bat <document>`. On macOS, the supported path
is `bash launch_macos.command` through Docker Desktop; no physical Mac test is
claimed. The release includes the exact final LayoutXLM and three PaddleOCR
weight sets, a synthetic sample, `MODEL_MANIFEST.json`, and no raw/private
data. The display-only rotation branch uses a hash-bound, version-neutral
numeric export rather than loading scikit-learn 1.8 pickles in the Python 3.10
runtime. See [portable usage](docs/PORTABLE_USAGE.md).

The optional `$review-ocr-document` repo skill uses the local `ocr_model` STDIO
MCP server to expose opaque result IDs and only user-selected fields after
confirmation. OCR text requires separate consent and is capped. GPT-5.6
returns suggestions without changing the local JSON. It uses the user's
signed-in Codex session, not an API key. See
[Codex integration](docs/CODEX_INTEGRATION.md).

Build Week submission materials are under [docs/devpost](docs/devpost).
Sanitized demo evidence includes the [ready GUI](docs/devpost/assets/ocr-model-ready.png),
[single loading indicator](docs/devpost/assets/ocr-model-loading-single.png),
[scrollable OCR text](docs/devpost/assets/ocr-model-scrollable-ocr.png),
[PDF preview](docs/devpost/assets/ocr-model-pdf-preview.png),
[completed field extraction](docs/devpost/assets/ocr-model-fields-final.png),
and [visual overlay](docs/devpost/assets/ocr-model-visual-tab-final.png).

## Verified status (2026-07-20)

| Surface | Executed evidence |
|---|---|
| Public annotations | 12,433 normalized pages from FATURA, SROIE, FUNSD, and CORU; 29,886 split identities checked with zero leakage; Gmail fit rows 0 |
| Final model build | `final-6be3e0b46b0a4e4c`; 11,684 examples (11,172 ground truth, 256 PaddleOCR, 256 hybrid); 7,782 train examples |
| Training | 4 epochs, 31,240 microsteps, 7,812 optimizer steps; epoch 4 selected with upright/37° score 0.824160; exact reload difference 0.0 |
| Checkpoint | Local D: checkpoint; `model.safetensors` SHA-256 `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189` |
| Calibration | 708 public `dev_calibration` examples; exact build/checkpoint/manifest binding; private/Gmail rows 0 |
| Locked in-domain test | 1,760 public ground-truth examples; calibrated entity F1 0.9813, canonical-evidence F1 0.9814, relation F1 0.4632; document selective accuracy 1.0 at 97.56% coverage |
| Layout angle grid | 18 required angles over 30 balanced pages; minimum calibrated entity F1 0.7491 (95.30% upright retention), canonical F1 0.9360, relation F1 0.3434 |
| End-to-end angle grid | 72/72 public/synthetic cases nonempty; bounded public OCR coverage 0.4026–0.4368 and entity F1 0.1314–0.1830; synthetic Thai text recovery 18/18 |
| Unseen CORU | 100/100 pages succeeded; 78.53% of 4,001 QA answer strings found in OCR; 15.68% canonical exact match; never used for fitting or selection |
| Private operation | 26/26 anonymous local documents and 203 pages succeeded; public report is aggregate-only and declares no filenames, OCR text, images, or per-document predictions |
| Verification | IE verifier 46/46 complete checks; exact OCR model/GPU checks and hash-bound image/rotation/Thai/multipage integration pass |
| Automated tests | Host suite: 243 passed, 2 environment-dependent skips; OCR-runtime partition: 122 passed; CUDA-layout partition: 2 passed |
| Portable release | Windows bundle doctor and exact OCR-model verification passed; native GPU GUI/CLI and `linux/amd64` Docker CPU extractions produced identical field values, OCR text, entity triplets, and relation triplets; version-neutral K-Means parameters matched all 7,520 public feature rows with zero cluster-label differences |
| Optional GPT-5.6 bridge | Real STDIO MCP client listed both tools; unconfirmed calls exposed no values; confirmed test exposed only the selected synthetic `total_amount` field |

Full reports are under [reports/final_model](reports/final_model), including
the [model card](reports/final_model/final_model_card.md) and
[error analysis](reports/final_model/error_analysis.md).

## Architecture

```text
image / PDF / multipage document
  -> EXIF, color, transparency, PDF-page normalization
  -> selected public preprocessing profile
  -> 0/90/180/270 OCR candidates + reliable automatic fine deskew
  -> PP-OCRv6 detector
  -> general PP-OCRv6 or Thai PP-OCRv5 recognizer
  -> persistent CUDA multi-task LayoutXLM worker
       entity | document type | canonical evidence | typed relation heads
  -> calibrated abstention + evidence conflicts/arithmetic checks
  -> generic key/value fallback + geometry tables
  -> JSON Schema validation + atomic JSON/optional visualization

same page -> preserved PCA/K-Means quadrant -> display field only
```

Paddle GPU and CUDA PyTorch load conflicting Windows cuDNN DLLs, so the OCR
and layout stages use separate Python 3.10 environments. Large assets live at:

```text
D:\CSX4201\vision-info-extraction-assets
```

- `environments\ie-ocr`: PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0,
  PaddleX 3.7.2, and CPU-only PyTorch required by PaddleX;
- `environments\ie-layout`: PyTorch 2.8.0+cu128, Transformers 4.57.6,
  SentencePiece, and Accelerate.

The layout encoder starts from `microsoft/layoutxlm-base` text and normalized
2D-layout weights. The Detectron2 visual backbone is omitted because no
compatible verified Windows runtime is available. The inherited checkpoint
license is CC-BY-NC-SA-4.0.

## Setup

Requirements: Windows, Python 3.10, C: and D: with at least 15 GiB free, and a
CUDA-capable NVIDIA GPU for the verified GPU path. The executed host used an
RTX 5050 Laptop GPU with 8,151 MiB.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ie_environment.ps1
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/download_ocr_models.py
& $ocr scripts/verify_ocr_models.py --device gpu:0
```

The setup redirects Paddle, Hugging Face, Torch, pip, and temporary caches to
D:. Use limits as sparingly as possible to avoid running out of quota: verify
a bounded profile and storage projection before expensive materialization or
training. See [docs/ocr_setup.md](docs/ocr_setup.md).

## Run the final pre-model

The executed local checkpoint is:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final
```

Run the CLI from the OCR environment:

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'
& $ocr scripts/extract_document.py `
  --input 'path\to\document.pdf' `
  --output 'D:\CSX4201\vision-info-extraction-assets\generated\run-001' `
  --language auto --device gpu:0 --model-checkpoint $checkpoint
```

Useful options:

- `--language auto|general|thai|en|tr|th` and trusted `--language-hint th`;
- `--deskew-angle <degrees>` to add externally supported evidence;
- `--confidence-threshold 0..1` to apply a stricter output floor;
- `--save-visualization` for non-private review;
- `--disable-kmeans-display` without changing OCR/extraction;
- `--max-pages`, `--continue-on-page-error`,
  `--continue-on-document-error`, and `--dry-run`;
- configured Gmail inputs are rejected unless `--private-output` is supplied,
  and that mode accepts only an ignored private destination.

Outputs validate against
[schemas/inference_output.schema.json](schemas/inference_output.schema.json).
Unsupported or conflicting canonical fields remain `null`; emitted values
include confidence, method, validation status, extraction source, and evidence.

## Reproduce final data, training, and calibration

Raw inputs are read-only. Derived model data and weights remain on D:.

```powershell
& $ocr scripts/normalize_ie_annotations.py --force
& $ocr scripts/verify_ie_annotations.py
& $ocr scripts/prepare_model_dataset.py `
  --profile final --device gpu:0 --force `
  --streams ground_truth paddleocr hybrid --ocr-variant-limit 256

$layout = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'
& $layout scripts/train_multitask_model.py `
  --profile final --checkpoint $checkpoint --device cuda `
  --streams ground_truth paddleocr hybrid --upright-probability 0.6
& $layout scripts/calibrate_multitask_model.py `
  --profile final --checkpoint $checkpoint --device cuda `
  --streams ground_truth paddleocr
```

The 1.1 GB final weight and resumable optimizer state are intentionally not in
Git. Reproduce them with the commands above, retain the verified local D:
checkpoint, or use the generated `D:\OCR_Model` release. The shareable release
contains inference weights only, not resumable optimizer state. The repository
commits code, manifests, calibration, metrics, hashes, and provenance, but no
private data or large model/cache artifacts.

## Reproduce evaluation and verification

```powershell
& $layout scripts/evaluate_multitask_model.py `
  --profile final --checkpoint $checkpoint --split test_in_domain `
  --streams ground_truth --device cuda --group-by dataset language `
  --calibration models\multitask_calibration.json `
  --report-name final_test_in_domain_ground_truth.json
& $layout scripts/evaluate_layout_angles.py `
  --checkpoint $checkpoint --device cuda --pages-per-dataset 10
& $ocr scripts/run_ocr_preprocessing_ablation.py --device gpu:0 --limit-per-dataset 1
& $ocr scripts/evaluate_end_to_end_angles.py `
  --checkpoint $checkpoint --device gpu:0 --pages-per-dataset 1
& $ocr scripts/evaluate_unseen_coru.py `
  --checkpoint $checkpoint --device gpu:0 --limit 100
& $ocr scripts/run_integration_smoke.py --device gpu:0
python scripts/compile_final_reports.py
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/verify_data.py
python scripts/verify_rotation_data.py --profile full --complete --portable
python scripts/verify_information_extraction.py --complete
```

The locked `test_in_domain` result is not used for model, threshold,
preprocessing, or hyperparameter selection. Reference-token scores isolate the
layout heads; end-to-end scores include OCR errors. Do not substitute one for
the other. See [docs/evaluation.md](docs/evaluation.md).

## Privacy and publication

`data/raw/private/gmail/` contains real financial and legal documents. Never
commit, upload, cache publicly, or send them to an external service. They may
only run locally after the model is fixed, and they cannot influence fitting,
calibration, selection, thresholds, or rules. Detailed results and manual
review stay under the ignored D: private root. See
[docs/private_testing.md](docs/private_testing.md) and
[docs/privacy.md](docs/privacy.md).

Before every push, confirm repository visibility, inspect ignored/staged paths
and sizes, scan for secrets/private names/content, and recheck all zero-private
provenance counters.

## Preserved rotation baseline

The historical bounded experiment remains reproducible: 603 pages, 8,332
rotations, 1,957-value handcrafted features, train-only StandardScaler/PCA,
four-cluster K-Means, and a training-only Hungarian cluster-to-zone mapping.
Mapped public held-out zone accuracy is about 38%. Exact-angle median error is
about 90° with zero reliable estimates at the configured threshold.

Run the preserved rotation workflow stage by stage, or use the final command
as the orchestrated equivalent:

```powershell
python scripts/prepare_page_images.py
python scripts/create_rotation_splits.py
python scripts/generate_rotation_data.py --profile full
python scripts/verify_rotation_data.py --profile full
python scripts/extract_rotation_features.py --profile full
python scripts/fit_rotation_preprocessing.py
python scripts/train_kmeans_rotation.py
python scripts/evaluate_kmeans_rotation.py
python scripts/evaluate_angle_estimation.py --profile full
```

`python scripts/run_rotation_experiment.py --profile full` runs that preserved
training/evaluation sequence. For the final portable release, export and check
the version-neutral display parameters after the model run:

```powershell
python scripts/export_rotation_inference_params.py --verify-feature-root data/processed/features/full/888fb4999c985ba0
python scripts/verify_rotation_data.py --profile full --complete --portable
```

The feature-cache suffix is the live `configuration_hash` in
`models/kmeans_rotation/feature_config.json`; use that value if the feature
configuration is intentionally changed.

These weak results are retained honestly. K-Means produces only
`rotation_display`; the failed exact-angle estimator is disabled. The OCR path
uses evidence-scored candidates and fine deskew independently.

## Known limitations

- End-to-end OCR is the main bottleneck. On the three-page fixed angle sample,
  text coverage is 0.4026–0.4368 and entity F1 is 0.1314–0.1830 even though
  reference-token held-out entity F1 is 0.9813.
- Relation supervision is confined to FUNSD. Locked relation F1 is about
  0.46, while the bounded real-OCR relation score is near zero.
- FUNSD headers/questions/answers are the weakest supported entity classes;
  the public datasets are highly imbalanced.
- Synthetic Thai text/routing passes every required angle, but no compatible
  labeled public Thai benchmark is available.
- Layout visual features are omitted on this Windows runtime.
- Native macOS execution was not tested because no physical Mac was available;
  the documented macOS route is the validated CPU-only `linux/amd64` Docker
  package and may be slow under Apple Silicon emulation.
- CORU has QA answer text but no compatible token polygons; its 78.53% answer
  recall is not entity/relation F1.
- Canonical fields, document types, official thresholds, and deliverable
  protocol still require professor confirmation.
- The CC-BY-NC-SA-4.0 checkpoint is noncommercial; review redistribution terms
  before sharing weights.

Further detail: [requirements](docs/requirements.md),
[design](docs/design.md),
[workflow](docs/information_extraction.md), and
[task status](docs/tasks.md).
