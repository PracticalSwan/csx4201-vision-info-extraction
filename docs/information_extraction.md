# Information-Extraction Workflow

## 1. Normalize and verify public annotations

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/normalize_ie_annotations.py --force
& $ocr scripts/verify_ie_annotations.py
```

The normalizer writes supported public records below the ignored
`data/processed/normalized_ie_annotations` tree and emits a public-safe
manifest, schema, mapping report, summary, and categorized error CSV. It never
modifies raw data. Eligibility is decided before materialization, so excluded
records do not create orphan outputs.

## 2. Build the leakage-safe final model dataset

```powershell
& $ocr scripts/prepare_model_dataset.py `
  --profile final --device gpu:0 --force `
  --streams ground_truth paddleocr hybrid --ocr-variant-limit 256
& $ocr scripts/report_final_dataset.py
```

The split builder groups document and duplicate identities before assigning
train, `dev_select`, `dev_calibration`, and `test_in_domain`. CORU is reserved
as `unseen_domain_test`. The final manifest is profile/build-bound and records
source, stream, OCR model hashes, labels, split, and privacy status.

Executed final build `final-6be3e0b46b0a4e4c` contains 11,684 examples:

| Split | Examples |
|---|---:|
| train | 7,782 |
| dev_select | 1,243 |
| dev_calibration | 763 |
| test_in_domain | 1,896 |

The streams are 11,172 ground-truth, 256 PaddleOCR, and 256 hybrid examples.
Training targets include 514,220 entity tokens, 152,875 canonical-evidence
tokens, 40,954 relation pairs, and 4,545 positive relations. Gmail fit rows
are zero.

## 3. Train the final multi-task checkpoint

```powershell
$layout = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'
& $layout scripts/train_multitask_model.py `
  --profile final --checkpoint $checkpoint --device cuda `
  --streams ground_truth paddleocr hybrid `
  --upright-probability 0.6
& $layout scripts/report_multitask_training.py
```

The executed run completed four epochs, 31,240 microsteps, and 7,812 optimizer
steps. Checkpoint selection combines upright dev-select quality (0.7) with a
fixed 37° robustness slice (0.3); epoch 4 scored 0.824160 and was selected.
Reloaded logits match exactly.

Current local checkpoint:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final
```

`model.safetensors` SHA-256:

```text
34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189
```

The 1.1 GB weight file and resumable optimizer state remain on D: and are not
committed. The source/derived license is CC-BY-NC-SA-4.0.

## 4. Calibrate without touching test or private data

```powershell
& $layout scripts/calibrate_multitask_model.py `
  --profile final --checkpoint `
  'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final' `
  --device cuda --streams ground_truth paddleocr
```

Calibration uses 708 public `dev_calibration` examples. It writes
`models/multitask_calibration.json` with temperatures and thresholds bound to
the exact build, manifest, and checkpoint hashes. It records zero private and
zero Gmail rows.

## 5. Run inference

```powershell
& $ocr scripts/extract_document.py `
  --input 'path\to\document.pdf' `
  --output 'D:\CSX4201\vision-info-extraction-assets\generated\run-001' `
  --language auto --device gpu:0 `
  --model-checkpoint `
  'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'
```

The OCR process selects orientation, preprocessing, and general/Thai route. A
persistent CUDA layout worker produces calibrated entities, document type,
canonical evidence, and typed relations. Evidence rules validate fields,
arithmetic, generic key/value pairs, and tables. Results are schema-validated
before atomic write; page errors can be isolated with
`--continue-on-page-error`.

Use `--save-visualization` for public/debug inputs only. For private inputs,
follow [private_testing.md](private_testing.md); detailed private outputs must
remain under the ignored D: root.

## 6. Reproduce final evidence

```powershell
& $layout scripts/evaluate_multitask_model.py `
  --profile final --checkpoint `
  'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final' `
  --split test_in_domain --streams ground_truth --device cuda `
  --group-by dataset language --calibration models\multitask_calibration.json `
  --report-name final_test_in_domain_ground_truth.json
& $layout scripts/evaluate_layout_angles.py `
  --checkpoint 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final' `
  --device cuda --pages-per-dataset 10
& $ocr scripts/run_ocr_preprocessing_ablation.py --device gpu:0 --limit-per-dataset 1
& $ocr scripts/evaluate_end_to_end_angles.py `
  --checkpoint 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final' `
  --device gpu:0 --pages-per-dataset 1
& $ocr scripts/evaluate_unseen_coru.py `
  --checkpoint 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final' `
  --device gpu:0 --limit 100
& $ocr scripts/run_integration_smoke.py --device gpu:0
python scripts/compile_final_reports.py
python scripts/verify_information_extraction.py --complete
```

The locked test is executed once and is not used for calibration, profile
selection, thresholds, or checkpoint selection. See
[evaluation.md](evaluation.md) for results and limitations.
