# Evaluation Protocol and Results

## Commands

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/evaluate_rule_baselines.py
& $ocr scripts/evaluate_information_extraction.py --profile smoke --device gpu:0
& $ocr scripts/evaluate_private_gmail.py --limit 2 --device gpu:0
& $ocr scripts/run_integration_smoke.py --device gpu:0
python scripts/verify_information_extraction.py --complete
```

## Public smoke scope

The executed profile uses one usable page from each of CORU, FATURA, FUNSD,
and SROIE at input angles 0, 45, 90, and 270 degrees: 16 runs total. This is a
bounded integration benchmark, not a final dataset benchmark.

| Metric | Result |
|---|---:|
| Successful/failed runs | 16 / 0 |
| OCR reference coverage | 12 / 16 |
| CER / WER on referenced runs | 0.7497 / 0.8856 |
| Recognized-text coverage on referenced runs | 0.2503 |
| Text-detection reference coverage | 12 / 16 |
| Text-detection precision / recall / F1 at IoU 0.5 | 0.5483 / 0.3333 / 0.4146 |
| Empty OCR output rate | 0.0 |
| Entity precision / recall / F1 | 0.0062 / 0.0217 / 0.0096 |
| Relation precision / recall / F1 | 0 / 0 / 0 |
| Canonical field accuracy | 0.05 |
| Document type accuracy | 0.25 |
| Exact cardinal orientation accuracy | 0.375 |
| Mean orientation error | 45 degrees |
| Rotation-retention entity-F1 ratio | 0.9856 |

CER/WER exclude rows without any source reference token text and expose
coverage explicitly. This avoids turning an unavailable reference into a
one-character denominator. CER/WER may exceed 1 when insertions exceed the
reference length. Recognized-text coverage is the nonnegative complement of
CER. Detection metrics greedily match convex reference/predicted polygons
one-to-one at IoU 0.5 and are unavailable when a page has no token geometry.

The similar upright and rotated F1 values are both extremely low. Rotation
retention therefore does not imply useful absolute extraction quality.

## Language and unseen scope

The public set reports English and Turkish-compatible general OCR. Thai and
mixed English/Thai routing are verified using synthetic OCR and multipage
integration fixtures, not a labeled public Thai benchmark.

`run_integration_smoke.py` writes only synthetic fixtures and full outputs
below the ignored D: external root. Its committable report contains counts,
assertions, and SHA-256 evidence, not OCR text. Complete verification re-hashes
the runner, core pipeline sources, config, schema, model setup, checkpoint,
fixtures, and outputs, revalidates every output schema, and independently
checks the image, 45-degree, general-to-Thai two-page PDF, Thai Unicode, exact
recognizer, unknown-type, and display-only K-Means conditions.

CORU had zero fit rows and is reported as a natural unseen dataset. Across its
four angle runs, OCR was nonempty every time, with means of 29.25 words, 24.25
entities, and 6.75 key-value pairs. Its selected annotations have no usable OCR
reference geometry, so OCR error is unavailable. This is not a separately
retrained leave-one-dataset-out experiment.

## Private scope

The private operational check processed 2/2 local Gmail pages. It reported
mean OCR words 89.0, mean entities 67.5, mean relations 13.0, and mean OCR
confidence 0.9528. These are operational aggregates only; no private ground
truth exists, so no accuracy is claimed.

## Required next research evaluation

Before presenting a final model, prepare a larger public aligned dataset,
train/select a checkpoint without private data, establish labeled Thai data,
and run professor-approved dataset/document holdouts with predetermined quality
thresholds. Do not tune from Gmail aggregates.
