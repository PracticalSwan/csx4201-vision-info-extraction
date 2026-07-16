# Final Evaluation Protocol and Results

## Split discipline

| Split | Allowed use |
|---|---|
| train | Fit model heads and encoder layers. |
| dev_select | Hyperparameters, checkpoint selection, and bounded OCR-profile selection. |
| dev_calibration | Temperature scaling and abstention thresholds only. |
| test_in_domain | One locked reference-token test plus predetermined robustness grids; never tune from it. |
| unseen_domain_test | CORU QA/OCR coverage after all choices are fixed. |
| private Gmail | Local aggregate operation only; never training, selection, thresholding, or accuracy. |

The final manifest build ID, manifest SHA-256, checkpoint SHA-256, and
calibration SHA-256 are validated before evaluation. Every public evaluator
rejects private/unmarked examples.

## Commands

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
$layout = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'

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
python scripts/verify_information_extraction.py --complete
```

Private testing has a separate command and output boundary in
[private_testing.md](private_testing.md).

## Locked in-domain layout results

The ground-truth-token test covers 1,760 examples (FATURA 1,584, SROIE 146,
FUNSD 30) and 1,761 windows.

| Metric | Raw | Calibrated/abstained |
|---|---:|---:|
| Entity micro-F1 | 0.9807 | 0.9813 |
| Entity macro-F1 | 0.7290 | 0.6940 |
| Canonical-evidence micro-F1 | 0.9792 | 0.9814 |
| Canonical-evidence macro-F1 | 0.9749 | 0.9759 |
| Relation micro-F1 | 0.4668 | 0.4632 |
| Relation macro-F1 | 0.5620 | 0.5741 |
| Composite score | 0.8538 | 0.8537 |

Document accuracy is 1.0; calibrated document coverage is 0.9756 with 1.0
selective accuracy. Calibration improves canonical precision and overall
entity micro-F1 but abstains on rare entity classes, reducing entity macro-F1.
The final report therefore retains both raw and abstained values.

Dataset interpretation:

- FATURA dominates the test and has strong entity/canonical evidence but no
  relation supervision;
- SROIE has entity F1 0.8681 and canonical-evidence F1 0.8254;
- FUNSD has entity F1 0.7454 and supplies the relation score (0.4668);
- B-HEADER F1 is 0.1277 and QUESTION_ANSWER relation F1 is 0.3874.

## Layout-only angle robustness

Thirty balanced public test pages are evaluated at:

```text
0, 1, 15, 30, 37, 45, 60, 89, 90, 91, 135, 179, 180, 225, 269, 270, 315, 359
```

The page and all target geometry rotate together, isolating the learned layout
heads from OCR. Across all angles, minimum calibrated entity F1 is 0.7491,
canonical F1 0.9360, relation F1 0.3434, and composite 0.7227. The weakest
composite slice is 225°. Entity/canonical retention never falls below
95.30%/98.66% of upright.

## End-to-end angle results

One deterministic public test page from each of FATURA, FUNSD, and SROIE plus
one synthetic Thai page runs at every required angle: 72 total cases. K-Means
is disabled for this test and never controls OCR.

| Metric across the 18 public angle aggregates | Range |
|---|---:|
| Nonempty output rate | 1.0–1.0 |
| Recognized-text coverage | 0.4026–0.4368 |
| WER | 0.6553–0.7790 |
| Polygon detection F1 | 0.3330–0.3592 |
| Entity F1 | 0.1314–0.1830 |
| Relation F1 | 0–0.0205 |
| Canonical-field accuracy | 0.2222–0.5556 |
| Entity retention versus upright | 0.7413–1.0328 |

Synthetic Thai text has 1.0 recognized-text coverage, 0 WER, a nonempty
result, and the Thai route at all 18 angles. It proves routing/rotation
integration only; there is no compatible labeled public Thai benchmark.

The large gap between reference-token and end-to-end scores is the central
quality result: learned layout heads work on aligned text/boxes, while OCR
coverage and detection on unfamiliar layouts constrain real extraction.

## OCR preprocessing selection

The public dev-select ablation uses one deterministic page per labeled dataset.
`original`, grayscale normalization, and optional Paddle orientation modules
tie at 0.9033 mean alignment coverage; original wins the explicit no-change
tie break. Denoise falls to 0.8473. Background normalization/quality-auto
slightly reduce coverage. Raster-derived PDF results tie at 200/250/300 DPI,
so 200 is chosen for lower cost. Test and private data are not used.

## Unseen CORU

CORU contributes zero fit, dev, calibration, or in-domain test rows. On a
deterministic 100-page sample from its 1,261-page unseen population:

| Metric | Result |
|---|---:|
| Successful / failed pages | 100 / 0 |
| Nonempty OCR rate | 1.0 |
| QA answers found in OCR | 78.53% of 4,001 |
| Canonical exact-match accuracy | 15.68% of 523 applicable fields |
| Mean entities / relations / non-null fields | 13.26 / 3.47 / 6.71 |
| Mean processing time | 31.04 seconds/page |

CORU QA has answer strings but no compatible token polygons. Entity and
relation F1 are therefore undefined rather than fabricated.

## Private and integration evidence

The fixed checkpoint completed 2/2 anonymous local Gmail documents and 2 pages
with zero failures. The public artifact contains aggregates only and explicitly
declares no filename, OCR text, image, or per-document prediction. There is no
private ground truth and no accuracy claim.

The synthetic integration runner covers upright image, 45° image,
general-to-Thai two-page PDF, and Thai metadata routing. Its report hashes the
runner, verifier, config, schema, model registry, final training/calibration,
pipeline sources, 1.1 GB checkpoint artifacts, fixtures, and full ignored
outputs. Complete verification re-hashes 11 artifacts, validates every output
schema, and independently checks semantics.

## Authoritative artifacts

- `reports/final_model/final_model_card.md`
- `reports/final_model/error_analysis.md`
- `reports/final_model/{ocr,entity,relation,field,angle,language,dataset}_metrics.json`
- `reports/final_model/verification.json`
- `reports/final_model/evaluations/final_test_in_domain_ground_truth.json`

These reports are bounded academic evidence. They do not establish safety for
automated financial, legal, identity, or other high-stakes decisions.
