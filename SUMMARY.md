# Summary — Rotation-Zone Baseline

**Project:** vision-info-extraction, CSX4201  
**Completed stages:** dataset organization and bounded rotation-zone experiment  
**Run date:** 2026-07-13

## Outcome

The rotation pipeline is implemented and has run end to end. It prepared a
leakage-safe selected corpus, generated 8,332 balanced rotations, extracted
fixed orientation features, fitted train-only preprocessing and K-Means, mapped
clusters to the professor's four zones, evaluated boundary cases, and tested a
zone-guided exact-angle estimator.

The engineering pipeline passed its integrity and privacy checks. The modeling
result is a baseline, not a successful final model: mapped public test accuracy
is 37.92%, and the exact-angle estimator produced no reliable estimates at the
configured threshold.

## Scope actually executed

The live dataset has 22,086 usable public full-document images. Available disk
space did not support all configured rotations over that entire corpus. The
measured capacity estimate projected 416,028 rotations and 219.08 GiB of new
space with 16.12 GiB free at the final full-profile gate and a 10 GiB reserve,
so it marked the unbounded run
unsafe. The executed run used a deterministic cap of 100 pages per public
dataset:

| Dataset | Selected pages |
|---------|---------------:|
| SROIE | 100 |
| FUNSD | 100 |
| FATURA | 100 |
| CORU | 100 |
| Gmail, private | 203 pages from 26 PDFs |
| **Total** | **603** |

This is a **bounded full-angle run**: every configured angle was applied to the
selected corpus, but the full public corpus was not materialized.

Public pages were assigned as 280 train, 61 validation, and 59 test pages.
Private pages remained in private_test. Documents and duplicate-related groups
did not cross public splits.

## Prepared data

Positive angles are counterclockwise. The implemented provisional convention
uses [0,90), [90,180), [180,270), and [270,360) for Zones 1–4.

- Training pages received 20 angles each.
- Validation and test pages received all 16 required boundary and interior
  angles.
- Private pages received 45, 135, 225, and 315 degrees.
- The run generated 8,332 RGB PNG rotations with no failures.
- Each zone contains exactly 2,083 rotations.
- Public source images remained read-only references.
- Private PDFs were rendered at 200 DPI to anonymous page IDs.
- The smoke run measured 52 rotations before the full disk gate.

## Features and model

Each rotation produced a 1,957-value feature vector:

- 1,764 HOG values;
- 48 Hough and line-statistic values;
- 136 projection-profile values;
- 4 directional-edge values;
- 5 geometry and density values.

All 8,332 vectors were finite and valid. StandardScaler and PCA fitted only the
5,600 public training rows. PCA reduced 1,957 values to 128 components and
retained 84.22% cumulative explained variance.

K-Means used k=4, seed 42, and 20 initializations. The fitted training cluster
sizes were 1,112, 1,114, 1,686, and 1,688. Hungarian assignment learned the
one-to-one training mapping:

    Cluster 0 -> Zone 1
    Cluster 1 -> Zone 2
    Cluster 2 -> Zone 4
    Cluster 3 -> Zone 3

The scaler, PCA, K-Means, and mapping all record train-only provenance. Private
fit rows: zero. Reloaded artifacts reproduced transforms and predictions.

## Zone results

| Split | Rotations | Accuracy | Macro F1 | ARI | NMI |
|-------|----------:|---------:|---------:|----:|----:|
| Train | 5,600 | 0.5000 | 0.4947 | 0.3422 | 0.5038 |
| Validation | 976 | 0.3781 | 0.3702 | 0.0886 | 0.1021 |
| Test | 944 | 0.3792 | 0.3772 | 0.0871 | 0.1035 |
| Private test, aggregate only | 812 | 0.5000 | 0.3654 | 0.4793 | 0.6174 |

The public held-out metrics are modest. The model does not cleanly recover four
rotation quadrants, and confidence is especially low around zone boundaries.
K-Means is unsupervised, so requiring four clusters does not guarantee that
those clusters correspond to the four desired angle intervals.

## Exact-angle result

The exact-angle stage used the predicted zone to bound a coarse-to-fine search.
It scored candidate corrections with projection, gradient, Hough,
minimum-area-rectangle, and content-preservation evidence.

| Evaluation | Attempts | Estimates | Hard failures | Reliable | Circular MAE | Median error |
|------------|---------:|----------:|--------------:|---------:|-------------:|-------------:|
| Public validation + test | 1,920 | 1,904 | 16 | 0 | 89.74° | 90.00° |
| Private test, aggregate only | 812 | 804 | 8 | 0 | 90.00° | 90.00° |

Every non-failed estimate was below the 0.50 reliability threshold. The
correction sign was implemented consistently as negative estimated angle, but
the 90-degree median error means the output is not a dependable exact-angle
corrector.

## Privacy and safety result

- Real Gmail filenames remain confined to ignored private metadata. The final
  scan covers committable source, tests, docs, config, and public artifacts.
- Public reports contain no private source path, preview, or per-row private
  prediction.
- Private K-Means and exact-angle results are aggregate-only.
- No generated file was placed under data/raw.
- The raw baseline still matches 128,793 files, 35,459,126,772 bytes, and 100
  deterministic sampled SHA-256 hashes.
- Rotation verification passed all 20 checks, including prepared/rotated PNG
  provenance, feature caches, fit
  provenance, model compatibility, and evaluation outputs.

The first independent review found private-name literals in tests and stale-PNG
reuse when page or rotation settings changed. The fixtures were anonymized,
the privacy scan was broadened, page/rotation PNGs gained embedded source and
configuration provenance, and the previously inert public-image
materialization setting was implemented. Regression tests cover all three
fixes. The single permitted follow-up review found no reproducible violations
and approved finalization.

Private source and derived data must not be committed or uploaded. Repository
visibility is still undecided.

## Reproduce and verify

Run the complete bounded experiment:

    python scripts/run_rotation_experiment.py --profile full

Verify artifacts and tests:

    python scripts/verify_rotation_data.py --profile full --complete
    python scripts/verify_data.py
    python -m pytest tests/ -q

Current test result: **113 passed**.

Key results are under:

- reports/rotation_preparation
- reports/feature_analysis
- reports/kmeans_evaluation
- reports/angle_estimation
- reports/verification
- models/kmeans_rotation

See [README.md](README.md) for commands and the artifact map, and
[docs/design.md](docs/design.md) for the design and limitations.

## What remains

The implementation intentionally excludes OCR, field extraction, supervised
classification, neural networks, APIs, GUI work, and deployment.

Professor confirmation is still required for boundary inclusivity, whether
K-Means is mandatory, the expected angle-estimation method, the meaning of
“pre-model,” target fields and document types, the official evaluation
protocol, private-set use, and the final deliverable format.

The next modeling decision should be driven by those answers. The current
metrics do not justify presenting the unsupervised baseline as the final
rotation solution.
