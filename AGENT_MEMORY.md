# AGENT_MEMORY.md — Shared project memory

> **READ RULE (MUST):** This file is shared across agents and sessions as
> orientation only. Its facts reflect a point in time. Verify every fact
> against the live filesystem and current user request before relying on it.
> Reading this file is not task completion. When a fact is stale, correct it
> and add a concise note to the change log.

## Project status

- Dataset organization, audit, and validation are complete.
- A bounded full-angle rotation-zone baseline completed on 2026-07-13.
- The baseline is technically reproducible but performs modestly: public
  validation/test zone accuracy is about 38%, and exact-angle reliability is
  0% at the configured threshold.
- A rotation-robust OCR/information-extraction inference and smoke-training
  lifecycle completed on 2026-07-15. It includes exact PaddleOCR general/Thai
  models, public annotation normalization, a Detectron2-free LayoutXLM text +
  2D-layout model, relations/fields, schema-valid image/PDF inference, bounded
  public evaluation, and aggregate-only private testing.
- This is implementation and smoke-lifecycle completion, not final model
  quality. The public smoke entity F1 is about 0.01, relation F1 is 0, and the
  trained checkpoint used three public training examples.
- The workspace is a Git repository with an existing GitHub remote. Recheck
  live visibility and staged privacy before every push.

## Confirmed goal

- The professor confirmed a vision pre-model for information extraction from
  images/files, with four rotation zones.
- Current provisional zones are half-open: [0,90), [90,180), [180,270), and
  [270,360). Positive angles are counterclockwise.
- Boundary inclusivity, whether K-Means is mandatory, the expected
  orientation-estimation method, and the meaning of “pre-model” remain open.
  The current implementation choices do not settle those questions.

## Verified dataset facts

- Workspace: C:\Assumption University\CSX4201\Project.
- Raw data: 128,793 files and 35,459,126,772 bytes.
- Public datasets: SROIE, FUNSD, FATURA, and CORU.
- Private data: 26 real Gmail PDFs under data/raw/private/gmail.
- The organization inventory contains 128,793 rows and no walk errors.
- Known invalid/unreadable/empty count remains 408: FUNSD macOS artifacts,
  six empty CORU text files, and four malformed CORU JSON files.
- The legacy vision_info_extraction_data directory remains an empty,
  non-destructively preserved husk.

## Verified rotation run

- Usable public full-document pool: 22,086 pages.
- Unbounded capacity estimate: 416,028 rotations and 219.08 GiB of new space
  with 16.12 GiB free at the final full-profile gate and a 10 GiB reserve;
  correctly marked unsafe.
- Bounded selection: 100 pages from each public dataset plus all 203 private
  pages rendered from 26 PDFs; 603 pages total.
- Public page splits: 280 train, 61 validation, 59 test. Private: 203
  private_test pages.
- Split grouping keeps logical documents, exact duplicates, reliable reported
  near duplicates, FATURA template families, and shared CORU source stems
  together. Latest split report records zero page, document, group, or
  public/private leakage.
- Smoke profile: 52 successful rotations.
- Full profile: 8,332 successful rotations, zero failures; 5,600 train, 976
  validation, 944 test, and 812 private_test.
- Each of the four zones contains exactly 2,083 rotations.
- Public sources remain read-only references. Private PDFs render at 200 DPI
  to anonymous page IDs in ignored processed storage.
- Rotation verification: 20/20 checks pass, including raw integrity, page and
  rotation PNG provenance, manifest
  consistency, boundary coverage, split isolation, private-name scan, and no
  generated files under raw.

## Verified feature and model facts

- Feature vector: 1,957 values: 1,764 HOG, 48 Hough/line, 136 projection,
  four directional-edge, and five geometry values.
- Feature extraction produced 8,332 finite vectors with zero failures.
- StandardScaler and PCA fit only 5,600 public training rows.
- PCA output: 128 dimensions; cumulative explained variance 84.22%.
- K-Means: k=4, seed 42, n_init=20; cluster sizes 1,112, 1,114, 1,686,
  and 1,688.
- Training-only Hungarian mapping: C0→Z1, C1→Z2, C2→Z4, C3→Z3.
- Saved scaler, PCA, and K-Means artifacts passed reload checks.
- Mapped-zone accuracy: train 50.00%, validation 37.81%, test 37.92%.
- Public test ARI/NMI: 0.0871/0.1035. These are modest unsupervised results,
  not evidence that the four clusters recover the required quadrants.
- Private zone results are aggregate-only; no private row prediction is
  written.

## Verified exact-angle facts

- Exact-angle evaluation runs on validation, test, and private_test only.
- Public: 1,920 attempts, 1,904 estimates, 16 hard failures, zero reliable
  estimates, circular MAE 89.74 degrees, median error 90 degrees.
- Private aggregate: 812 attempts, 804 estimates, eight hard failures, zero
  reliable estimates, circular MAE 90.00 degrees.
- Every non-failed estimate was marked low-confidence at the configured 0.50
  threshold.
- Correction semantics are negative estimated angle, but the output is not a
  dependable exact-angle corrector. Pixel orientation-score improvement does
  not override the poor circular error.

## Tooling and artifacts

- Organization CLIs: inspect_data.py, organize_data.py, audit_data.py,
  verify_data.py.
- Rotation CLIs: prepare_page_images.py, create_rotation_splits.py,
  generate_rotation_data.py, verify_rotation_data.py,
  extract_rotation_features.py, fit_rotation_preprocessing.py,
  train_kmeans_rotation.py, evaluate_kmeans_rotation.py,
  evaluate_angle_estimation.py, run_rotation_experiment.py.
- Main rotation modules: rotation_common.py, page_preparation.py,
  rotation_dataset.py, orientation_features.py, rotation_model.py,
  angle_estimation.py.
- Main result roots: data/metadata, data/splits, data/processed,
  models/kmeans_rotation, and reports.
- OCR/IE CLIs: normalize_ie_annotations.py, setup_ie_environment.ps1,
  download_ocr_models.py, verify_ocr_models.py, prepare_model_dataset.py,
  train_layout_model.py, extract_document.py, evaluate_information_extraction.py,
  evaluate_private_gmail.py, verify_ie_annotations.py, and
  verify_information_extraction.py.
- Large OCR/layout/model/checkpoint/cache assets live below
  D:\CSX4201\vision-info-extraction-assets in isolated Python 3.10 OCR and
  CUDA-layout environments.
- Required OCR models: PP-OCRv6_medium_det, PP-OCRv6_medium_rec, and
  th_PP-OCRv5_mobile_rec; model hashes and GPU smoke initialization pass.
- Public annotation normalization produced 12,433 records; the aligned smoke
  model dataset contains 9 usable examples (3 train, 2 validation, 4 test),
  with Gmail fit rows 0.
- Layout smoke training saved/reloaded model and relation-head logits with
  maximum difference 0.0. Validation loss was 2.3490 and token accuracy 0.1007.
- Public angle smoke completed 16/16 runs. Private operational inference
  completed 2/2 pages and published aggregate counts only.
- Current development suite: 153 tests pass with one environment-dependent
  skip; OCR-runtime and CUDA-layout partitions pass 50 and 3 tests.

## Open questions

- Do the provisional canonical fields and document types match the professor's
  final target scope?
- Are exact boundary angles assigned to the lower or upper zone?
- Is K-Means specifically required, or may a deterministic/supervised
  four-way orientation method be used?
- Which angle-estimation approach is expected?
- What does “pre-model” mean in the final deliverable?
- What are the official quality thresholds and held-out protocol? Current
  smoke and natural CORU-holdout results are not final benchmarks.
- May any additional derived artifact be produced from the private Gmail set
  beyond ignored local inference and aggregate-only reporting?
- Is the deliverable a model, notebook, report, or submission?
- Will a future GitHub repository be public or private?

## Standing cautions

- Never commit or externally upload private Gmail source or derived data.
- Never write private per-row predictions, identifiers, paths, or previews to
  public reports.
- Recheck live ignore rules and repository visibility before any commit.
- Do not call the bounded full-angle run a full-corpus run.
- Do not present the current K-Means or exact-angle metrics as a successful
  final model.
- Verify artifact hashes and train-only provenance before reusing cached
  features or models.
- Materialized page and rotation PNG reuse must match embedded source and
  configuration provenance. Privacy scans must include committable source,
  tests, docs, and root config, not only generated report directories.

## Change log

- 2026-07-13 — Initialized with dataset structure, privacy cautions, and open
  project questions.
- 2026-07-13 — Recorded the professor-confirmed vision pre-model and four-zone
  goal.
- 2026-07-13 — Completed organization/audit/validation: 128,793 raw files,
  public/private separation, reusable tooling, and 47 tests at that stage.
- 2026-07-13 — Completed the bounded rotation-zone baseline: 603 pages, 8,332
  rotations, 1,957-value features, train-only 128-component PCA and K-Means,
  mapped/boundary evaluation, exact-angle evaluation, 20/20 rotation checks,
  and 113 passing tests. Recorded modest zone accuracy and failed exact-angle
  reliability without overstating the result.
- 2026-07-13 - Independent review found private-name literals in synthetic
  tests, incomplete privacy-scan coverage, stale derived-PNG reuse after config
  changes, and an inert public-image materialization flag. Fixed all findings,
  rebuilt the provenance-bound pipeline, and added regressions.
- 2026-07-13 - The single follow-up review validated live-source hashing,
  exact manifest enums, privacy coverage, artifact regeneration, tests, and
  both verifiers; no reproducible violations remained.
- 2026-07-15 - Added the D:-backed, process-isolated PaddleOCR/LayoutXLM
  implementation; normalized public annotations; smoke-trained and reloaded a
  public-only layout checkpoint; verified image, rotated, Thai, unknown, and
  multipage-PDF inference; executed bounded public and aggregate-only private
  evaluation; retained K-Means as a failure-isolated display branch. Recorded
  low model-quality metrics without presenting the smoke checkpoint as final.
- 2026-07-15 - Final-review correction pass replaced static integration claims
  with a tracked hash-bound runner and independent semantic verification,
  required real rotated phrase recovery, corrected automatic Thai retry and
  bounded script scoring, and added polygon detection plus recognized-text
  metrics. Public smoke detection P/R/F1 is 0.5483/0.3333/0.4146 and
  recognized-text coverage is 0.2503; the development/OCR/layout partitions
  pass 158 (1 skipped), 53, and 3 tests respectively. The repository remote was
  confirmed private before publication; Gmail fit rows remain 0.
- 2026-07-15 - The permitted second and final independent review rechecked the
  three prior blockers, validated 11 integration artifacts plus 13 focused
  regressions, and confirmed all three closed with no reproducible completion
  blocker remaining.
