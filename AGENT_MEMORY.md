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
- A full rotation-robust OCR/information-extraction lifecycle completed on
  2026-07-17. It includes exact PaddleOCR general/Thai models, public
  annotation normalization, a Detectron2-free LayoutXLM text + 2D-layout
  model, calibrated entities/relations/fields, schema-valid image/PDF
  inference, locked public evaluation, unseen-domain testing, and
  aggregate-only private operation.
- Final build `final-6be3e0b46b0a4e4c` contains 11,684 examples and trained
  four epochs over 7,782 public training examples. Its exact checkpoint hash
  is `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`.
  Reference-token entity F1 is 0.9813; bounded end-to-end entity F1 is only
  0.1314-0.1830 because OCR remains the main bottleneck. Treat it as a final
  academic pre-model, not a production or high-stakes system.
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
- OCR/IE CLIs include normalization and verification, environment/model
  setup, final dataset preparation, multi-task training/calibration,
  image/PDF inference, locked and angle-grid evaluation, unseen CORU testing,
  bounded private operation, integration smoke, report compilation, and the
  complete information-extraction verifier.
- Large OCR/layout/model/checkpoint/cache assets live below
  D:\CSX4201\vision-info-extraction-assets in isolated Python 3.10 OCR and
  CUDA-layout environments.
- Required OCR models: PP-OCRv6_medium_det, PP-OCRv6_medium_rec, and
  th_PP-OCRv5_mobile_rec; model hashes and GPU smoke initialization pass.
- Public annotation normalization produced 12,433 authoritative records. The
  final aligned build contains 11,684 public examples, with Gmail fit rows 0.
- Final multi-task training saved/reloaded all heads with maximum logit
  difference 0.0. Public-only calibration is bound to the exact build,
  manifest, and checkpoint hashes.
- Locked in-domain calibrated entity/canonical/relation F1 is
  0.9813/0.9814/0.4632. The 18-angle layout grid retains at least 95.30% of
  upright entity F1; the 72-case end-to-end grid exposes the weaker real-OCR
  path while synthetic Thai recovery passes 18/18 angles.
- Unseen CORU completed 100/100 pages without failures. Private operational
  inference completed 2/2 pages and published aggregate counts only.
- Current host suite: 227 tests pass with two environment-dependent skips;
  OCR-runtime and CUDA-layout partitions pass 122 and 2 tests.

## Open questions

- Do the provisional canonical fields and document types match the professor's
  final target scope?
- Are exact boundary angles assigned to the lower or upper zone?
- Is K-Means specifically required, or may a deterministic/supervised
  four-way orientation method be used?
- Which angle-estimation approach is expected?
- What does “pre-model” mean in the final deliverable?
- What are the professor's official quality thresholds and held-out protocol?
  The executed locked and unseen-domain evaluations are project evidence, not
  an official course benchmark.
- May any additional derived artifact be produced from the private Gmail set
  beyond ignored local inference and aggregate-only reporting?
- Is the deliverable a model, notebook, report, or submission?
- Must the repository remain private for the final course handoff? Recheck
  live visibility before every publication action.

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
- 2026-07-17 - Completed the final public multi-task run, public-only
  calibration, one locked in-domain test, required layout and end-to-end angle
  grids, exact OCR/integration verification, deterministic 100-page unseen
  CORU evaluation, and aggregate-only two-page private operation. The final
  checkpoint reloads exactly; host/OCR/layout test partitions pass
  227 (2 skipped), 122, and 2 tests. Real OCR remains the documented
  end-to-end bottleneck, and K-Means remains display-only.
- 2026-07-17 - Final independent review closed explicit-checkpoint
  documentation, fail-closed calibration and private-input boundaries,
  transitive learned-worker integration hashes, locked 100/100 unseen
  verification, required private-inventory scanning, and exact model-example
  reuse validation. Fresh integration passes with 17 source hashes, 11
  external artifacts, four cases, and zero private inputs; complete IE
  verification passes 46/46. Conservative cleanup removed 19.740 GiB of
  obsolete development/smoke datasets and checkpoints while preserving and
  re-hashing the final checkpoint, resume state, and final model dataset.
