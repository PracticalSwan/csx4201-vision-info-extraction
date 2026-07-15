# Information-Extraction Task Tracker

## Preserved baseline

- [x] Freeze rotation artifacts, reports, and historical metrics.
- [x] Re-run raw-data, artifact-reload, privacy, and 20-check rotation verification.
- [x] Add a failure-isolated, display-only K-Means wrapper.
- [x] Keep the unreliable exact-angle experiment disabled for inference.

## OCR and storage

- [x] Create D:-backed OCR and layout environments with a 15 GiB reserve gate.
- [x] Isolate Paddle GPU from CUDA PyTorch on Windows.
- [x] Download and hash the exact detector, general, and Thai models.
- [x] Implement general/Thai/auto routing and OCR-evidence orientation scoring.
- [x] Add provenance-aware public OCR caching and private-cache exclusion.
- [x] Verify general, Thai, rotated, image, PDF, and multipage paths.

## Data and model

- [x] Define the universal JSON and annotation schemas.
- [x] Normalize SROIE, FUNSD, FATURA, and supported CORU annotations.
- [x] Build public/private-safe information-extraction and model manifests.
- [x] Implement continuous rotation plus polygon/box transformation.
- [x] Prepare the aligned public smoke model dataset with Gmail fit rows 0.
- [x] Implement the Detectron2-free LayoutXLM text + 2D-layout model.
- [x] Smoke-train on CUDA and verify model/tokenizer/relation-head reload.
- [x] Implement entity inference, key-value relations, canonical rules, and generic fallback.

## Inference and evaluation

- [x] Implement image, PDF, multipage, rotated, multilingual, and unknown-type CLI inference.
- [x] Validate and atomically write schema-compliant JSON.
- [x] Run rule and OCR baselines.
- [x] Run 16-case public angle smoke evaluation with explicit OCR reference coverage.
- [x] Report recognized-text coverage and one-to-one polygon detection precision/recall/F1.
- [x] Execute a natural zero-fit-row CORU holdout and report its limitations.
- [x] Run aggregate-only private Gmail operational inference.
- [x] Generate and cryptographically revalidate synthetic image/rotation/Thai/multipage evidence.
- [x] Add unit/regression tests and environment-specific test partitions.
- [ ] Train a final-quality model over a larger public corpus.
- [ ] Run professor-approved labeled Thai and leave-one-dataset-out benchmarks.

## Finalization

- [x] Update README, summary, requirements, design, setup, IE, evaluation, privacy, memory, and lessons.
- [x] Run final combined verification after documentation stabilizes.
- [x] Complete both bounded independent reviews and fix every confirmed finding.
- [x] Inspect staged content, create the feature branch, commit, and push to the existing safe remote.

The unchecked research items are not hidden implementation failures. They need
a bounded full-training plan and official evaluation decisions. The unchecked
finalization items are completed in the same implementation session before
publication.
