# Final Model Preimplementation Audit

**Audit date:** 2026-07-16
**Repository:** `PracticalSwan/csx4201-vision-info-extraction` (private)
**Audited branch:** `feat/rotation-robust-information-extraction` at `29d082d`
**Task classification:** local implementation, model training, conservative generated-artifact cleanup, and publication to `main`

## Decision

The repository has a verified end-to-end smoke lifecycle, but it does not yet
contain a final trained pre-model. The existing checkpoint must not be promoted
or renamed as final. Final work may proceed only after the dataset split,
profile binding, target retention, and real relation-training defects below are
covered by failing tests and fixed.

The raw corpus and preserved K-Means branch are healthy. The primary blocker is
the model-data/training path, not a shortage of public source annotations.

## Baseline verification

All commands below were executed against the live workspace before production
code changes.

| Gate | Result |
|---|---|
| Development tests | `158 passed, 1 skipped` in 43.11 seconds |
| Python compilation | `python -m compileall -q src scripts tests` exited 0 |
| Raw-data verifier | 20/20 checks passed; 128,767 inventory paths resolved; 100 sampled raw hashes unchanged |
| Rotation verifier | 20/20 checks passed; 603 prepared pages and 8,332 rotations provenance-valid; K-Means artifacts reload |
| Annotation verifier | Atomically refreshed `status: passed`; 12,433 normalized public annotations, 0 validation errors, 0 document-ID split leaks, Gmail fit rows 0 |
| OCR artifact/runtime verifier | Passed on `gpu:0`; general, Thai, and rotated phrase-recovery checks passed |
| OCR runtime partition | 53 tests passed (51 OCR/inference/privacy plus 2 layout-data tests) |
| CUDA layout partition | 1 model test passed in 56.59 seconds; together with the 2 layout-data tests this is the documented three-test layout surface |
| Complete IE verifier | 29/29 checks passed for the existing smoke lifecycle |
| GitHub state | Private repository; default branch `main`; no open or closed PRs returned; no force-push required |

The complete IE verifier currently accepts any nonempty model dataset. Its
29/29 result proves the existing smoke contract only; it is not evidence that
a final dataset or final checkpoint exists.

## Storage and runtime compatibility

Current live free space is 27.361 GiB on C: and 386.118 GiB on D:. Both pass
the required 15 GiB reserve, but C: has only about 12.36 GiB of task headroom.
All large new datasets, caches, checkpoints, evaluation outputs, logs, and
temporary files must remain below:

```text
D:\CSX4201\vision-info-extraction-assets
```

The required cache variables resolve to D: (`PADDLE_PDX_CACHE_HOME`,
`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`,
`PIP_CACHE_DIR`, `TMP`, and `TEMP`). The external asset inventory currently
contains:

- 9 smoke model-example JSON files (468,991 bytes);
- 10 OCR cache entries (595,423 bytes);
- a 2.114 GiB smoke checkpoint tree including its duplicate best snapshot;
- 0.137 GiB of exact PaddleOCR model files;
- 2.121 GiB of cached LayoutXLM source files.

Paddle GPU and CUDA PyTorch remain process-isolated because their Windows
cuDNN runtimes conflict. The layout environment is Python 3.10.11 with
PyTorch 2.8.0+cu128 and Transformers 4.57.6. A live compatibility probe found
no `detectron2` import and `pip index versions detectron2` returned no matching
Windows distribution. The final bounded implementation will therefore retain
the Detectron2-free text plus 2D-layout encoder and document the accuracy and
license implications rather than repeatedly attempting an unavailable wheel.

## OCR model identity

The exact required models were re-hashed and initialized through the real
Paddle GPU runtime:

| Role | Model | Aggregate artifact SHA-256 |
|---|---|---|
| Detector | `PP-OCRv6_medium_det` | `393f629d341e6388ca72d19b25983b96cae36dfdf1f7146adf42d6ab68789388` |
| General recognizer | `PP-OCRv6_medium_rec` | `666f7c4d6d5c846c7e202f63356b3fc4a7a3d4ff7040e763b93c5771608c7ae0` |
| Thai recognizer | `th_PP-OCRv5_mobile_rec` | `507c659e3abb6c2c7262a07965ec880119751a29e3c8ed7437b337f74b36608d` |

The preserved K-Means branch remains display-only and failure-isolated. The
failed historical exact-angle estimator remains disabled and will not be
reused for OCR correction.

## Public data inventory and source issues

The IE manifest contains 52,636 rows. It exposes 12,433 public usable normalized
pages and 203 redacted private-test rows. Usable public counts are:

| Dataset | Usable pages |
|---|---:|
| FATURA | 10,000 |
| CORU | 1,261 |
| SROIE | 973 |
| FUNSD | 199 |

The normalized directory contains 20,870 JSON files (455.34 MiB). Of those,
8,437 are unsupported CORU KIE records that were written before exclusion.
They are reproducible generated artifacts and are candidates for later
quarantine/removal only after the normalizer is changed not to recreate them
and a path-by-path cleanup manifest proves they are unreferenced.

Known genuine source defects remain separately classified: macOS artifacts,
empty CORU text, malformed CORU JSON, and components without full-document
images. These are not reasons to discard valid SROIE, FUNSD, FATURA, or CORU QA
pages.

## Why the smoke model has only nine examples

The 12,433 normalized records were not converted and then mostly rejected.
`src/information_extraction/model_dataset.py` first caps the smoke profile at
16 candidates. Its round-robin selection chose six CORU QA, four FATURA, three
FUNSD, and three SROIE pages. The current builder then:

1. rejects all six CORU QA pages because they have questions/answers but no
   source token polygons;
2. runs upright-only PaddleOCR on the remaining ten pages;
3. rejects an entire FUNSD page because its alignment coverage is 0.4943,
   below the hard 0.70 threshold.

That FUNSD page still retains 87/176 matched source tokens, 39/43 entities, and
16/20 relations in the existing cache. Whole-page rejection therefore throws
away substantial valid supervision. The resulting nine examples are four
FATURA, three SROIE, and two FUNSD pages, split 3 train / 2 validation / 4 test.

## Confirmed model-data defects

### Split and profile safety

- Document-ID leakage is zero, but broader required grouping is not safe.
  Fifty FATURA template-family groups span splits and cover 9,900 rows.
- Eight exact SROIE SHA-256 duplicate groups span splits and cover 16 rows.
- One shared `model_dataset_manifest.csv` is overwritten by every profile.
  `train_layout_model.py --profile final` does not assert that manifest rows,
  output roots, or provenance belong to `final`; it could currently train a
  falsely labeled final checkpoint from the three smoke training rows.
- Existing example JSON is reused without source/config/model provenance
  validation unless `--force` is supplied, and stale/orphan profile files are
  not reconciled.

### Target retention and labels

- The source-token path is not used even when official tokens, polygons,
  entities, and relations already exist.
- The hard page-level alignment gate does not preserve partially matched
  pages or confidence-weight valid targets.
- All 223 canonical-field token references in the nine smoke examples point
  to source token IDs that do not exist in the OCR-token example.
- BIO conversion infers boundaries only from adjacent class names, so adjacent
  distinct entities with the same class can be merged.
- Alignment OCR is fixed at orientation 0 and has no preprocessing-selection
  ablation or fine-angle candidate.

### Training and verification

- Token classification is the only learned task fed real dataset targets.
- The stored entity objects, 32 train relations, and canonical fields do not
  enter training.
- The saved relation head is trained on random tensors and synthetic labels;
  its reload test proves serialization, not learned relation extraction.
- There is no learned document-type head, canonical candidate-ranking head,
  confidence calibration, staged freeze/unfreeze schedule, composite
  checkpoint metric, bounded hyperparameter trial ledger, or checkpoint
  dataset-manifest hash binding.
- Dataset preparation and training do not call a disk gate.
- Final verification does not require profile `final`, 2,000 pages, dataset
  diversity, group-level leakage closure, real positive relations, or a
  checkpoint bound to the final manifest.

## Current quality baseline

The existing checkpoint used three public training examples, two micro-steps,
and one optimizer step. Validation loss is 2.3490 and token accuracy is 0.1007.
Checkpoint and synthetic relation-head reload differences are 0.0.

On the 16-run public smoke evaluation:

- text-detection precision/recall/F1: 0.5483 / 0.3333 / 0.4146;
- recognized-text coverage: 0.2503;
- CER/WER on referenced rows: 0.7497 / 0.8856;
- entity precision/recall/F1: 0.0062 / 0.0217 / 0.0096;
- relation F1: 0.0;
- canonical-field accuracy: 0.05;
- document-type accuracy: 0.25;
- 45-degree exact-orientation accuracy: 0.0.

These metrics are historical smoke evidence and must remain in the repository.
They do not meet the requested final-quality gates.

## Reusable components

The following components should be extended rather than rewritten:

- SROIE, FUNSD, and FATURA source-token/geometry adapters;
- CORU QA normalization and canonical question mapping;
- normalized annotation schema validation and atomic writers;
- exact-model `ModelRegistry` and hash verification;
- D:-backed provenance-complete OCR cache;
- partial alignment output with unmatched-token evidence;
- expanded-canvas pixel/polygon/box transforms;
- LayoutXLM tokenizer windowing and Detectron2-free multilingual 2D encoder;
- relation geometry features and evidence-backed deterministic field rules;
- page-isolated image/PDF inference, JSON Schema validation, and atomic output;
- public/private separation, private-output enforcement, and aggregate-only
  private reporting;
- display-only K-Means failure isolation.

## Exact implementation gates

Changes will follow red-green-refactor and will not start with expensive OCR or
training.

1. Add failing tests for source-token, OCR-aligned, and hybrid examples;
   partial alignment; canonical evidence remapping; entity boundaries;
   relation retention; no-private fit rows; profile binding; provenance reuse;
   disk gates; and document/template/hash/duplicate leakage.
2. Assign leakage-safe document groups across the complete eligible public
   corpus before profile sampling. Keep documents, PDFs, exact hashes, reliable
   near duplicates, FATURA templates, CORU stems, and derived views together.
3. Write profile-specific manifests and summaries. Require training to verify
   profile, manifest hash, dataset root, split policy, model-data schema, and
   private-fit count before loading an example.
4. Implement ground-truth, PaddleOCR, and hybrid token streams. Use official
   geometry directly where present; preserve partial OCR matches; record token
   source, alignment, entity/relation retention, OCR confidence, unmatched
   tokens, and data-quality score. Exclude only examples without a valid
   learning target.
5. Remap entity, relation, and canonical evidence IDs to the selected training
   token stream. Add deterministic OCR-noise augmentation without materializing
   rotated image copies.
6. Add independent coarse plus robust polygon-baseline fine-angle OCR search,
   bounded refinement, preprocessing ablations, and route evidence. Keep it
   separate from K-Means and the disabled historical estimator.
7. Feed real entity, document-type, and relation targets to multi-task
   training. Add balanced negatives, class weighting where validation
   justifies it, staged freeze/unfreeze, composite validation selection,
   deterministic resume, best-checkpoint reload, and calibration fitted only
   on public validation.
8. Run smoke, then at least 500-page development, then a final profile of at
   least 2,000 leakage-safe public pages and three datasets. Prefer all safely
   usable pages when the measured runtime/storage gate permits it.
9. Evaluate public held-out, natural unseen-domain, language, upright,
   cardinal, and arbitrary-angle slices. Diagnose and fix avoidable dominant
   errors within a maximum of six bounded trials. Preserve actual metrics if
   targets plateau; do not relabel a weak checkpoint as production quality.
10. Only after final artifacts reload and evaluation completes, update the
    private-test command, documentation, final verifier, and cleanup inventory;
    then run independent review, privacy/security diff review, and GitHub
    reconciliation to `main`.

## Stop conditions and honesty boundary

The final profile must actually run, the best checkpoint must reload in a clean
process, and every published metric must come from recorded predictions. If a
quality target remains below threshold after justified bounded attempts, the
best checkpoint may still be delivered as the final executed project model,
but the shortfall and bottleneck must be reported plainly. No private Gmail
content may influence training, calibration, threshold selection, or
hyperparameters.
