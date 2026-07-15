# Rotation Pipeline Task Tracker

Execution status for the rotation baseline and its remaining review gate.

## Foundation

- [x] Inspect live repository, metadata, code, tests, privacy boundaries, and
  disk capacity.
- [x] Run baseline tests and organization verification.
- [x] Define requirements, design, stop conditions, angle semantics, and
  bounded-corpus terminology.
- [x] Extend configuration and runtime dependencies.

## Preparation

- [x] Implement deterministic source selection and public/private page
  preparation.
- [x] Implement document-, duplicate-, template-, and private-safe splits.
- [x] Implement counterclockwise rotation generation, smoke estimation, and
  the full disk gate.
- [x] Implement raw-integrity, manifest, image, balance, boundary, leakage, and
  privacy verification.
- [x] Execute and verify the 52-rotation smoke profile.
- [x] Execute the bounded full-angle profile: 603 pages and 8,332 rotations,
  with zero failures and 2,083 rows per zone.

## Modeling

- [x] Implement and run fixed 1,957-value orientation feature extraction.
- [x] Fit train-only StandardScaler and 128-component PCA with provenance and
  reload checks.
- [x] Fit train-only K-Means k=4 and learn the one-to-one Hungarian mapping.
- [x] Evaluate raw clusters, mapped zones, confidence, datasets, document
  types, and boundary angles.
- [x] Implement and evaluate zone-guided exact-angle correction with explicit
  failure and low-confidence handling.
- [x] Record the modest held-out zone result and failed exact-angle reliability
  without presenting either as a final model.

## Quality and handoff

- [x] Add synthetic angle, split, rotation, feature-cache, leakage, provenance,
  artifact-reload, and model tests.
- [x] Run the full suite: 113 tests pass.
- [x] Run the full rotation verifier: 20/20 checks pass.
- [x] Update README, summary, requirements, design, task tracker, project
  memory, and genuine lessons from the executed run.
- [x] Complete one independent evidence-based review after the implementation
  and documentation tree is stable.
- [x] Fix confirmed review defects: private fixture literals, incomplete privacy
  scan coverage, stale page/rotation reuse, and public-image materialization.
- [x] Run the one permitted follow-up review over the repaired stable tree; no
  reproducible violations remained and the reviewer approved finalization.

## Remaining project decisions

- [ ] Confirm exact zone-boundary inclusivity with the professor.
- [ ] Confirm whether K-Means is mandatory or whether a deterministic or
  supervised orientation method is acceptable.
- [ ] Confirm the expected exact-angle source/method and the meaning of
  “pre-model.”
- [ ] Define target document types, fields, official metric, and held-out
  protocol.
- [ ] Define permitted private-set use, final deliverable format, and repository
  visibility.
