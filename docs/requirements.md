# Rotation Pipeline Requirements

Acceptance criteria and measured outcomes for the completed bounded baseline.

## Purpose and scope

The implemented baseline prepares document pages, generates balanced synthetic
rotations, learns four unsupervised clusters, maps those clusters to the
professor's four angle zones, and evaluates a zone-guided exact-angle method.

The executed full profile is bounded to 100 pages per public dataset plus all
203 private pages. “Full” means complete configured angle coverage over that
selected corpus, not the full 22,086-page public corpus.
The unbounded capacity estimate projected 416,028 rotations and 219.08 GiB of
new space with 16.12 GiB free at the final full-profile gate and a 10 GiB
reserve.

## Functional requirements

| ID | Requirement | Validated state |
|----|-------------|-----------------|
| U-001 | Treat every file below data/raw as read-only. | Pass: raw baseline unchanged; no generated output under raw. |
| U-002 | Use half-open zones [0,90), [90,180), [180,270), [270,360). | Pass as provisional convention; professor confirmation remains open. |
| U-003 | Define positive angles as counterclockwise and normalize modulo 360. | Pass in generation, filenames, manifests, tests, and reports. |
| U-004 | Separate public train, validation, test, and private_test at page, document, and duplicate-group level. | Pass: zero page, document, split-group, or private/public leakage. |
| U-005 | Use Gmail PDFs only for private_test transformation and aggregate evaluation. | Pass: 203 pages; zero private fit rows and zero private row-level prediction outputs. |
| U-006 | Extract fixed HOG, Hough, projection, directional-edge, and geometry features without OCR. | Pass: 8,332 vectors, 1,957 values each, zero invalid. |
| U-007 | Fit StandardScaler, optional PCA, K-Means, and cluster mapping from public training data only. | Pass: all fit provenance records train only and zero private rows. |
| U-008 | Fit scikit-learn K-Means with exactly four non-empty clusters. | Pass: sizes 1,112, 1,114, 1,686, 1,688. |
| U-009 | Learn a deterministic one-to-one cluster-to-zone mapping with Hungarian assignment. | Pass: C0→Z1, C1→Z2, C2→Z4, C3→Z3. |
| U-010 | Evaluate raw clusters, mapped zones, boundary angles, confidence, and circular exact-angle error. | Pass as evaluation coverage; model quality remains modest. |

## Safety and lifecycle requirements

| ID | Requirement | Validated state |
|----|-------------|-----------------|
| S-001 | Record raw counts, size, and deterministic sampled hashes before preprocessing. | Pass: 128,793 files, 35,459,126,772 bytes, 100 sampled hashes. |
| S-002 | Estimate storage from smoke output and preserve at least 10 GiB free before full materialization. | Pass: the 52-rotation smoke profile projected 219.08 GiB for the 416,028-rotation unbounded run, which was rejected. |
| S-003 | Stop before fitting if page, document, duplicate-group, or private leakage is detected. | Pass in validator and synthetic negative tests. |
| S-004 | Keep real private filenames, paths, previews, and per-row private predictions out of committable code and public outputs. | Pass: scanner covers source, tests, docs, config, metadata, reports, and models; private evaluation is aggregate-only. |
| S-005 | Use stable IDs, configuration hashes, source hashes, atomic writes, and cache validation for repeatability. | Pass: materialized page/rotation PNGs carry verified provenance and stale artifacts regenerate. |
| S-006 | Reloaded artifacts must reproduce transforms and predictions within tolerance. | Pass for scaler, PCA, and K-Means. |
| S-007 | Describe the run as bounded full-angle coverage over a deterministic selected corpus. | Pass in README, summary, reports, design, and memory. |

## Quality results

The requirements above establish pipeline behavior and safety. They do not
imply that the learned model meets a production-quality accuracy target.

- Public mapped-zone accuracy: 37.81% validation and 37.92% test.
- Public test ARI/NMI: 0.0871/0.1035.
- Public exact-angle evaluation: 1,920 attempts, 16 hard failures, zero reliable
  estimates, 89.74-degree circular MAE, and 90-degree median error.
- Private exact-angle evaluation: 812 aggregate-only attempts, eight hard
  failures, zero reliable estimates, and 90.00-degree circular MAE.
- Rotation verification: 20/20 pass.
- Synthetic tests: 113/113 pass.

The exact-angle stage therefore satisfies evaluation and failure-reporting
requirements but fails the intended reliability objective. Its output must not
be treated as a dependable correction.

## Out of scope

The current system does not implement:

- OCR;
- key-field or information extraction;
- a supervised rotation classifier;
- neural networks;
- APIs, GUI work, deployment, or production serving.

## Open acceptance decisions

Professor or user confirmation is still required for:

- exact zone-boundary inclusivity;
- whether “clustering” specifically requires K-Means;
- the expected angle-estimation source or method;
- the meaning of “pre-model”;
- target document types and extraction fields;
- official metric and held-out protocol;
- allowed use of private Gmail-derived artifacts;
- final deliverable format and repository visibility.

The current half-open boundaries and K-Means approach are provisional
implementation decisions, not confirmed answers.
