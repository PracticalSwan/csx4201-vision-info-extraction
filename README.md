# vision-info-extraction

CSX4201 vision preprocessing project for document rotation analysis. The
workspace contains a completed dataset organization layer and a completed,
reproducible rotation-zone baseline. OCR and information extraction have not
been implemented.

## Current result

The executed experiment is a **bounded full-angle run**. It applies every
configured angle to a deterministic sample of 100 public pages from each of
SROIE, FUNSD, FATURA, and CORU, plus all 203 rendered pages from 26 private
Gmail PDFs. It is not a full-corpus experiment: the live audit found 22,086
usable public full-document images, and materializing all required angles for
all of them would require an estimated 416,028 rotations and 219.08 GiB of
new space. The final full-profile gate had 16.12 GiB free and a required 10 GiB
reserve, so it correctly marked the unbounded run unsafe.

Positive angles are counterclockwise. Angles are normalized to [0, 360), and
the current provisional zone convention is:

| Zone | Half-open interval |
|------|--------------------|
| 1 | [0, 90) |
| 2 | [90, 180) |
| 3 | [180, 270) |
| 4 | [270, 360) |

This boundary convention is implemented and tested, but the professor still
needs to confirm it. See [Open project questions](#open-project-questions).

## Verified experiment at a glance

| Stage | Verified result |
|-------|-----------------|
| Selected pages | 603: 100 per public dataset and 203 private pages |
| Public splits | 280 train, 61 validation, 59 test pages |
| Private split | 203 pages in private_test only |
| Generated rotations | 8,332 successful, 0 failed |
| Zone balance | 2,083 rotations in each zone |
| Features | 8,332 fixed vectors, 1,957 values each, 0 invalid |
| Preprocessing | train-only StandardScaler and PCA from 1,957 to 128 dimensions |
| PCA variance retained | 84.22% cumulative explained variance |
| K-Means fit | k=4 on 5,600 public training rotations only |
| Cluster mapping | C0→Z1, C1→Z2, C2→Z4, C3→Z3 |
| Public zone accuracy | 37.81% validation; 37.92% test |
| Exact-angle reliability | 0 reliable public or private estimates at the configured threshold |
| Rotation verification | 20/20 checks pass |
| Tests | 113/113 pass |

The unsupervised result is modest. Validation and test mapped-zone accuracy are
about 38%, only moderately above the 25% four-class chance baseline. The
clustering metrics are also weak on held-out public data:

| Split | Rows | Accuracy | Macro F1 | ARI | NMI | Silhouette |
|-------|-----:|---------:|---------:|----:|----:|-----------:|
| Train | 5,600 | 0.5000 | 0.4947 | 0.3422 | 0.5038 | 0.1450 |
| Validation | 976 | 0.3781 | 0.3702 | 0.0886 | 0.1021 | 0.0288 |
| Test | 944 | 0.3792 | 0.3772 | 0.0871 | 0.1035 | 0.0245 |
| Private test, aggregate only | 812 | 0.5000 | 0.3654 | 0.4793 | 0.6174 | 0.4891 |

Cluster IDs are arbitrary. The one-to-one cluster-to-zone mapping above was
learned after fitting by Hungarian assignment on training labels only. It was
then fixed for validation, public test, and private aggregate evaluation.

## Exact-angle result

The exact-angle stage searches only inside the mapped K-Means zone, using
projection-profile, gradient, Hough-line, minimum-area-rectangle, and content
preservation evidence. It applies the correction as the signed negative of the
estimated counterclockwise angle.

This stage did not meet its reliability objective:

- Public validation and test: 1,920 attempts, 1,904 estimates, 16 hard
  failures, and all 1,904 estimates marked low-confidence.
- Public circular mean absolute error: 89.74 degrees; median error: 90 degrees.
- Private test: 812 aggregate-only attempts, 804 estimates, 8 hard failures,
  all 804 estimates low-confidence, and circular MAE 90.00 degrees.
- Reliable rate at the configured 0.50 threshold: 0% on public and private
  evaluation.

The reported correction direction is internally consistent, and pixel-derived
orientation scores often improve after correction, but that does not establish
angle correctness. With a 90-degree median error and zero reliable estimates,
the exact-angle output must be treated as a failed baseline, not a usable
orientation corrector.

## Pipeline design

The executed flow is:

    raw inventory
      -> public-safe page manifest and private operational manifest
      -> document/duplicate-safe 70/15/15 public splits plus private_test
      -> expanded RGB rotations on white canvas
      -> 1,957-value HOG/Hough/projection/edge/geometry features
      -> train-only StandardScaler and 128-component PCA
      -> train-only K-Means k=4
      -> train-only Hungarian cluster-to-zone mapping
      -> public row-level and private aggregate-only evaluation
      -> zone-guided exact-angle baseline

Public source images stay as read-only references. Private PDFs are rendered at
200 DPI to anonymized page IDs. Before rotation, each page is resized without
cropping to a maximum dimension of 1,024 pixels. Rotation uses bicubic
interpolation, an expanded canvas, and white fill.

The feature vector contains:

- 1,764 OpenCV HOG values;
- 48 Hough orientation and line-statistic values;
- 136 horizontal and vertical projection-profile values;
- 4 directional-edge values;
- 5 page geometry and density values.

No label is used to fit the feature transform, scaler, PCA, or K-Means. Labels
enter only after K-Means fitting to assign arbitrary cluster IDs to Zones 1–4.

## Data and privacy

| Dataset | Access | Selected pages | Role in this run |
|---------|--------|---------------:|------------------|
| SROIE | public | 100 | train/validation/test |
| FUNSD | public | 100 | train/validation/test |
| FATURA | public | 100 | train/validation/test |
| CORU | public | 100 | train/validation/test |
| Gmail | private | 203 pages from 26 PDFs | private_test only |

For CORU, only full-document images from Receipt Images & Key Information
Detection and Receipt Question Answering are eligible. The 30,148 OCR line
crops are excluded. Unreadable FUNSD macOS artifacts are also excluded.

The Gmail documents contain real personal financial and legal information:

- never commit or upload data/raw/private/gmail;
- never commit data/processed/private or private operational manifests;
- do not publish per-document private predictions, identifiers, paths, or
  previews;
- keep private evaluation aggregate-only;
- confirm repository visibility before any commit or external upload.

The latest rotation verifier found no private filenames in committable code,
tests, docs, config, or public artifacts.
That check does not authorize publication: inspect ignore rules and repository
visibility again before sharing.

## Installation

Requirements: Python 3.10 or newer. The executed environment used Python 3.14.

    python -m pip install -r requirements.txt

The rotation pipeline uses NumPy, OpenCV, scikit-learn, SciPy, joblib,
matplotlib, pandas, Pillow, PyMuPDF, PyYAML, and pytest. It does not require an
OCR framework or a deep-learning framework.

## Run the pipeline

Run from the project root. The complete command executes stages in dependency
order and stops if rotation verification fails:

    python scripts/run_rotation_experiment.py --profile full

The full profile means full configured angle coverage over the bounded selected
corpus. It does not mean the full 22,086-page public corpus.

For a fresh staged run:

    python scripts/prepare_page_images.py
    python scripts/create_rotation_splits.py
    python scripts/generate_rotation_data.py --profile smoke
    python scripts/verify_rotation_data.py --profile smoke
    python scripts/generate_rotation_data.py --profile full
    python scripts/verify_rotation_data.py --profile full
    python scripts/extract_rotation_features.py --profile full
    python scripts/fit_rotation_preprocessing.py
    python scripts/train_kmeans_rotation.py
    python scripts/evaluate_kmeans_rotation.py
    python scripts/evaluate_angle_estimation.py --profile full
    python scripts/verify_rotation_data.py --profile full --complete

The smoke run creates 52 rotations and provides the empirical storage estimate
used by the full-profile disk gate. The gate reserves at least 10 GiB free.
Use --force only when intentionally rebuilding replaceable derived artifacts.

Dataset organization commands remain available:

    python scripts/inspect_data.py --depth 2
    python scripts/organize_data.py --mode reference
    python scripts/audit_data.py
    python scripts/verify_data.py

Validation:

    python scripts/verify_rotation_data.py --profile full --complete
    python -m pytest tests/ -q

The --complete flag adds feature-cache, train-only provenance, saved-model,
and evaluation-output checks to the preparation/rotation checks.

## Artifact map

| Path | Contents |
|------|----------|
| data/raw/public | Original public datasets, read-only |
| data/raw/private/gmail | Original private PDFs, never publish |
| data/metadata/page_manifest.csv | Public-safe page inventory and selection |
| data/metadata/private_page_manifest.csv | Private source-path mapping, ignored |
| data/metadata/split_manifest.csv | Leakage-safe page split assignments |
| data/metadata/rotation_manifest.csv | One row per generated rotation |
| data/metadata/feature_manifest.csv | One row per feature vector |
| data/splits | Per-split page manifests |
| data/processed/private/page_images | Anonymized private PDF page renders |
| data/processed/rotated_images/full | Bounded full-profile rotated PNGs |
| data/processed/features/full | Raw and transformed NPZ feature caches |
| models/kmeans_rotation | Scaler, PCA, K-Means, mapping, and provenance |
| reports/rotation_preparation | Selection, split, smoke, and generation summaries |
| reports/feature_analysis | Feature method summary |
| reports/kmeans_evaluation | Metrics, predictions, boundary analysis, and plots |
| reports/angle_estimation | Public exact-angle rows and private aggregates |
| reports/verification | Raw baseline and rotation verification |

Private and bulk derived paths are intended to remain ignored by Git. Review
[.gitignore](.gitignore) before repository initialization.

## Reproducibility and safety controls

- Seed 42 drives selection, split assignment, PCA, K-Means, and metric sampling.
- Stable IDs and configuration/manifest hashes detect stale caches.
- Materialized page and rotation PNGs embed configuration and source-hash
  provenance; mismatched artifacts are regenerated instead of silently reused.
- Setting `materialize_existing_images: true` converts selected public sources
  to EXIF-normalized RGB PNGs; the executed run keeps the default read-only
  reference mode.
- Documents, exact duplicates, reliable reported near duplicates, FATURA
  template families, and shared CORU stems stay in one public split group.
- StandardScaler, PCA, K-Means, and Hungarian mapping record train-only fit
  provenance; private training count is zero.
- Saved scaler, PCA, and K-Means artifacts are reloaded and checked.
- Raw count, byte size, and 100 deterministic sampled SHA-256 hashes are checked
  against the recorded baseline.
- The privacy check covers source, tests, docs, root config, and public report
  surfaces as well as generated metadata and model summaries.
- The full run stops on leakage, stale artifacts, invalid vectors, non-bijective
  mapping, raw drift, privacy leakage, or insufficient disk reserve.

See [docs/requirements.md](docs/requirements.md) and
[docs/design.md](docs/design.md) for the acceptance criteria and design.

## Known limitations

- The public experiment uses 400 selected pages, not the entire public corpus.
- The selection is deterministic but is not claimed to be statistically
  representative of every template, language, or capture condition.
- K-Means models visual similarity without labels; four clusters do not
  naturally guarantee the four required angular quadrants.
- The training-derived mapping reaches only 50% training accuracy, and public
  held-out accuracy is about 38%.
- The exact-angle baseline has zero reliable estimates at the configured
  threshold and about 90-degree circular MAE.
- Confidence values are heuristic centroid or score margins, not calibrated
  probabilities.
- The organization-stage near-duplicate scan is bounded, so its recall is not
  exhaustive.
- The SROIE download contains a pre-existing LayoutLM checkpoint. It remains
  untouched and unused.

## Out of scope

This implementation does not perform:

- OCR;
- key-field or information extraction;
- a supervised rotation classifier;
- neural-network training;
- APIs, GUI work, deployment, or production serving.

## Open project questions

The following still require professor or user confirmation:

- Are exactly 90, 180, and 270 degrees assigned to the lower or upper zone?
  The implementation currently uses the upper zone via half-open intervals.
- Does “clustering” specifically require K-Means, or is a deterministic or
  supervised four-way orientation model acceptable?
- Which orientation-estimation approach is expected before zoning?
- What exactly does “pre-model” mean in the required deliverable?
- Which document types and fields must later information extraction target?
- What is the official evaluation metric and held-out test protocol?
- May any derived artifact be produced from the private Gmail set?
- Is the final deliverable a model, notebook, report, or submission?
- Will the future repository be public or private?

Do not treat the current implementation choices as answers to those open
questions.
