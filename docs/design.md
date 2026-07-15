# Rotation Pipeline Design

Implemented architecture, safety controls, artifacts, and observed tradeoffs.

## Design status

This document describes the implementation that completed its bounded
full-angle run on 2026-07-13. The pipeline and safety controls passed
verification. Model quality is modest, and the exact-angle baseline did not
meet its reliability threshold.

## Scope decision

The live audit found 22,086 usable public full-document images and 203 private
PDF pages. Materializing every configured angle for the entire public pool
would create an estimated 416,028 rotations and consume 219.08 GiB of new
space. The final full-profile gate observed 16.12 GiB free and enforced a 10 GiB
reserve, so it marked the unbounded profile unsafe.

The implemented full profile therefore selects, deterministically:

- 100 SROIE pages;
- 100 FUNSD pages;
- 100 FATURA pages;
- 100 CORU full-document pages;
- all 203 pages from 26 private Gmail PDFs.

The public cap preserves source and split strata where practical. FATURA uses
round-robin template-family selection. CORU selects equally from Receipt Images
& Key Information Detection and Receipt Question Answering. CORU OCR line
crops and CSV-only item extraction are excluded.

A 52-rotation smoke run measures average artifact size. The full run proceeds
only if its empirical estimate preserves at least 10 GiB free and stays below
the configured disk-use ceiling.

## Data flow

    organization inventory
      -> page preparation and deterministic selection
      -> unioned leakage groups and split assignment
      -> smoke or bounded full rotation materialization
      -> rotation verification
      -> fixed orientation feature extraction
      -> train-only StandardScaler and PCA
      -> train-only K-Means k=4
      -> train-only Hungarian cluster-to-zone mapping
      -> raw-cluster and mapped-zone evaluation
      -> zone-guided exact-angle evaluation
      -> final rotation verification

The orchestrator in scripts/run_rotation_experiment.py follows this order and
stops before fitting if rotation verification fails.

## Identity and leakage control

Page IDs and document IDs are stable hashes of source identity. A union-find
structure combines selected public pages when any of these conditions apply:

- same logical document;
- same exact SHA-256 image hash;
- same reliable near-duplicate group reported by the organization audit;
- same FATURA template family;
- same source stem shared across CORU KIE and QA components.

Each unioned group receives one deterministic public split assignment. Private
documents bypass public assignment and enter private_test only.

The executed assignment contains:

| Split | Pages | Rotation profile | Rotations |
|-------|------:|------------------|----------:|
| Train | 280 | 20 interior angles | 5,600 |
| Validation | 61 | 16 boundary/interior angles | 976 |
| Test | 59 | 16 boundary/interior angles | 944 |
| Private test | 203 | 45, 135, 225, 315 degrees | 812 |

The split report records zero page, document, split-group, or private/public
leakage.

## Page preparation and rotations

Public images remain read-only references to data/raw/public. Private PDFs are
rendered at 200 DPI as RGB PNGs with anonymous page IDs under
data/processed/private/page_images. Only the ignored private operational
manifest maps those IDs to real private paths.

Before rotation, a page is EXIF-normalized, converted to RGB, and resized
without cropping so its longest side is at most 1,024 pixels. Positive rotation
is counterclockwise. Pillow applies bicubic rotation with an expanded canvas
and white fill.

The full run produced 8,332 valid rotations with zero generation failures and
exact balance of 2,083 rows in each zone.

## Feature representation

Each rotated image is fitted without distortion into a 128 by 128 white canvas.
Histogram equalization precedes the configured feature extraction.

The 1,957-value vector concatenates:

| Feature group | Dimensions | Purpose |
|---------------|-----------:|---------|
| OpenCV HOG | 1,764 | Local gradient layout |
| Hough and line statistics | 48 | Dominant line orientations and edge density |
| Projection profiles | 136 | Horizontal/vertical ink distribution |
| Directional edges | 4 | Axis and diagonal gradient energy |
| Geometry | 5 | Aspect ratio, ink density, edge density |

Feature extraction performs no fitting. Split NPZ caches store feature,
manifest, and configuration hashes. The completed run produced 8,332 finite
vectors with no failures.

## Train-only preprocessing

StandardScaler fits the 5,600 public training vectors only. PCA then fits those
scaled training rows only, reducing 1,957 dimensions to the requested 128
components. The fitted PCA retained 84.22% cumulative explained variance.

The scaler and PCA transform validation, test, and private_test only after
fitting. Provenance records the training split, training rotation-ID digest,
private fit count of zero, configuration hashes, and artifact reload checks.

## K-Means and mapping

K-Means fits four clusters on the transformed public training matrix without
using zone labels. Configuration: seed 42, n_init 20, maximum 300 iterations,
and tolerance 0.0001.

The fitted cluster sizes are 1,112, 1,114, 1,686, and 1,688. Because raw
cluster IDs are arbitrary, a 4 by 4 cluster/true-zone count matrix is passed to
Hungarian assignment after fitting. Training labels are used only for this
one-to-one mapping:

    C0 -> Z1
    C1 -> Z2
    C2 -> Z4
    C3 -> Z3

The mapping is fixed for all later splits. Evaluation reports raw clustering
metrics separately from mapped classification metrics. Centroid confidence is
the normalized margin between nearest and second-nearest centroid distances;
it is a heuristic, not a calibrated probability.

## Observed K-Means behavior

Mapped accuracy is 50.00% on training, 37.81% on validation, and 37.92% on
public test. Public test ARI is 0.0871 and NMI is 0.1035. Boundary confidence is
low, and the confusion matrices show substantial quadrant ambiguity.

This behavior is consistent with a key limitation of the design: visual
document orientation can be symmetric over 180 degrees or dominated by content
and template style. Four unsupervised clusters are not guaranteed to represent
four predefined angular quadrants.

## Exact-angle estimator

The mapped K-Means zone restricts a coarse-to-fine search:

- 2-degree coarse step inside the half-open predicted zone;
- 0.25-degree fine step within a 3-degree window around the best coarse result;
- 128-pixel safely padded scoring canvas;
- projection, gradient, Hough, minimum-area-rectangle, and content-preservation
  scores;
- explicit failure for insufficient ink, insufficient edges, invalid images,
  or non-finite scores.

For a candidate angle, the estimator applies the signed negative candidate to
the scoring image. It records the best and runner-up score margin, evidence
density, confidence, and pixel-derived orientation score before and after
correction.

The executed evaluation shows that this design is not reliable:

- public: 1,920 attempts, 16 hard failures, 1,904 low-confidence estimates,
  zero reliable estimates, 89.74-degree circular MAE, 90-degree median error;
- private: 812 aggregate-only attempts, eight hard failures, 804
  low-confidence estimates, zero reliable estimates, 90.00-degree MAE.

Low-confidence estimates remain in primary error metrics. Hard failures have
no fabricated angle. A higher corrected-orientation score is diagnostic only;
it does not compensate for a 90-degree circular error.

## Privacy design

Public metadata may contain anonymous private page IDs but never real private
filenames or source paths. Public K-Means and exact-angle reports contain
row-level public results. Private evaluation writes aggregates only:

- no private prediction CSV rows;
- no private document identifiers;
- no private path or filename;
- no private preview or corrected image.

The privacy scanner checks public text artifacts against the real private
filenames held in the ignored inventory. Repository visibility and ignore
rules still require manual confirmation before sharing.

## Artifacts

| Root | Main artifacts |
|------|----------------|
| data/metadata | Page, split, rotation, and feature manifests and summaries |
| data/splits | Per-split selected-page CSV files |
| data/processed/rotated_images/full | Generated bounded full-profile PNGs |
| data/processed/features/full | Raw and transformed NPZ caches |
| models/kmeans_rotation | Feature config, scaler, PCA, K-Means, mapping, training summaries |
| reports/rotation_preparation | Page, split, smoke, disk, and full-run summaries |
| reports/kmeans_evaluation | Metrics, public predictions, boundary tables, confusion matrices |
| reports/angle_estimation | Public angle rows, boundary/group metrics, private aggregates |
| reports/verification | Raw baseline and 20-check full-pipeline verification |

## Stop gates

The workflow stops on:

- raw count, size, or sampled-hash drift;
- missing or unreadable selected pages;
- page, document, duplicate-group, or private/public leakage;
- insufficient disk reserve;
- mismatched rotation manifest and physical artifacts;
- stale configuration or manifest hashes;
- missing or mismatched embedded page/rotation PNG provenance;
- missing, non-finite, or inconsistent feature vectors;
- private rows in fit data;
- incompatible saved artifacts or failed reload checks;
- fewer than four non-empty K-Means clusters;
- non-bijective cluster-to-zone mapping;
- private filename leakage.

Page and rotation PNGs store the source SHA-256 and applicable configuration
hash in PNG text metadata. Reuse requires valid RGB/PNG structure plus exact
provenance agreement; otherwise the pipeline atomically regenerates the
artifact. The privacy scan includes committable code, tests, docs, root config,
metadata, reports, and model summaries while excluding explicitly private
operational manifests.

Exact-angle low confidence does not abort evaluation because it is a measured
quality outcome. It is instead reported explicitly and prevents the baseline
from being described as a usable correction method.

## Alternatives and next design decision

The current K-Means baseline honors a literal unsupervised interpretation of
“clustering.” Plausible alternatives include a deterministic orientation
estimator, a supervised four-way classifier, direct angle regression, or a
coarse-zone classifier followed by a specialized skew estimator.

Do not choose among them until the professor confirms whether K-Means is
mandatory, how boundaries are assigned, and what “pre-model” and the official
evaluation protocol mean. The current held-out metrics justify a method review,
not parameter-only tuning presented as completion.
