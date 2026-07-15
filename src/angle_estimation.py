"""Zone-guided exact-angle estimation and circular-error evaluation.

The K-Means model supplies a mapped 90-degree zone.  This module then searches
only that zone for the counterclockwise angle already applied to a document.
For every candidate it applies the signed correction (``-candidate``) to a
small, safely padded grayscale image and scores the corrected orientation with
projection-profile, gradient, Hough-line, and content-preservation evidence.

No estimator is fitted here.  Public validation/test rows may be written to
reports, while private-test results are deliberately reduced to aggregate
metrics before any artifact is persisted.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import joblib
import matplotlib
import numpy as np

from . import config as cfgmod
from .orientation_features import load_feature_split
from .rotation_common import (
    BOUNDARY_ANGLES,
    ArtifactMismatchError,
    LeakageError,
    RotationPipelineError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    circular_angular_error,
    configuration_hash,
    ensure_not_raw_output,
    normalize_angle,
    parse_dataset_filter,
    read_csv_rows,
    signed_correction_angle,
)
from .rotation_model import MODEL_VERSION, centroid_confidence

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANGLE_ESTIMATION_VERSION = "zone-guided-angle-v1"
DEFAULT_TOLERANCES = (1.0, 2.0, 3.0, 5.0, 10.0)
ANGLE_METHOD = "combined_hough_projection_gradient_min_area_rect"
PUBLIC_PREDICTION_COLUMNS = [
    "rotation_id",
    "document_id",
    "page_id",
    "dataset",
    "dataset_component",
    "document_type",
    "project_split",
    "true_angle",
    "true_zone",
    "raw_cluster_id",
    "predicted_zone",
    "zone_correct",
    "zone_confidence",
    "estimated_angle",
    "correction_angle",
    "correction_angle_normalized",
    "circular_error",
    "residual_angle",
    "status",
    "failure_reason",
    "method",
    "fallback_used",
    "fallback_reason",
    "angle_confidence",
    "combined_confidence",
    "reliable",
    "ink_fraction",
    "edge_fraction",
    "best_score",
    "runner_up_score",
    "projection_score",
    "gradient_score",
    "hough_score",
    "content_preservation_score",
    "min_area_rect_score",
    "original_orientation_score",
    "corrected_orientation_score",
    "orientation_score_delta",
    "correction_direction_valid",
    "corrected_within_5_degrees",
    "image_path",
]


@dataclass(frozen=True)
class AngleEstimate:
    """Serializable result for one zone-guided exact-angle attempt."""

    cluster_id: int | None
    predicted_zone: int
    estimated_angle: float | None
    correction_angle: float | None
    correction_angle_normalized: float | None
    status: str
    failure_reason: str
    method: str
    fallback_used: bool
    fallback_reason: str
    confidence: float
    reliable: bool
    ink_fraction: float
    edge_fraction: float
    best_score: float | None
    runner_up_score: float | None
    projection_score: float | None
    gradient_score: float | None
    hough_score: float | None
    content_preservation_score: float | None
    min_area_rect_score: float | None
    original_orientation_score: float | None
    corrected_orientation_score: float | None
    orientation_score_delta: float | None
    coarse_candidate_count: int
    fine_candidate_count: int
    hough_line_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ScoringContext:
    gray: np.ndarray
    ink: np.ndarray
    gradient_angles: np.ndarray
    gradient_weights: np.ndarray
    hough_angles: np.ndarray
    hough_weights: np.ndarray
    rect_angle: float | None
    ink_fraction: float
    edge_fraction: float


def estimate_exact_angle(
    image: str | Path | np.ndarray,
    predicted_zone: int,
    angle_cfg: Mapping[str, Any] | None = None,
    *,
    zone_confidence: float | None = None,
    cluster_id: int | None = None,
) -> dict[str, Any]:
    """Estimate an exact angle inside a mapped K-Means zone.

    Parameters
    ----------
    image:
        Filesystem path or grayscale/BGR/RGB image array.
    predicted_zone:
        Mapped K-Means zone in ``{1, 2, 3, 4}``.
    angle_cfg:
        The ``angle_estimation`` configuration mapping.
    zone_confidence:
        Optional centroid-margin confidence.  When supplied, reliability uses
        the lower of zone and exact-angle confidence.

    Returns
    -------
    dict
        JSON-serializable estimate, correction, diagnostics, and explicit
        ``success``/``low_confidence``/``failure`` status.
    """
    cfg = _normalized_angle_config(angle_cfg or {})
    if predicted_zone not in {1, 2, 3, 4}:
        raise ValueError(f"predicted_zone must be 1-4, got {predicted_zone!r}")
    gray = _load_gray(image)
    context = _prepare_scoring_context(gray, cfg)
    if context.ink_fraction < cfg["minimum_ink_fraction"]:
        return _failure_estimate(
            predicted_zone,
            "insufficient_ink",
            context,
            zone_confidence=zone_confidence,
            cluster_id=cluster_id,
        ).to_dict()
    if context.edge_fraction < cfg["minimum_edge_fraction"]:
        return _failure_estimate(
            predicted_zone,
            "insufficient_edges",
            context,
            zone_confidence=zone_confidence,
            cluster_id=cluster_id,
        ).to_dict()

    low, high = _zone_bounds(predicted_zone)
    coarse_candidates = _candidate_grid(low, high, cfg["coarse_step_degrees"])
    coarse_scores = [_score_candidate(context, value, cfg) for value in coarse_candidates]
    if not coarse_scores or not all(math.isfinite(item["total"]) for item in coarse_scores):
        return _failure_estimate(
            predicted_zone,
            "non_finite_candidate_scores",
            context,
            zone_confidence=zone_confidence,
            cluster_id=cluster_id,
        ).to_dict()
    coarse_best = coarse_scores[int(np.argmax([item["total"] for item in coarse_scores]))]
    fine_low = max(low, coarse_best["angle"] - cfg["fine_window_degrees"])
    fine_high = min(high, coarse_best["angle"] + cfg["fine_window_degrees"])
    fine_candidates = _candidate_grid(
        fine_low,
        fine_high,
        cfg["fine_step_degrees"],
        include_high=not math.isclose(fine_high, high),
    )
    if not fine_candidates:
        fine_candidates = [coarse_best["angle"]]
    fine_scores = [_score_candidate(context, value, cfg) for value in fine_candidates]
    ranked = sorted(fine_scores, key=lambda item: (-item["total"], item["angle"]))
    best = ranked[0]
    runner_up = _separated_runner_up(ranked, best["angle"], cfg["fine_step_degrees"])
    confidence = _estimate_confidence(best, runner_up, context, cfg)
    if zone_confidence is not None:
        confidence = min(confidence, float(np.clip(zone_confidence, 0.0, 1.0)))
    reliable = confidence >= cfg["reliability_threshold"]
    status = "success" if reliable else "low_confidence"
    estimated = normalize_angle(best["angle"])
    correction = signed_correction_angle(estimated)

    # This is an image-level validation, not merely angle arithmetic: apply the
    # reported correction and score the corrected pixels at zero residual.
    corrected = _rotate_square(context.gray, correction, border_value=255)
    corrected_context = _prepare_scoring_context(corrected, cfg, already_square=True)
    corrected_score = _score_candidate(corrected_context, 0.0, cfg)["total"]
    original_score = _score_candidate(context, 0.0, cfg)["total"]
    missing_evidence = []
    if not len(context.hough_angles):
        missing_evidence.append("no_hough_lines")
    if context.rect_angle is None:
        missing_evidence.append("no_min_area_rect_angle")
    estimate = AngleEstimate(
        cluster_id=cluster_id,
        predicted_zone=predicted_zone,
        estimated_angle=estimated,
        correction_angle=correction,
        correction_angle_normalized=normalize_angle(correction),
        status=status,
        failure_reason="" if reliable else "confidence_below_threshold",
        method=ANGLE_METHOD,
        fallback_used=bool(missing_evidence),
        fallback_reason=",".join(missing_evidence),
        confidence=confidence,
        reliable=reliable,
        ink_fraction=context.ink_fraction,
        edge_fraction=context.edge_fraction,
        best_score=float(best["total"]),
        runner_up_score=float(runner_up["total"]),
        projection_score=float(best["projection"]),
        gradient_score=float(best["gradient"]),
        hough_score=float(best["hough"]),
        content_preservation_score=float(best["preservation"]),
        min_area_rect_score=float(best["min_area_rect"]),
        original_orientation_score=float(original_score),
        corrected_orientation_score=float(corrected_score),
        orientation_score_delta=float(corrected_score - original_score),
        coarse_candidate_count=len(coarse_candidates),
        fine_candidate_count=len(fine_candidates),
        hough_line_count=int(len(context.hough_angles)),
    )
    return estimate.to_dict()


def apply_angle_correction(
    image: np.ndarray,
    estimated_angle: float,
    *,
    expand: bool = True,
    border_value: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    """Apply the documented clockwise correction ``-estimated_angle``.

    This public helper is useful to validate downstream correction semantics.
    It never writes the corrected image to disk.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    correction = signed_correction_angle(estimated_angle)
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, correction, 1.0)
    if expand:
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        target_width = max(1, int(math.ceil(height * sine + width * cosine)))
        target_height = max(1, int(math.ceil(height * cosine + width * sine)))
        matrix[0, 2] += target_width / 2.0 - center[0]
        matrix[1, 2] += target_height / 2.0 - center[1]
    else:
        target_width, target_height = width, height
    return cv2.warpAffine(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def evaluate_angle_estimation(
    cfg: Mapping[str, Any],
    *,
    profile: str | None = None,
    limit: int = 0,
    datasets: str | Sequence[str] | None = None,
    splits: str | Sequence[str] | None = None,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run mapped-zone prediction followed by exact-angle evaluation.

    The function is intentionally CLI-friendly: every argument is scalar or a
    comma-separated/list filter and the returned summary is JSON-serializable.
    Private results are aggregate-only in both the return value and files.
    """
    del force  # Evaluation is deterministic and its atomic reports are replaced.
    started = time.perf_counter()
    root = cfgmod.project_root(cfg)
    metadata_root = cfgmod.resolve_path(cfg, "metadata")
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    report_root = cfgmod.resolve_path(cfg, "reports") / "angle_estimation"
    ensure_not_raw_output(report_root, root)
    ensure_not_raw_output(model_root, root)

    angle_cfg = _normalized_angle_config(cfg.get("angle_estimation", {}))
    feature_config = _load_json(model_root / "feature_config.json")
    preprocessing = _load_json(model_root / "preprocessing_summary.json")
    training = _load_json(model_root / "training_summary.json")
    mapping_payload = _load_json(model_root / "cluster_to_zone.json")
    if mapping_payload.get("training_hash") != training.get("training_hash"):
        raise ArtifactMismatchError("cluster mapping and K-Means training hashes differ")
    if training.get("model_version") != MODEL_VERSION:
        raise ArtifactMismatchError("K-Means model version is incompatible")
    model = joblib.load(model_root / "kmeans.joblib")
    if getattr(model, "n_clusters", None) != 4:
        raise ArtifactMismatchError("saved K-Means does not use four clusters")
    mapping = {int(key): int(value) for key, value in mapping_payload["mapping"].items()}
    if set(mapping) != {0, 1, 2, 3} or set(mapping.values()) != {1, 2, 3, 4}:
        raise ArtifactMismatchError("saved cluster-to-zone mapping is not bijective")

    active_profile = profile or str(feature_config["profile"])
    if active_profile != str(feature_config["profile"]):
        raise ArtifactMismatchError(
            "requested angle-estimation profile does not match the trained feature profile"
        )
    rotation_manifest_path = metadata_root / "rotation_manifest.csv"
    manifest_rows = [
        row
        for row in read_csv_rows(rotation_manifest_path)
        if row.get("generation_status") == "success"
        and row.get("generation_profile") == active_profile
    ]
    if not manifest_rows:
        raise ValueError(f"no successful rotation rows exist for profile {active_profile!r}")
    rotation_by_id = {row["rotation_id"]: row for row in manifest_rows}
    if len(rotation_by_id) != len(manifest_rows):
        raise RotationPipelineError("rotation manifest contains duplicate rotation IDs")

    dataset_filter = parse_dataset_filter(datasets)
    split_filter = parse_dataset_filter(splits)
    available_splits = _feature_splits(metadata_root)
    selected_splits = _select_evaluation_splits(available_splits, split_filter)
    if not selected_splits:
        raise ValueError("no feature splits match the requested evaluation filter")

    tasks: list[dict[str, Any]] = []
    for split in selected_splits:
        values = load_feature_split(cfg, split, transformed=True)
        if str(values["preprocessing_hash"][0]) != preprocessing["preprocessing_hash"]:
            raise ArtifactMismatchError(f"transformed {split} cache has a stale preprocessing hash")
        if split == "private_test" and not np.all(values["private"] != 0):
            raise LeakageError("public rows are present in private_test features")
        if split != "private_test" and np.any(values["private"] != 0):
            raise LeakageError(f"private rows are present in public split {split}")
        clusters = model.predict(values["X"])
        distances = model.transform(values["X"])
        _, _, zone_confidences = centroid_confidence(distances)
        for index, rotation_value in enumerate(values["rotation_ids"]):
            rotation_id = str(rotation_value)
            source = rotation_by_id.get(rotation_id)
            if source is None:
                raise ArtifactMismatchError(f"feature row {rotation_id} is absent from the rotation manifest")
            dataset = str(values["datasets"][index])
            if dataset_filter is not None and dataset.lower() not in dataset_filter:
                continue
            raw_cluster = int(clusters[index])
            tasks.append({
                "rotation_id": rotation_id,
                "document_id": str(values["document_ids"][index]),
                "page_id": str(values["page_ids"][index]),
                "dataset": dataset,
                "dataset_component": source.get("dataset_component", ""),
                "document_type": source.get("document_type", ""),
                "project_split": split,
                "true_angle": float(values["true_angles"][index]),
                "true_zone": int(values["true_zones"][index]),
                "raw_cluster_id": raw_cluster,
                "predicted_zone": mapping[raw_cluster],
                "zone_confidence": float(zone_confidences[index]),
                "image_path": source["rotated_image_path"],
                "private": split == "private_test",
            })

    tasks.sort(key=lambda row: (row["project_split"], row["rotation_id"]))
    if limit:
        if limit < 1:
            raise ValueError("limit must be zero or a positive integer")
        tasks = tasks[: int(limit)]
    if not tasks:
        raise ValueError("angle evaluation has no selected rows")

    max_workers = max(1, int(workers or cfg.get("runtime", {}).get("workers", 2)))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_task = {
            pool.submit(
                estimate_exact_angle,
                root / task["image_path"],
                int(task["predicted_zone"]),
                angle_cfg,
                zone_confidence=float(task["zone_confidence"]),
                cluster_id=int(task["raw_cluster_id"]),
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                estimate = future.result()
            except Exception as exc:  # keep a corrupt page from losing the evaluation
                estimate = _exception_failure(task["predicted_zone"], exc)
            records.append(_combine_evaluation_record(task, estimate))
    records.sort(key=lambda row: (row["project_split"], row["rotation_id"]))

    public_records = [
        {key: value for key, value in row.items() if key != "private"}
        for row in records
        if not row["private"]
    ]
    private_records = [
        {key: value for key, value in row.items() if key != "private"}
        for row in records
        if row["private"]
    ]
    public_split_metrics = {
        split: _metric_summary([row for row in public_records if row["project_split"] == split])
        for split in sorted({row["project_split"] for row in public_records})
    }
    public_overall = _metric_summary(public_records)
    private_aggregate = _private_aggregate(private_records) if private_records else None
    by_zone_rows = _group_metric_rows(public_records, ("project_split", "true_zone"))
    by_dataset_rows = _group_metric_rows(public_records, ("project_split", "dataset"))
    by_document_type_rows = _group_metric_rows(public_records, ("project_split", "document_type"))
    by_angle_rows = _group_metric_rows(public_records, ("project_split", "true_angle"))
    boundary_rows = _boundary_metric_rows(public_records)
    correction_rows = _correction_validation_rows(public_records)

    evaluation_hash = configuration_hash({
        "version": ANGLE_ESTIMATION_VERSION,
        "angle_estimation": angle_cfg,
        "profile": active_profile,
        "training_hash": training["training_hash"],
        "preprocessing_hash": preprocessing["preprocessing_hash"],
        "selected_splits": selected_splits,
        "dataset_filter": sorted(dataset_filter) if dataset_filter else None,
        "limit": int(limit),
    })
    artifact = {
        "version": ANGLE_ESTIMATION_VERSION,
        "evaluation_hash": evaluation_hash,
        "profile": active_profile,
        "angle_configuration": angle_cfg,
        "zone_model_version": training["model_version"],
        "zone_training_hash": training["training_hash"],
        "zone_mapping": {str(key): value for key, value in sorted(mapping.items())},
        "search_scope": "mapped K-Means zone only",
        "method": ANGLE_METHOD,
        "fallback_policy": "continue with remaining geometric evidence and mark fallback_used",
        "rotation_direction": "counterclockwise",
        "correction_semantics": "correction_angle = -estimated_angle",
        "private_row_outputs": False,
    }
    atomic_write_json(model_root / "angle_estimation_config.json", artifact)

    metrics_payload = {
        "version": ANGLE_ESTIMATION_VERSION,
        "evaluation_hash": evaluation_hash,
        "profile": active_profile,
        "selected_splits": selected_splits,
        "selected_task_count": len(tasks),
        "public_overall": public_overall,
        "public_by_split": public_split_metrics,
        "private_test_aggregate": private_aggregate,
        "private_row_outputs_written": 0,
        "tolerances_degrees": list(DEFAULT_TOLERANCES),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(report_root / "metrics.json", metrics_payload)
    atomic_write_json(report_root / "angle_metrics.json", metrics_payload)
    atomic_write_csv(report_root / "public_predictions.csv", public_records, PUBLIC_PREDICTION_COLUMNS)
    atomic_write_csv(report_root / "angle_predictions.csv", public_records, PUBLIC_PREDICTION_COLUMNS)
    atomic_write_csv(
        report_root / "metrics_by_zone.csv",
        by_zone_rows,
        _group_metric_columns(("project_split", "true_zone")),
    )
    atomic_write_csv(
        report_root / "angle_error_by_zone.csv",
        by_zone_rows,
        _group_metric_columns(("project_split", "true_zone")),
    )
    atomic_write_csv(
        report_root / "metrics_by_dataset.csv",
        by_dataset_rows,
        _group_metric_columns(("project_split", "dataset")),
    )
    atomic_write_csv(
        report_root / "metrics_by_document_type.csv",
        by_document_type_rows,
        _group_metric_columns(("project_split", "document_type")),
    )
    atomic_write_csv(
        report_root / "angle_error_by_angle.csv",
        by_angle_rows,
        _group_metric_columns(("project_split", "true_angle")),
    )
    atomic_write_csv(
        report_root / "boundary_case_results.csv",
        boundary_rows,
        _group_metric_columns(("project_split", "true_angle")),
    )
    atomic_write_csv(
        report_root / "corrected_orientation_validation.csv",
        correction_rows,
        [
            "project_split",
            "sample_count",
            "estimated_count",
            "correction_direction_valid_count",
            "correction_direction_valid_rate",
            "mean_original_orientation_score",
            "mean_corrected_orientation_score",
            "mean_orientation_score_delta",
            "corrected_within_5_degrees_rate",
        ],
    )
    atomic_write_text(
        report_root / "angle_estimation_summary.md",
        _summary_markdown(metrics_payload),
    )
    _write_angle_error_histogram(report_root, public_records)
    return {
        "evaluation_hash": evaluation_hash,
        "profile": active_profile,
        "selected_task_count": len(tasks),
        "public_prediction_count": len(public_records),
        "private_test_sample_count": len(private_records),
        "public_overall": public_overall,
        "public_by_split": public_split_metrics,
        "private_test_aggregate": private_aggregate,
        "report_directory": str(report_root.relative_to(root)),
        "artifact_path": str((model_root / "angle_estimation_config.json").relative_to(root)),
        "elapsed_seconds": metrics_payload["elapsed_seconds"],
    }


def _normalized_angle_config(values: Mapping[str, Any]) -> dict[str, Any]:
    cfg = {
        "methods": [str(value) for value in values.get(
            "methods", ["hough", "projection_profile", "min_area_rect"]
        )],
        "coarse_step_degrees": float(values.get("coarse_step_degrees", 2.0)),
        "fine_step_degrees": float(values.get("fine_step_degrees", 0.25)),
        "fine_window_degrees": float(values.get("fine_window_degrees", 3.0)),
        "scoring_size": int(values.get("scoring_size", 128)),
        "reliability_threshold": float(values.get("reliability_threshold", 0.5)),
        "minimum_ink_fraction": float(values.get("minimum_ink_fraction", 0.005)),
        "minimum_edge_fraction": float(values.get("minimum_edge_fraction", 0.002)),
    }
    if cfg["coarse_step_degrees"] <= 0 or cfg["coarse_step_degrees"] > 45:
        raise ValueError("coarse_step_degrees must be in (0, 45]")
    if cfg["fine_step_degrees"] <= 0 or cfg["fine_step_degrees"] > cfg["coarse_step_degrees"]:
        raise ValueError("fine_step_degrees must be positive and no larger than coarse_step_degrees")
    if cfg["fine_window_degrees"] < cfg["fine_step_degrees"]:
        raise ValueError("fine_window_degrees must be at least one fine step")
    if cfg["scoring_size"] < 64:
        raise ValueError("scoring_size must be at least 64 pixels")
    if not 0 <= cfg["reliability_threshold"] <= 1:
        raise ValueError("reliability_threshold must be between zero and one")
    for name in ("minimum_ink_fraction", "minimum_edge_fraction"):
        if not 0 <= cfg[name] < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    return cfg


def _load_gray(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded = np.fromfile(path, dtype=np.uint8)
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("unable to decode image")
        return gray
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a path or non-empty NumPy array")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        # Orientation evidence is color-order insensitive for grayscale conversion.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _prepare_scoring_context(
    gray: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    already_square: bool = False,
) -> _ScoringContext:
    size = int(cfg["scoring_size"])
    if already_square and gray.shape == (size, size):
        square = gray.copy()
    else:
        square = _safe_square(gray, size)
    _, ink = cv2.threshold(square, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (ink > 0).astype(np.uint8)
    edges = cv2.Canny(square, 50, 150)
    ink_fraction = float(ink.mean())
    edge_fraction = float(np.count_nonzero(edges) / edges.size)

    dx = cv2.Sobel(square, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(square, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, gradient_angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    gradient_mask = magnitude > max(8.0, float(np.percentile(magnitude, 65)))
    gradient_angles = np.mod(gradient_angle[gradient_mask], 180.0).astype(np.float32)
    gradient_weights = magnitude[gradient_mask].astype(np.float32)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(12, size // 10),
        minLineLength=max(10, size // 9),
        maxLineGap=max(4, size // 20),
    )
    if lines is None:
        hough_angles = np.empty(0, dtype=np.float32)
        hough_weights = np.empty(0, dtype=np.float32)
    else:
        segments = lines.reshape(-1, 4).astype(np.float32)
        horizontal = segments[:, 2] - segments[:, 0]
        vertical = segments[:, 3] - segments[:, 1]
        hough_angles = np.mod(np.degrees(np.arctan2(vertical, horizontal)), 180.0).astype(np.float32)
        hough_weights = np.hypot(horizontal, vertical).astype(np.float32)
    points = np.column_stack(np.nonzero(ink))
    rect_angle = _long_axis_angle(points) if len(points) >= 8 else None
    return _ScoringContext(
        gray=square,
        ink=ink,
        gradient_angles=gradient_angles,
        gradient_weights=gradient_weights,
        hough_angles=hough_angles,
        hough_weights=hough_weights,
        rect_angle=rect_angle,
        ink_fraction=ink_fraction,
        edge_fraction=edge_fraction,
    )


def _safe_square(gray: np.ndarray, size: int) -> np.ndarray:
    # Crop only uniform outer background introduced by expand=True.  A generous
    # 250 threshold preserves light document regions and all actual ink.
    foreground = np.argwhere(gray < 250)
    if len(foreground):
        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0) + 1
        gray = gray[y0:y1, x0:x1]
    height, width = gray.shape
    # Fit inside the inscribed square of every possible rotation, so candidate
    # corrections cannot crop content and preservation remains measurable.
    safe_extent = max(8, int((size - 4) / math.sqrt(2.0)))
    scale = min(safe_extent / max(width, 1), safe_extent / max(height, 1))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (target_width, target_height), interpolation=interpolation)
    output = np.full((size, size), 255, dtype=np.uint8)
    x = (size - target_width) // 2
    y = (size - target_height) // 2
    output[y:y + target_height, x:x + target_width] = resized
    return output


def _long_axis_angle(points_yx: np.ndarray) -> float | None:
    if len(points_yx) < 8:
        return None
    points_xy = points_yx[:, ::-1].astype(np.float32)
    rectangle = cv2.minAreaRect(points_xy)
    corners = cv2.boxPoints(rectangle)
    vectors = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.any(lengths > 0):
        return None
    vector = vectors[int(np.argmax(lengths))]
    return float(normalize_angle(np.degrees(np.arctan2(vector[1], vector[0]))) % 180.0)


def _zone_bounds(zone: int) -> tuple[float, float]:
    return float((zone - 1) * 90), float(zone * 90)


def _candidate_grid(
    low: float,
    high: float,
    step: float,
    *,
    include_high: bool = False,
) -> list[float]:
    if high < low:
        return []
    epsilon = min(step * 1e-5, 1e-6)
    stop = high + epsilon if include_high else high - epsilon
    if stop < low:
        return [float(low)]
    count = int(math.floor((stop - low) / step)) + 1
    values = [float(low + index * step) for index in range(count)]
    if include_high and (not values or high - values[-1] > epsilon):
        values.append(float(high))
    # A zone's upper boundary belongs to the next zone.  Keep candidates inside
    # the half-open interval, including Zone 4 before normalization to zero.
    zone_high = math.ceil((low + epsilon) / 90.0) * 90.0
    if math.isclose(low % 90.0, 0.0, abs_tol=epsilon):
        zone_high = low + 90.0
    return sorted({min(value, zone_high - 1e-6) for value in values if value < zone_high + epsilon})


def _score_candidate(
    context: _ScoringContext,
    candidate: float,
    cfg: Mapping[str, Any],
) -> dict[str, float]:
    corrected_ink = _rotate_square(context.ink, -candidate, border_value=0, nearest=True)
    horizontal = corrected_ink.mean(axis=1).astype(np.float32)
    vertical = corrected_ink.mean(axis=0).astype(np.float32)
    horizontal_energy = float(np.mean(np.abs(np.diff(horizontal)))) if len(horizontal) > 1 else 0.0
    vertical_energy = float(np.mean(np.abs(np.diff(vertical)))) if len(vertical) > 1 else 0.0
    coefficient = float(horizontal.std() / (horizontal.mean() + 0.01))
    peak_contrast = float(np.percentile(horizontal, 95) - np.median(horizontal))
    anisotropy = horizontal_energy / (horizontal_energy + vertical_energy + 1e-8)
    projection = float(np.clip(
        0.40 * np.tanh(1.5 * coefficient)
        + 0.35 * np.tanh(8.0 * peak_contrast)
        + 0.25 * anisotropy,
        0.0,
        1.0,
    ))

    gradient = _direction_support(
        context.gradient_angles,
        context.gradient_weights,
        candidate,
        favor_vertical=True,
    )
    hough = _direction_support(
        context.hough_angles,
        context.hough_weights,
        candidate,
        favor_vertical=False,
    )
    if not len(context.hough_angles):
        hough = gradient
    preservation = float(np.clip(
        corrected_ink.sum() / max(float(context.ink.sum()), 1.0), 0.0, 1.0
    ))
    rect_support = 0.5
    if context.rect_angle is not None:
        # OpenCV image coordinates increase downward.  A visual clockwise
        # correction of ``-candidate`` therefore adds ``candidate`` to angles
        # measured with atan2(dy, dx) in image coordinates.
        corrected_rect = math.radians((context.rect_angle + candidate) % 90.0)
        rect_support = float(abs(math.cos(2.0 * corrected_rect)))
    methods = {value.lower() for value in cfg.get("methods", [])}
    weights = {
        "projection": 0.45 if "projection_profile" in methods else 0.30,
        "gradient": 0.23,
        "hough": 0.20 if "hough" in methods else 0.08,
        "preservation": 0.07,
        "min_area_rect": 0.05 if "min_area_rect" in methods else 0.0,
    }
    weight_total = sum(weights.values())
    total = (
        weights["projection"] * projection
        + weights["gradient"] * gradient
        + weights["hough"] * hough
        + weights["preservation"] * preservation
        + weights["min_area_rect"] * rect_support
    ) / weight_total
    return {
        "angle": float(candidate),
        "total": float(total),
        "projection": projection,
        "gradient": gradient,
        "hough": hough,
        "preservation": preservation,
        "min_area_rect": rect_support,
    }


def _direction_support(
    angles: np.ndarray,
    weights: np.ndarray,
    candidate: float,
    *,
    favor_vertical: bool,
) -> float:
    if not len(angles) or float(weights.sum()) <= 0:
        return 0.0
    # See the image-coordinate convention note in _score_candidate.
    corrected = np.radians(np.mod(angles + candidate, 180.0))
    axis = np.abs(np.cos(2.0 * corrected))
    favored = np.sin(corrected) ** 2 if favor_vertical else np.cos(corrected) ** 2
    support = 0.70 * axis + 0.30 * favored
    return float(np.clip(np.average(support, weights=weights), 0.0, 1.0))


def _rotate_square(
    image: np.ndarray,
    angle: float,
    *,
    border_value: int,
    nearest: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _separated_runner_up(
    ranked: Sequence[Mapping[str, float]],
    best_angle: float,
    fine_step: float,
) -> Mapping[str, float]:
    minimum_separation = max(1.0, 4.0 * fine_step)
    for candidate in ranked[1:]:
        if abs(float(candidate["angle"]) - best_angle) >= minimum_separation:
            return candidate
    return ranked[1] if len(ranked) > 1 else ranked[0]


def _estimate_confidence(
    best: Mapping[str, float],
    runner_up: Mapping[str, float],
    context: _ScoringContext,
    cfg: Mapping[str, Any],
) -> float:
    margin = max(0.0, float(best["total"]) - float(runner_up["total"]))
    margin_quality = float(np.clip(margin / 0.06, 0.0, 1.0))
    evidence_quality = min(
        1.0,
        context.ink_fraction / max(4.0 * cfg["minimum_ink_fraction"], 1e-8),
        context.edge_fraction / max(4.0 * cfg["minimum_edge_fraction"], 1e-8),
    )
    line_quality = min(1.0, len(context.hough_angles) / 12.0)
    confidence = (
        0.45 * float(np.clip(best["total"], 0.0, 1.0))
        + 0.30 * margin_quality
        + 0.15 * evidence_quality
        + 0.10 * line_quality
    )
    return float(np.clip(confidence, 0.0, 1.0))


def _failure_estimate(
    predicted_zone: int,
    reason: str,
    context: _ScoringContext,
    *,
    zone_confidence: float | None,
    cluster_id: int | None = None,
) -> AngleEstimate:
    del zone_confidence
    return AngleEstimate(
        cluster_id=cluster_id,
        predicted_zone=predicted_zone,
        estimated_angle=None,
        correction_angle=None,
        correction_angle_normalized=None,
        status="failure",
        failure_reason=reason,
        method=ANGLE_METHOD,
        fallback_used=True,
        fallback_reason=reason,
        confidence=0.0,
        reliable=False,
        ink_fraction=context.ink_fraction,
        edge_fraction=context.edge_fraction,
        best_score=None,
        runner_up_score=None,
        projection_score=None,
        gradient_score=None,
        hough_score=None,
        content_preservation_score=None,
        min_area_rect_score=None,
        original_orientation_score=None,
        corrected_orientation_score=None,
        orientation_score_delta=None,
        coarse_candidate_count=0,
        fine_candidate_count=0,
        hough_line_count=int(len(context.hough_angles)),
    )


def _exception_failure(predicted_zone: int, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        reason = "image_not_found"
    elif isinstance(exc, (ValueError, cv2.error)):
        reason = "invalid_image_or_configuration"
    else:
        reason = "estimation_error"
    return {
        **_failure_estimate(
            int(predicted_zone),
            reason,
            _ScoringContext(
                gray=np.empty((0, 0), dtype=np.uint8),
                ink=np.empty((0, 0), dtype=np.uint8),
                gradient_angles=np.empty(0),
                gradient_weights=np.empty(0),
                hough_angles=np.empty(0),
                hough_weights=np.empty(0),
                rect_angle=None,
                ink_fraction=0.0,
                edge_fraction=0.0,
            ),
            zone_confidence=None,
        ).to_dict(),
        "failure_exception_type": type(exc).__name__,
    }


def _combine_evaluation_record(
    task: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    estimated = estimate.get("estimated_angle")
    error = circular_angular_error(float(estimated), float(task["true_angle"])) if estimated is not None else None
    residual = normalize_angle(float(task["true_angle"]) - float(estimated)) if estimated is not None else None
    correction = estimate.get("correction_angle")
    direction_valid = bool(
        estimated is not None
        and correction is not None
        and math.isclose(float(correction), -normalize_angle(float(estimated)), abs_tol=1e-9)
    )
    return {
        "rotation_id": task["rotation_id"],
        "document_id": task["document_id"],
        "page_id": task["page_id"],
        "dataset": task["dataset"],
        "dataset_component": task["dataset_component"],
        "document_type": task["document_type"],
        "project_split": task["project_split"],
        "true_angle": float(task["true_angle"]),
        "true_zone": int(task["true_zone"]),
        "raw_cluster_id": int(task["raw_cluster_id"]),
        "predicted_zone": int(task["predicted_zone"]),
        "zone_correct": int(task["predicted_zone"] == task["true_zone"]),
        "zone_confidence": float(task["zone_confidence"]),
        "estimated_angle": estimated,
        "correction_angle": correction,
        "correction_angle_normalized": estimate.get("correction_angle_normalized"),
        "circular_error": error,
        "residual_angle": residual,
        "status": estimate["status"],
        "failure_reason": estimate["failure_reason"],
        "method": estimate.get("method", ANGLE_METHOD),
        "fallback_used": int(bool(estimate.get("fallback_used"))),
        "fallback_reason": estimate.get("fallback_reason", ""),
        "angle_confidence": float(estimate["confidence"]),
        "combined_confidence": float(min(task["zone_confidence"], estimate["confidence"])),
        "reliable": int(bool(estimate["reliable"])),
        "ink_fraction": estimate["ink_fraction"],
        "edge_fraction": estimate["edge_fraction"],
        "best_score": estimate.get("best_score"),
        "runner_up_score": estimate.get("runner_up_score"),
        "projection_score": estimate.get("projection_score"),
        "gradient_score": estimate.get("gradient_score"),
        "hough_score": estimate.get("hough_score"),
        "content_preservation_score": estimate.get("content_preservation_score"),
        "min_area_rect_score": estimate.get("min_area_rect_score"),
        "original_orientation_score": estimate.get("original_orientation_score"),
        "corrected_orientation_score": estimate.get("corrected_orientation_score"),
        "orientation_score_delta": estimate.get("orientation_score_delta"),
        "correction_direction_valid": int(direction_valid),
        "corrected_within_5_degrees": int(error is not None and error <= 5.0),
        "image_path": task["image_path"],
        "private": bool(task["private"]),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    estimated_rows = [row for row in rows if row.get("estimated_angle") is not None]
    reliable_rows = [row for row in estimated_rows if bool(row.get("reliable"))]
    errors = np.asarray([row["circular_error"] for row in estimated_rows], dtype=np.float64)
    payload: dict[str, Any] = {
        "attempted_count": attempted,
        "estimated_count": len(estimated_rows),
        "failure_count": attempted - len(estimated_rows),
        "failure_rate": float((attempted - len(estimated_rows)) / attempted) if attempted else None,
        "low_confidence_count": sum(row.get("status") == "low_confidence" for row in rows),
        "reliable_count": len(reliable_rows),
        "reliable_rate": float(len(reliable_rows) / attempted) if attempted else None,
        "zone_accuracy": float(np.mean([row["zone_correct"] for row in rows])) if rows else None,
        "mean_angle_confidence": float(np.mean([row["angle_confidence"] for row in rows])) if rows else None,
        "mean_combined_confidence": float(np.mean([row["combined_confidence"] for row in rows])) if rows else None,
        "circular_mae_degrees": float(errors.mean()) if len(errors) else None,
        "circular_median_ae_degrees": float(np.median(errors)) if len(errors) else None,
        "circular_std_degrees": float(errors.std()) if len(errors) else None,
        "circular_rmse_degrees": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "maximum_circular_error_degrees": float(errors.max()) if len(errors) else None,
    }
    for tolerance in DEFAULT_TOLERANCES:
        token = _tolerance_token(tolerance)
        payload[f"within_{token}_degrees_count"] = int(np.sum(errors <= tolerance)) if len(errors) else 0
        payload[f"within_{token}_degrees_rate"] = float(np.mean(errors <= tolerance)) if len(errors) else None
    return payload


def _private_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _metric_summary(rows)
    estimated = [row for row in rows if row.get("estimated_angle") is not None]
    return {
        "sample_count": len(rows),
        "metrics": metrics,
        "correction_direction_valid_rate": _mean_or_none([
            row["correction_direction_valid"] for row in estimated
        ]),
        "corrected_within_5_degrees_rate": _mean_or_none([
            row["corrected_within_5_degrees"] for row in estimated
        ]),
        "mean_original_orientation_score": _mean_or_none([
            row["original_orientation_score"] for row in estimated
        ]),
        "mean_corrected_orientation_score": _mean_or_none([
            row["corrected_orientation_score"] for row in estimated
        ]),
        "mean_orientation_score_delta": _mean_or_none([
            row["orientation_score_delta"] for row in estimated
        ]),
        "row_level_artifacts_written": 0,
        "identity_or_path_fields_written": 0,
    }


def _group_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        metrics = _metric_summary(group)
        output.append({**dict(zip(keys, group_key, strict=True)), **metrics})
    return output


def _boundary_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {float(value) for value in BOUNDARY_ANGLES}
    selected = [
        row for row in rows
        if any(math.isclose(float(row["true_angle"]), value, abs_tol=1e-9) for value in required)
    ]
    return _group_metric_rows(selected, ("project_split", "true_angle"))


def _correction_validation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["project_split"])].append(row)
    output = []
    for split, group in sorted(grouped.items()):
        estimated = [row for row in group if row.get("estimated_angle") is not None]
        output.append({
            "project_split": split,
            "sample_count": len(group),
            "estimated_count": len(estimated),
            "correction_direction_valid_count": sum(row["correction_direction_valid"] for row in estimated),
            "correction_direction_valid_rate": _mean_or_none([
                row["correction_direction_valid"] for row in estimated
            ]),
            "mean_original_orientation_score": _mean_or_none([
                row["original_orientation_score"] for row in estimated
            ]),
            "mean_corrected_orientation_score": _mean_or_none([
                row["corrected_orientation_score"] for row in estimated
            ]),
            "mean_orientation_score_delta": _mean_or_none([
                row["orientation_score_delta"] for row in estimated
            ]),
            "corrected_within_5_degrees_rate": _mean_or_none([
                row["corrected_within_5_degrees"] for row in estimated
            ]),
        })
    return output


def _write_angle_error_histogram(
    report_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    errors = [
        float(row["circular_error"])
        for row in rows
        if row.get("circular_error") is not None
        and math.isfinite(float(row["circular_error"]))
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if errors:
        bins = np.linspace(0.0, 180.0, 37)
        ax.hist(errors, bins=bins, color="#3568a8", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No successful public angle estimates", ha="center", va="center")
    ax.set_xlim(0.0, 180.0)
    ax.set_xlabel("Circular angular error (degrees)")
    ax.set_ylabel("Public samples")
    ax.set_title("Exact-angle error distribution")
    fig.tight_layout()
    fig.savefig(report_root / "angle_error_histogram.png", dpi=150)
    plt.close(fig)


def _group_metric_columns(keys: Sequence[str]) -> list[str]:
    columns = list(keys) + [
        "attempted_count",
        "estimated_count",
        "failure_count",
        "failure_rate",
        "low_confidence_count",
        "reliable_count",
        "reliable_rate",
        "zone_accuracy",
        "mean_angle_confidence",
        "mean_combined_confidence",
        "circular_mae_degrees",
        "circular_median_ae_degrees",
        "circular_std_degrees",
        "circular_rmse_degrees",
        "maximum_circular_error_degrees",
    ]
    for tolerance in DEFAULT_TOLERANCES:
        token = _tolerance_token(tolerance)
        columns.extend([f"within_{token}_degrees_count", f"within_{token}_degrees_rate"])
    return columns


def _feature_splits(metadata_root: Path) -> list[str]:
    summary = _load_json(metadata_root / "feature_summary.json")
    return sorted(str(value) for value in summary.get("counts_per_split", {}))


def _select_evaluation_splits(
    available: Sequence[str],
    requested: set[str] | None,
) -> list[str]:
    available_set = set(available)
    if requested is not None:
        unknown = requested - available_set
        if unknown:
            raise ValueError(f"requested angle-evaluation splits are unavailable: {sorted(unknown)}")
        return sorted(requested)
    preferred = [value for value in ("validation", "test", "private_test") if value in available_set]
    return preferred or sorted(available_set)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Exact-Angle Estimation Evaluation",
        "",
        "Each angle search is restricted to the mapped K-Means zone. Positive",
        "angles are counterclockwise; the applied correction is the signed",
        "negative estimate. Circular errors use the shortest distance on [0,360).",
        "",
        "Low-confidence estimates remain in primary error metrics and are marked",
        "explicitly. Hard failures have no angle and are reported via failure rate.",
        "",
        "## Public results",
        "",
    ]
    for split, metrics in payload["public_by_split"].items():
        mae = metrics["circular_mae_degrees"]
        median = metrics["circular_median_ae_degrees"]
        lines.append(
            f"- {split}: n={metrics['attempted_count']}, estimated={metrics['estimated_count']}, "
            f"circular MAE={_format_metric(mae)}, median={_format_metric(median)}, "
            f"within 5 degrees={_format_metric(metrics['within_5_degrees_rate'])}"
        )
    if payload.get("private_test_aggregate"):
        private = payload["private_test_aggregate"]
        private_metrics = private["metrics"]
        lines.extend([
            "",
            "## Private test",
            "",
            f"- Aggregate-only sample count: {private['sample_count']}",
            f"- Circular MAE: {_format_metric(private_metrics['circular_mae_degrees'])}",
            f"- Failure rate: {_format_metric(private_metrics['failure_rate'])}",
            "- No private filename, path, identifier, preview, or row prediction was written.",
        ])
    lines.extend([
        "",
        "Correction validation scores are pixel-derived orientation scores before",
        "and after applying `correction_angle = -estimated_angle`; they are not a",
        "second fitted model or a calibrated confidence probability.",
    ])
    return "\n".join(lines) + "\n"


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _tolerance_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _mean_or_none(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
