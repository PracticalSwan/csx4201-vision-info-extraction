"""Train-only preprocessing, K-Means k=4, Hungarian mapping, and evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

from . import config as cfgmod
from .orientation_features import load_feature_split
from .rotation_common import (
    ArtifactMismatchError,
    LeakageError,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    configuration_hash,
    read_csv_rows,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_VERSION = "kmeans-rotation-v1"
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
    "correct",
    "nearest_distance",
    "second_nearest_distance",
    "confidence",
    "centroid_distances",
    "feature_configuration_hash",
    "preprocessing_hash",
    "model_version",
    "image_path",
]


def fit_rotation_preprocessing(
    cfg: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    feature_config = _load_json(model_root / "feature_config.json")
    feature_summary = _load_json(cfgmod.resolve_path(cfg, "metadata") / "feature_summary.json")
    splits = sorted(feature_summary["counts_per_split"])
    if "train" not in splits:
        raise ValueError("training features are required")
    data = {split: load_feature_split(cfg, split) for split in splits}
    train = data["train"]
    if np.any(train["private"] != 0):
        raise LeakageError("private samples are present in preprocessing fit data")
    if train["X"].shape[0] < 4:
        raise ValueError("at least four public training feature rows are required")
    fit_ids = [str(value) for value in train["rotation_ids"]]
    fit_id_hash = _ids_hash(fit_ids)
    preprocessing_hash = configuration_hash({
        "feature_hash": feature_config["configuration_hash"],
        "rotation_manifest_hash": feature_config["rotation_manifest_hash"],
        "fit_rotation_id_hash": fit_id_hash,
        "feature_preprocessing": cfg.get("feature_preprocessing", {}),
        "pca": cfg.get("pca", {}),
    })
    scaler_path = model_root / "scaler.joblib"
    pca_path = model_root / "pca.joblib"
    summary_path = model_root / "preprocessing_summary.json"
    if summary_path.exists() and not force:
        existing = _load_json(summary_path)
        if (
            existing.get("preprocessing_hash") == preprocessing_hash
            and scaler_path.is_file()
            and (not existing.get("pca_enabled") or pca_path.is_file())
            and all(
                _transformed_cache_is_valid(
                    _transformed_path(cfg, feature_config, split), preprocessing_hash
                )
                for split in splits
            )
        ):
            return {"skipped": True, "summary": existing}

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train["X"])
    if not np.isfinite(scaled_train).all():
        raise ValueError("scaled training features contain non-finite values")
    pca_cfg = cfg.get("pca", {})
    pca_enabled = bool(pca_cfg.get("enabled", True))
    pca_model: PCA | None = None
    effective_components = int(scaled_train.shape[1])
    explained_variance: list[float] = []
    if pca_enabled:
        requested = int(pca_cfg.get("n_components", 128))
        maximum = min(scaled_train.shape[1], scaled_train.shape[0] - 1)
        effective_components = min(requested, maximum)
        if effective_components < 1:
            raise ValueError("PCA has no valid component count")
        pca_model = PCA(
            n_components=effective_components,
            random_state=int(pca_cfg.get("random_state", 42)),
            svd_solver=str(pca_cfg.get("solver", "randomized")),
        )
        # Use the same public transform path during fitting and after reload.
        # ``PCA.fit_transform`` may use an optimized randomized-SVD result whose
        # floating-point values differ measurably from a later ``transform`` on
        # very small smoke datasets, even though the fitted model is unchanged.
        pca_model.fit(scaled_train)
        transformed_train = pca_model.transform(scaled_train)
        explained_variance = pca_model.explained_variance_ratio_.astype(float).tolist()
    else:
        transformed_train = scaled_train
        if pca_path.exists():
            pca_path.unlink()

    _atomic_joblib_dump(scaler, scaler_path)
    if pca_model is not None:
        _atomic_joblib_dump(pca_model, pca_path)
    reloaded_scaler = joblib.load(scaler_path)
    reloaded_scaled = reloaded_scaler.transform(train["X"])
    if not np.allclose(reloaded_scaled, scaled_train, rtol=1e-6, atol=1e-7):
        raise ArtifactMismatchError("reloaded scaler transform differs from in-memory transform")
    if pca_model is not None:
        reloaded_pca = joblib.load(pca_path)
        reloaded_transformed = reloaded_pca.transform(reloaded_scaled)
        if not np.allclose(reloaded_transformed, transformed_train, rtol=1e-5, atol=1e-6):
            raise ArtifactMismatchError("reloaded PCA transform differs from in-memory transform")

    for split, values in data.items():
        scaled = reloaded_scaler.transform(values["X"])
        transformed = joblib.load(pca_path).transform(scaled) if pca_model is not None else scaled
        if not np.isfinite(transformed).all():
            raise ValueError(f"transformed {split} features contain non-finite values")
        metadata = {key: array for key, array in values.items() if key != "X"}
        atomic_save_npz(
            _transformed_path(cfg, feature_config, split),
            X=transformed.astype(np.float32),
            **metadata,
            preprocessing_hash=np.asarray([preprocessing_hash]),
        )

    summary = {
        "model_version": MODEL_VERSION,
        "profile": feature_config["profile"],
        "feature_configuration_hash": feature_config["configuration_hash"],
        "rotation_manifest_hash": feature_config["rotation_manifest_hash"],
        "preprocessing_hash": preprocessing_hash,
        "fit_splits": ["train"],
        "fit_private_sample_count": int(train["private"].sum()),
        "fit_rotation_count": len(fit_ids),
        "fit_rotation_id_hash": fit_id_hash,
        "fit_rotation_ids": fit_ids,
        "original_feature_dimension": int(train["X"].shape[1]),
        "scaled_feature_dimension": int(scaled_train.shape[1]),
        "pca_enabled": pca_enabled,
        "pca_requested_components": int(pca_cfg.get("n_components", 128)) if pca_enabled else None,
        "pca_effective_components": effective_components if pca_enabled else None,
        "pca_explained_variance_ratio": explained_variance,
        "pca_cumulative_explained_variance": float(sum(explained_variance)) if explained_variance else None,
        "output_dimension": int(transformed_train.shape[1]),
        "artifact_reload_verified": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(summary_path, summary)
    return {"skipped": False, "summary": summary}


def train_kmeans_rotation(
    cfg: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    preprocessing = _load_json(model_root / "preprocessing_summary.json")
    train = load_feature_split(cfg, "train", transformed=True)
    if str(train["preprocessing_hash"][0]) != preprocessing["preprocessing_hash"]:
        raise ArtifactMismatchError("training cache has a stale preprocessing hash")
    if np.any(train["private"] != 0):
        raise LeakageError("private rows are present in K-Means fit data")
    if _ids_hash([str(value) for value in train["rotation_ids"]]) != preprocessing["fit_rotation_id_hash"]:
        raise ArtifactMismatchError("training provenance differs from preprocessing fit provenance")
    kmeans_cfg = cfg.get("kmeans", {})
    n_clusters = int(kmeans_cfg.get("n_clusters", 4))
    if n_clusters != 4:
        raise ValueError("K-Means must use exactly four clusters")
    training_hash = configuration_hash({
        "preprocessing_hash": preprocessing["preprocessing_hash"],
        "fit_rotation_id_hash": preprocessing["fit_rotation_id_hash"],
        "rotation_manifest_hash": preprocessing["rotation_manifest_hash"],
        "kmeans": kmeans_cfg,
        "model_version": MODEL_VERSION,
    })
    kmeans_path = model_root / "kmeans.joblib"
    mapping_path = model_root / "cluster_to_zone.json"
    summary_path = model_root / "training_summary.json"
    if summary_path.exists() and not force:
        existing = _load_json(summary_path)
        if existing.get("training_hash") == training_hash and kmeans_path.is_file() and mapping_path.is_file():
            return {"skipped": True, "summary": existing}

    model = KMeans(
        n_clusters=4,
        random_state=int(kmeans_cfg.get("random_state", 42)),
        n_init=int(kmeans_cfg.get("n_init", 20)),
        max_iter=int(kmeans_cfg.get("max_iter", 300)),
        tol=float(kmeans_cfg.get("tolerance", 0.0001)),
    )
    raw_clusters = model.fit_predict(train["X"])
    cluster_sizes = np.bincount(raw_clusters, minlength=4)
    if len(cluster_sizes) != 4 or np.any(cluster_sizes == 0):
        raise ValueError(f"K-Means did not produce four usable clusters: {cluster_sizes.tolist()}")
    mapping, count_matrix = hungarian_cluster_mapping(raw_clusters, train["true_zones"])
    mapped = np.asarray([mapping[int(cluster)] for cluster in raw_clusters], dtype=np.int8)
    if set(mapping) != {0, 1, 2, 3} or set(mapping.values()) != {1, 2, 3, 4}:
        raise ValueError(f"cluster-zone mapping is not bijective: {mapping}")
    _atomic_joblib_dump(model, kmeans_path)
    atomic_write_json(mapping_path, {
        "model_version": MODEL_VERSION,
        "training_hash": training_hash,
        "mapping": {str(key): int(value) for key, value in sorted(mapping.items())},
        "learned_from_split": "train",
        "private_samples_used": 0,
    })
    reloaded = joblib.load(kmeans_path)
    if not np.array_equal(reloaded.predict(train["X"]), raw_clusters):
        raise ArtifactMismatchError("reloaded K-Means predictions differ")
    rows = []
    for cluster in range(4):
        rows.append({
            "cluster_id": cluster,
            "zone_1": int(count_matrix[cluster, 0]),
            "zone_2": int(count_matrix[cluster, 1]),
            "zone_3": int(count_matrix[cluster, 2]),
            "zone_4": int(count_matrix[cluster, 3]),
            "mapped_zone": mapping[cluster],
        })
    report_root = cfgmod.resolve_path(cfg, "reports") / "kmeans_evaluation"
    atomic_write_csv(
        report_root / "cluster_zone_count_matrix.csv",
        rows,
        ["cluster_id", "zone_1", "zone_2", "zone_3", "zone_4", "mapped_zone"],
    )
    warning_fraction = float(kmeans_cfg.get("small_cluster_fraction_warning", 0.01))
    warnings = [
        f"cluster {index} is below the configured small-cluster fraction"
        for index, count in enumerate(cluster_sizes)
        if count / len(raw_clusters) < warning_fraction
    ]
    summary = {
        "model_version": MODEL_VERSION,
        "training_hash": training_hash,
        "preprocessing_hash": preprocessing["preprocessing_hash"],
        "fit_splits": ["train"],
        "fit_private_sample_count": 0,
        "fit_rotation_count": int(len(raw_clusters)),
        "fit_rotation_id_hash": preprocessing["fit_rotation_id_hash"],
        "n_clusters": 4,
        "random_state": int(kmeans_cfg.get("random_state", 42)),
        "n_init": int(kmeans_cfg.get("n_init", 20)),
        "max_iter": int(kmeans_cfg.get("max_iter", 300)),
        "tolerance": float(kmeans_cfg.get("tolerance", 0.0001)),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "cluster_sizes": {str(index): int(count) for index, count in enumerate(cluster_sizes)},
        "cluster_to_zone": {str(key): int(value) for key, value in sorted(mapping.items())},
        "training_mapped_accuracy": float(accuracy_score(train["true_zones"], mapped)),
        "artifact_reload_verified": True,
        "warnings": warnings,
        "training_matrix_bytes": int(train["X"].nbytes),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_text(
        report_root / "kmeans_methodology.md",
        _methodology_report(summary, preprocessing),
    )
    return {"skipped": False, "summary": summary}


def hungarian_cluster_mapping(
    clusters: np.ndarray,
    true_zones: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    clusters = np.asarray(clusters, dtype=int)
    zones = np.asarray(true_zones, dtype=int)
    if clusters.shape != zones.shape:
        raise ValueError("cluster and zone arrays must have the same shape")
    matrix = np.zeros((4, 4), dtype=np.int64)
    for cluster, zone in zip(clusters, zones, strict=True):
        if cluster not in {0, 1, 2, 3} or zone not in {1, 2, 3, 4}:
            raise ValueError("clusters must be 0-3 and zones must be 1-4")
        matrix[cluster, zone - 1] += 1
    rows, columns = linear_sum_assignment(-matrix)
    mapping = {int(row): int(column + 1) for row, column in zip(rows, columns, strict=True)}
    if len(mapping) != 4:
        raise ValueError("Hungarian assignment did not return four mappings")
    return mapping, matrix


def centroid_confidence(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest, second-nearest, and bounded heuristic confidence."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("centroid distance matrix must have at least two columns")
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    confidence = np.clip((second - nearest) / np.maximum(second, 1e-12), 0.0, 1.0)
    return nearest, second, confidence


def evaluate_kmeans_rotation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    root = cfgmod.project_root(cfg)
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    training = _load_json(model_root / "training_summary.json")
    preprocessing = _load_json(model_root / "preprocessing_summary.json")
    mapping_payload = _load_json(model_root / "cluster_to_zone.json")
    if mapping_payload["training_hash"] != training["training_hash"]:
        raise ArtifactMismatchError("cluster mapping and K-Means training hashes differ")
    mapping = {int(key): int(value) for key, value in mapping_payload["mapping"].items()}
    model: KMeans = joblib.load(model_root / "kmeans.joblib")
    if model.n_clusters != 4:
        raise ArtifactMismatchError("saved K-Means does not use four clusters")

    metadata = cfgmod.resolve_path(cfg, "metadata")
    feature_summary = _load_json(metadata / "feature_summary.json")
    rotation_rows = {
        row["rotation_id"]: row for row in read_csv_rows(metadata / "rotation_manifest.csv")
        if row.get("generation_status") == "success"
    }
    report_root = cfgmod.resolve_path(cfg, "reports") / "kmeans_evaluation"
    all_metrics: dict[str, Any] = {}
    public_predictions: list[dict[str, Any]] = []
    private_aggregate: dict[str, Any] | None = None
    centroid_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    misclassified: list[dict[str, Any]] = []
    by_dataset_rows: list[dict[str, Any]] = []
    by_document_type_rows: list[dict[str, Any]] = []
    by_angle_rows: list[dict[str, Any]] = []
    split_payloads: dict[str, dict[str, Any]] = {}

    for split in sorted(feature_summary["counts_per_split"]):
        values = load_feature_split(cfg, split, transformed=True)
        if str(values["preprocessing_hash"][0]) != preprocessing["preprocessing_hash"]:
            raise ArtifactMismatchError(f"transformed {split} cache has a stale preprocessing hash")
        clusters = model.predict(values["X"])
        predicted_zones = np.asarray([mapping[int(cluster)] for cluster in clusters], dtype=np.int8)
        distances = model.transform(values["X"])
        nearest, second, confidence = centroid_confidence(distances)
        metrics = _evaluation_metrics(
            values["true_zones"],
            predicted_zones,
            clusters,
            values["X"],
            cfg,
        )
        all_metrics[split] = metrics
        split_payloads[split] = {
            "values": values,
            "clusters": clusters,
            "predicted_zones": predicted_zones,
            "distances": distances,
            "nearest": nearest,
            "second": second,
            "confidence": confidence,
        }
        if split == "private_test":
            private_aggregate = {
                "sample_count": int(len(clusters)),
                "metrics": metrics,
                "mean_confidence": float(confidence.mean()),
                "reliable_identity_exposure": False,
            }
        for cluster in range(4):
            mask = clusters == cluster
            if np.any(mask):
                centroid_rows.append({
                    "project_split": split,
                    "cluster_id": cluster,
                    "sample_count": int(mask.sum()),
                    "mean_nearest_distance": float(nearest[mask].mean()),
                    "median_nearest_distance": float(np.median(nearest[mask])),
                    "mean_second_distance": float(second[mask].mean()),
                })
        confidence_rows.append({
            "project_split": split,
            "sample_count": len(confidence),
            "mean_confidence": float(confidence.mean()),
            "median_confidence": float(np.median(confidence)),
            "minimum_confidence": float(confidence.min()),
            "maximum_confidence": float(confidence.max()),
        })
        if split == "private_test":
            continue
        for index, rotation_id_value in enumerate(values["rotation_ids"]):
            rotation_id = str(rotation_id_value)
            source = rotation_rows[rotation_id]
            record = {
                "rotation_id": rotation_id,
                "document_id": str(values["document_ids"][index]),
                "page_id": str(values["page_ids"][index]),
                "dataset": str(values["datasets"][index]),
                "dataset_component": source["dataset_component"],
                "document_type": source["document_type"],
                "project_split": split,
                "true_angle": float(values["true_angles"][index]),
                "true_zone": int(values["true_zones"][index]),
                "raw_cluster_id": int(clusters[index]),
                "predicted_zone": int(predicted_zones[index]),
                "correct": int(predicted_zones[index] == values["true_zones"][index]),
                "nearest_distance": float(nearest[index]),
                "second_nearest_distance": float(second[index]),
                "confidence": float(confidence[index]),
                "centroid_distances": json.dumps([float(value) for value in distances[index]]),
                "feature_configuration_hash": preprocessing["feature_configuration_hash"],
                "preprocessing_hash": preprocessing["preprocessing_hash"],
                "model_version": training["model_version"],
                "image_path": source["rotated_image_path"],
            }
            public_predictions.append(record)
            if not record["correct"]:
                misclassified.append(record)
        _append_group_metrics(by_dataset_rows, split, "dataset", public_predictions, values, predicted_zones)
        _append_group_metrics(by_document_type_rows, split, "document_type", public_predictions, values, predicted_zones)
        _append_group_metrics(by_angle_rows, split, "true_angle", public_predictions, values, predicted_zones)

    elapsed = time.perf_counter() - started
    total_samples = sum(int(values["sample_count"]) for values in all_metrics.values())
    atomic_write_json(report_root / "metrics.json", {
        "model_version": MODEL_VERSION,
        "training_hash": training["training_hash"],
        "classification_and_clustering_metrics": all_metrics,
        "private_test_aggregate": private_aggregate,
        "failure_counts": {
            "feature_extraction": int(feature_summary.get("missing_vectors", 0)),
            "evaluation": 0,
        },
        "elapsed_seconds": elapsed,
        "processing_seconds_per_sample": elapsed / max(1, total_samples),
    })
    atomic_write_csv(report_root / "predictions.csv", public_predictions, PUBLIC_PREDICTION_COLUMNS)
    atomic_write_csv(report_root / "misclassified_samples.csv", misclassified, PUBLIC_PREDICTION_COLUMNS)
    atomic_write_csv(
        report_root / "centroid_distance_summary.csv",
        centroid_rows,
        ["project_split", "cluster_id", "sample_count", "mean_nearest_distance", "median_nearest_distance", "mean_second_distance"],
    )
    atomic_write_csv(
        report_root / "confidence_summary.csv",
        confidence_rows,
        ["project_split", "sample_count", "mean_confidence", "median_confidence", "minimum_confidence", "maximum_confidence"],
    )
    atomic_write_csv(
        report_root / "metrics_by_dataset.csv",
        by_dataset_rows,
        ["project_split", "dataset", "sample_count", "accuracy", "macro_f1"],
    )
    atomic_write_csv(
        report_root / "metrics_by_document_type.csv",
        by_document_type_rows,
        ["project_split", "document_type", "sample_count", "accuracy", "macro_f1"],
    )
    atomic_write_csv(
        report_root / "metrics_by_angle.csv",
        by_angle_rows,
        ["project_split", "true_angle", "sample_count", "accuracy", "macro_f1"],
    )
    _write_boundary_analysis(report_root, public_predictions)
    _write_evaluation_plots(report_root, split_payloads, training)
    atomic_write_text(
        report_root / "evaluation_summary.md",
        _evaluation_report(all_metrics, training, private_aggregate),
    )
    return {
        "metrics": all_metrics,
        "private_test_aggregate": private_aggregate,
        "public_prediction_count": len(public_predictions),
        "misclassified_count": len(misclassified),
    }


def _evaluation_metrics(true_zones, predicted_zones, clusters, matrix, cfg):
    true_zones = np.asarray(true_zones, dtype=int)
    predicted_zones = np.asarray(predicted_zones, dtype=int)
    clusters = np.asarray(clusters, dtype=int)
    labels = [1, 2, 3, 4]
    precision, recall, f1, support = precision_recall_fscore_support(
        true_zones, predicted_zones, labels=labels, zero_division=0
    )
    cm = confusion_matrix(true_zones, predicted_zones, labels=labels)
    per_zone = {}
    for index, zone in enumerate(labels):
        truth = true_zones == zone
        predicted = predicted_zones == zone
        one_vs_rest_accuracy = float(np.mean(truth == predicted))
        per_zone[str(zone)] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "one_vs_rest_accuracy": one_vs_rest_accuracy,
        }
    silhouette = None
    if len(matrix) > 4 and len(np.unique(clusters)) > 1:
        max_samples = int(cfg.get("evaluation", {}).get("silhouette_max_samples", 2000))
        sample_size = min(max_samples, len(matrix))
        silhouette = float(silhouette_score(
            matrix,
            clusters,
            sample_size=sample_size if sample_size < len(matrix) else None,
            random_state=int(cfg.get("runtime", {}).get("random_seed", 42)),
        ))
    return {
        "sample_count": int(len(true_zones)),
        "accuracy": float(accuracy_score(true_zones, predicted_zones)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_zone": per_zone,
        "confusion_matrix": cm.astype(int).tolist(),
        "adjusted_rand_index": float(adjusted_rand_score(true_zones, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(true_zones, clusters)),
        "homogeneity": float(homogeneity_score(true_zones, clusters)),
        "completeness": float(completeness_score(true_zones, clusters)),
        "v_measure": float(v_measure_score(true_zones, clusters)),
        "silhouette_score": silhouette,
        "cluster_purity": _cluster_purity(true_zones, clusters),
    }


def _cluster_purity(true_zones, clusters):
    correct = 0
    for cluster in range(4):
        labels = true_zones[clusters == cluster]
        if len(labels):
            correct += Counter(int(value) for value in labels).most_common(1)[0][1]
    return float(correct / len(true_zones)) if len(true_zones) else 0.0


def _append_group_metrics(output, split, column, all_predictions, values, predicted_zones):
    current_ids = {str(value) for value in values["rotation_ids"]}
    current = [row for row in all_predictions if row["rotation_id"] in current_ids]
    grouped = defaultdict(list)
    for row in current:
        grouped[row[column]].append(row)
    for key, rows in sorted(grouped.items()):
        true = np.asarray([row["true_zone"] for row in rows])
        pred = np.asarray([row["predicted_zone"] for row in rows])
        _, _, f1, _ = precision_recall_fscore_support(true, pred, labels=[1, 2, 3, 4], zero_division=0)
        output.append({
            "project_split": split,
            column: key,
            "sample_count": len(rows),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1.mean()),
        })


def _write_boundary_analysis(report_root, predictions):
    boundary = [
        row for row in predictions
        if row["project_split"] in {"validation", "test"}
        and int(round(float(row["true_angle"]))) in {0, 1, 45, 89, 90, 91, 135, 179, 180, 181, 225, 269, 270, 271, 315, 359}
    ]
    columns = [
        "project_split", "true_angle", "dataset", "document_type", "sample_count", "accuracy",
        "mean_confidence", "mean_nearest_distance", "neighboring_zone_errors",
    ]
    rows = []
    grouped = defaultdict(list)
    for row in boundary:
        grouped[(
            row["project_split"],
            int(round(float(row["true_angle"]))),
            row["dataset"],
            row["document_type"],
        )].append(row)
    for (split, angle, dataset, document_type), group in sorted(grouped.items()):
        true_zone = group[0]["true_zone"]
        neighboring = sum(
            not row["correct"] and abs(int(row["predicted_zone"]) - int(true_zone)) in {1, 3}
            for row in group
        )
        rows.append({
            "project_split": split,
            "true_angle": angle,
            "dataset": dataset,
            "document_type": document_type,
            "sample_count": len(group),
            "accuracy": float(np.mean([row["correct"] for row in group])),
            "mean_confidence": float(np.mean([row["confidence"] for row in group])),
            "mean_nearest_distance": float(np.mean([row["nearest_distance"] for row in group])),
            "neighboring_zone_errors": neighboring,
        })
    atomic_write_csv(report_root / "boundary_case_results.csv", rows, columns)
    lines = [
        "# Boundary-Angle Analysis",
        "",
        "Measured results are reported for the required angles. Low confidence and",
        "neighboring-zone errors are evidence; possible causes remain hypotheses.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['project_split']} angle {row['true_angle']} "
            f"({row['dataset']}, {row['document_type']}): n={row['sample_count']}, "
            f"accuracy={row['accuracy']:.4f}, mean confidence={row['mean_confidence']:.4f}"
        )
    atomic_write_text(report_root / "boundary_analysis.md", "\n".join(lines) + "\n")


def _write_evaluation_plots(report_root, payloads, training):
    report_root.mkdir(parents=True, exist_ok=True)
    generic_split = next(
        (split for split in ("test", "validation", "train") if split in payloads),
        None,
    )
    for split, payload in payloads.items():
        cm = confusion_matrix(payload["values"]["true_zones"], payload["predicted_zones"], labels=[1, 2, 3, 4])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        ax.set_xticks(range(4), labels=["Z1", "Z2", "Z3", "Z4"])
        ax.set_yticks(range(4), labels=["Z1", "Z2", "Z3", "Z4"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Mapped zone confusion matrix: {split}")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(report_root / f"confusion_matrix_{split}.png", dpi=150)
        if split == generic_split:
            fig.savefig(report_root / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    sizes = [training["cluster_sizes"][str(index)] for index in range(4)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(4), sizes)
    ax.set_xticks(range(4))
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Training samples")
    ax.set_title("K-Means cluster sizes")
    fig.tight_layout()
    fig.savefig(report_root / "cluster_size_distribution.png", dpi=150)
    plt.close(fig)

    train_payload = payloads.get("train")
    if train_payload:
        zones = train_payload["values"]["true_zones"]
        clusters = train_payload["clusters"]
        matrix = np.zeros((4, 4), dtype=int)
        for cluster, zone in zip(clusters, zones, strict=True):
            matrix[int(cluster), int(zone) - 1] += 1
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Purples")
        ax.set_xticks(range(4), labels=["Z1", "Z2", "Z3", "Z4"])
        ax.set_yticks(range(4), labels=["C0", "C1", "C2", "C3"])
        ax.set_xlabel("True zone")
        ax.set_ylabel("Raw cluster")
        ax.set_title("Training cluster-zone distribution")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(report_root / "cluster_zone_distribution.png", dpi=150)
        plt.close(fig)


def _methodology_report(training, preprocessing):
    return "\n".join([
        "# K-Means Rotation Methodology",
        "",
        "K-Means is fitted without labels on public training features only.",
        "The scaler and optional PCA are also fitted only on that same training scope.",
        "Cluster IDs are arbitrary, so a 4 by 4 training count matrix is mapped",
        "one-to-one to Zones 1-4 using Hungarian assignment after K-Means fitting.",
        "",
        f"- Clusters: {training['n_clusters']}",
        f"- Training rows: {training['fit_rotation_count']}",
        f"- PCA enabled: {preprocessing['pca_enabled']}",
        f"- Output dimension: {preprocessing['output_dimension']}",
        f"- Inertia: {training['inertia']:.6f}",
        f"- Iterations: {training['n_iter']}",
        f"- Mapping: {training['cluster_to_zone']}",
        "",
        "Centroid-distance confidence is a heuristic margin, not a calibrated probability.",
    ]) + "\n"


def _evaluation_report(metrics, training, private):
    lines = [
        "# K-Means Rotation Evaluation",
        "",
        "Positive rotation angles are counterclockwise and zones use half-open 90-degree intervals.",
        "Classification metrics use the fixed training-derived cluster mapping. Clustering",
        "metrics use raw cluster IDs. No artifact is refitted during evaluation.",
        "",
        f"- K-Means inertia: {training['inertia']:.6f}",
        f"- Cluster sizes: {training['cluster_sizes']}",
        "",
        "## Metrics by split",
        "",
    ]
    for split, values in sorted(metrics.items()):
        lines.append(
            f"- {split}: accuracy={values['accuracy']:.4f}, macro F1={values['macro_f1']:.4f}, "
            f"ARI={values['adjusted_rand_index']:.4f}, NMI={values['normalized_mutual_information']:.4f}"
        )
    if private:
        lines.extend([
            "",
            "Private-test results are aggregate-only. No private filename, preview,",
            "source path, or per-document prediction is written.",
        ])
    return "\n".join(lines) + "\n"


def _transformed_path(cfg, feature_config, split):
    return (
        cfgmod.resolve_path(cfg, "features")
        / feature_config["profile"]
        / feature_config["configuration_hash"]
        / f"transformed_{split}.npz"
    )


def _transformed_cache_is_valid(path: Path, preprocessing_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            matrix = data["X"]
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                return False
            if str(data["preprocessing_hash"][0]) != preprocessing_hash:
                return False
            if len(data["rotation_ids"]) != matrix.shape[0]:
                return False
    except Exception:
        return False
    return True


def _atomic_joblib_dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".joblib", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        joblib.dump(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _ids_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
