#!/usr/bin/env python3
"""Export and verify version-neutral parameters for display-only K-Means inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.kmeans_display import (  # noqa: E402
    PORTABLE_SCHEMA_VERSION,
    VersionNeutralKMeans,
)


PUBLIC_SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def export_parameters(model_root: Path) -> dict[str, Any]:
    source_paths = {
        name: model_root / name
        for name in ("scaler.joblib", "pca.joblib", "kmeans.joblib")
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    scaler = joblib.load(source_paths["scaler.joblib"])
    pca = joblib.load(source_paths["pca.joblib"])
    kmeans = joblib.load(source_paths["kmeans.joblib"])
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("exporter requires the fitted mean/std StandardScaler")
    if getattr(kmeans, "n_clusters", None) != 4:
        raise ValueError("exporter requires the preserved four-cluster K-Means")

    parameter_path = model_root / "inference_params.npz"
    temporary = parameter_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            scaler_mean=np.asarray(scaler.mean_),
            scaler_scale=np.asarray(scaler.scale_),
            pca_components=np.asarray(pca.components_),
            pca_mean=np.asarray(pca.mean_),
            pca_whiten=np.asarray([int(bool(pca.whiten))], dtype=np.uint8),
            pca_explained_variance=np.asarray(pca.explained_variance_),
            cluster_centers=np.asarray(kmeans.cluster_centers_),
        )
    temporary.replace(parameter_path)

    metadata = {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "parameters_file": parameter_path.name,
        "parameters_sha256": sha256_file(parameter_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_sklearn_version": sklearn.__version__,
        "source_numpy_version": np.__version__,
        "source_artifacts": {
            name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in source_paths.items()
        },
        "feature_dimension": int(np.asarray(scaler.mean_).shape[0]),
        "transformed_dimension": int(np.asarray(pca.components_).shape[0]),
        "cluster_count": int(kmeans.n_clusters),
        "pca_whiten": bool(pca.whiten),
    }
    write_json(model_root / "inference_params.json", metadata)
    return metadata


def verify_public_parity(
    model_root: Path, feature_root: Path, report_path: Path
) -> dict[str, Any]:
    scaler = joblib.load(model_root / "scaler.joblib")
    pca = joblib.load(model_root / "pca.joblib")
    kmeans = joblib.load(model_root / "kmeans.joblib")
    portable = VersionNeutralKMeans(model_root)

    total_rows = 0
    total_mismatches = 0
    maximum_transform_error = 0.0
    maximum_distance_error = 0.0
    maximum_confidence_error = 0.0
    split_rows: dict[str, int] = {}

    for split in PUBLIC_SPLITS:
        path = feature_root / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["X"])
            if np.any(np.asarray(payload["private"]) != 0):
                raise ValueError(f"{split} parity input contains private rows")

        expected_scaled = scaler.transform(features)
        expected_transformed = pca.transform(expected_scaled)
        expected_labels = kmeans.predict(expected_transformed)
        expected_distances = kmeans.transform(expected_transformed)

        actual_transformed = portable.transform(features)
        actual_labels, actual_distances = portable.predict_features(features)
        expected_confidence = _centroid_confidence(expected_distances)
        actual_confidence = _centroid_confidence(actual_distances)

        rows = int(features.shape[0])
        split_rows[split] = rows
        total_rows += rows
        total_mismatches += int(np.count_nonzero(expected_labels != actual_labels))
        maximum_transform_error = max(
            maximum_transform_error,
            float(np.max(np.abs(expected_transformed - actual_transformed))),
        )
        maximum_distance_error = max(
            maximum_distance_error,
            float(np.max(np.abs(expected_distances - actual_distances))),
        )
        maximum_confidence_error = max(
            maximum_confidence_error,
            float(np.max(np.abs(expected_confidence - actual_confidence))),
        )

    passed = (
        total_mismatches == 0
        and maximum_transform_error <= 1e-6
        and maximum_distance_error <= 1e-5
        and maximum_confidence_error <= 1e-6
    )
    report = {
        "status": "pass" if passed else "fail",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "public_only": True,
        "private_rows": 0,
        "splits": split_rows,
        "rows": total_rows,
        "cluster_label_mismatches": total_mismatches,
        "maximum_transform_absolute_error": maximum_transform_error,
        "maximum_distance_absolute_error": maximum_distance_error,
        "maximum_confidence_absolute_error": maximum_confidence_error,
        "tolerances": {
            "transform_absolute_error": 1e-6,
            "distance_absolute_error": 1e-5,
            "confidence_absolute_error": 1e-6,
        },
        "parameters_sha256": sha256_file(model_root / "inference_params.npz"),
        "source_sklearn_version": sklearn.__version__,
        "source_numpy_version": np.__version__,
    }
    write_json(report_path, report)
    if not passed:
        raise RuntimeError(
            "version-neutral K-Means parity failed; see "
            f"{report_path}"
        )
    return report


def _centroid_confidence(distances: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(distances, dtype=np.float64), axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    return np.clip((second - nearest) / np.maximum(second, 1e-12), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "kmeans_rotation",
    )
    parser.add_argument(
        "--verify-feature-root",
        type=Path,
        help="public feature directory containing train/validation/test NPZ files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "rotation"
        / "version_neutral_kmeans_parity.json",
    )
    args = parser.parse_args()

    model_root = args.model_root.expanduser().resolve()
    metadata = export_parameters(model_root)
    parity = None
    if args.verify_feature_root is not None:
        feature_root = args.verify_feature_root.expanduser().resolve()
        parity = verify_public_parity(model_root, feature_root, args.report.resolve())
    print(
        json.dumps(
            {
                "status": "complete",
                "parameters": str(model_root / "inference_params.npz"),
                "parameters_sha256": metadata["parameters_sha256"],
                "parity": parity,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
