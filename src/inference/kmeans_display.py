"""Failure-isolated display-only wrapper around the preserved K-Means model."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from src import config as cfgmod
from src.orientation_features import extract_feature_vector

DISPLAY_WARNING = "This value does not control OCR or extraction."
PORTABLE_SCHEMA_VERSION = "rotation-display-inference-1.0"


class VersionNeutralKMeans:
    """Run the preserved scaler/PCA/K-Means parameters without pickle loading."""

    def __init__(self, model_root: Path) -> None:
        metadata = _load_json(model_root / "inference_params.json")
        if metadata.get("schema_version") != PORTABLE_SCHEMA_VERSION:
            raise ValueError("unsupported version-neutral K-Means artifact schema")

        parameter_name = str(metadata.get("parameters_file", ""))
        if not parameter_name or Path(parameter_name).name != parameter_name:
            raise ValueError("invalid version-neutral K-Means parameter filename")
        parameter_path = model_root / parameter_name
        if not parameter_path.is_file():
            raise FileNotFoundError(parameter_path)
        if _sha256_file(parameter_path) != metadata.get("parameters_sha256"):
            raise ValueError("version-neutral K-Means parameter hash mismatch")

        with np.load(parameter_path, allow_pickle=False) as payload:
            required = {
                "scaler_mean",
                "scaler_scale",
                "pca_components",
                "pca_mean",
                "pca_whiten",
                "pca_explained_variance",
                "cluster_centers",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(
                    "version-neutral K-Means parameters missing: "
                    + ", ".join(sorted(missing))
                )
            self.scaler_mean = np.asarray(payload["scaler_mean"]).copy()
            self.scaler_scale = np.asarray(payload["scaler_scale"]).copy()
            self.pca_components = np.asarray(payload["pca_components"]).copy()
            self.pca_mean = np.asarray(payload["pca_mean"]).copy()
            self.pca_whiten = bool(
                int(np.asarray(payload["pca_whiten"]).reshape(-1)[0])
            )
            self.pca_explained_variance = np.asarray(
                payload["pca_explained_variance"]
            ).copy()
            self.cluster_centers = np.asarray(payload["cluster_centers"]).copy()

        self._validate()
        self.n_clusters = int(self.cluster_centers.shape[0])

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the preserved StandardScaler and PCA numeric operations."""
        transformed = np.asarray(features, dtype=np.float32)
        if transformed.ndim == 1:
            transformed = transformed.reshape(1, -1)
        if transformed.ndim != 2 or transformed.shape[1] != self.scaler_mean.shape[0]:
            raise ValueError("unexpected K-Means feature dimension")

        transformed = transformed.copy()
        transformed -= self.scaler_mean.astype(transformed.dtype, copy=False)
        transformed /= self.scaler_scale.astype(transformed.dtype, copy=False)
        transformed = transformed @ self.pca_components.T
        transformed -= self.pca_mean.reshape(1, -1) @ self.pca_components.T
        if self.pca_whiten:
            scale = np.sqrt(self.pca_explained_variance)
            minimum = np.finfo(scale.dtype).eps
            scale = np.maximum(scale, minimum)
            transformed /= scale
        return transformed

    def distances(self, features: np.ndarray) -> np.ndarray:
        """Return Euclidean distances to the preserved cluster centers."""
        transformed = self.transform(features)
        deltas = (
            transformed[:, None, :].astype(np.float64)
            - self.cluster_centers[None, :, :].astype(np.float64)
        )
        return np.sqrt(np.einsum("nkd,nkd->nk", deltas, deltas))

    def predict_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cluster IDs and distances for one or more feature rows."""
        distances = self.distances(features)
        return np.argmin(distances, axis=1), distances

    def _validate(self) -> None:
        feature_count = self.scaler_mean.shape
        if (
            self.scaler_mean.ndim != 1
            or self.scaler_scale.shape != feature_count
            or self.pca_mean.shape != feature_count
            or self.pca_components.ndim != 2
            or self.pca_components.shape[1] != feature_count[0]
            or self.cluster_centers.ndim != 2
            or self.cluster_centers.shape[0] != 4
            or self.cluster_centers.shape[1] != self.pca_components.shape[0]
            or self.pca_explained_variance.shape
            != (self.pca_components.shape[0],)
        ):
            raise ValueError("invalid version-neutral K-Means parameter shapes")
        arrays = (
            self.scaler_mean,
            self.scaler_scale,
            self.pca_components,
            self.pca_mean,
            self.pca_explained_variance,
            self.cluster_centers,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("non-finite version-neutral K-Means parameters")
        if np.any(self.scaler_scale <= 0):
            raise ValueError("invalid version-neutral StandardScaler scale")
        if self.pca_whiten and np.any(self.pca_explained_variance < 0):
            raise ValueError("invalid version-neutral PCA variance")


class KMeansRotationDisplay:
    """Load the existing artifacts and predict a non-controlling quadrant."""

    purpose = "display_only"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_root = cfgmod.resolve_path(cfg, "rotation_models")
        self.feature_configuration = _load_json(model_root / "feature_config.json")["configuration"]
        portable_manifest = model_root / "inference_params.json"
        self.portable_model = (
            VersionNeutralKMeans(model_root) if portable_manifest.is_file() else None
        )
        self.scaler = None
        self.pca = None
        self.model = None
        if self.portable_model is None:
            self.scaler = joblib.load(model_root / "scaler.joblib")
            pca_path = model_root / "pca.joblib"
            self.pca = joblib.load(pca_path) if pca_path.is_file() else None
            self.model = joblib.load(model_root / "kmeans.joblib")
        mapping_payload = _load_json(model_root / "cluster_to_zone.json")
        self.mapping = {int(key): int(value) for key, value in mapping_payload["mapping"].items()}
        cluster_count = (
            self.portable_model.n_clusters
            if self.portable_model is not None
            else getattr(self.model, "n_clusters", None)
        )
        if cluster_count != 4 or set(self.mapping) != {0, 1, 2, 3}:
            raise ValueError("invalid four-cluster K-Means display artifacts")

    def predict(self, image: Image.Image) -> dict[str, Any]:
        rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        vector = np.asarray(
            extract_feature_vector(bgr, self.feature_configuration), dtype=np.float32
        ).reshape(1, -1)
        if self.portable_model is not None:
            clusters, distances = self.portable_model.predict_features(vector)
            cluster_id = int(clusters[0])
        else:
            transformed = self.scaler.transform(vector)
            if self.pca is not None:
                transformed = self.pca.transform(transformed)
            cluster_id = int(self.model.predict(transformed)[0])
            distances = self.model.transform(transformed)
        confidence = _centroid_confidence(distances)
        return {
            "cluster_id": cluster_id,
            "zone": self.mapping[cluster_id],
            "confidence": float(confidence[0]),
            "purpose": self.purpose,
            "warning": DISPLAY_WARNING,
        }


def safe_kmeans_display(
    predictor: Any | None, image: Image.Image, *, enabled: bool = True
) -> dict[str, Any]:
    """Return schema-valid null output on every display-branch failure."""
    if not enabled:
        return _unavailable("K-Means display branch disabled")
    if predictor is None:
        return _unavailable("K-Means display branch unavailable")
    try:
        result = dict(predictor.predict(image))
        result["purpose"] = "display_only"
        result["warning"] = DISPLAY_WARNING
        return result
    except Exception as exc:
        return _unavailable(f"K-Means display branch failed independently: {type(exc).__name__}: {exc}")


def _unavailable(warning: str) -> dict[str, Any]:
    return {
        "cluster_id": None,
        "zone": None,
        "confidence": None,
        "purpose": "display_only",
        "warning": warning,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centroid_confidence(distances: np.ndarray) -> np.ndarray:
    """Return the preserved bounded centroid-margin confidence without plotting imports."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("centroid distance matrix must have at least two columns")
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    return np.clip((second - nearest) / np.maximum(second, 1e-12), 0.0, 1.0)
