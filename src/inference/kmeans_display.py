"""Failure-isolated display-only wrapper around the preserved K-Means model."""
from __future__ import annotations

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


class KMeansRotationDisplay:
    """Load the existing artifacts and predict a non-controlling quadrant."""

    purpose = "display_only"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        model_root = cfgmod.resolve_path(cfg, "rotation_models")
        self.feature_configuration = _load_json(model_root / "feature_config.json")["configuration"]
        self.scaler = joblib.load(model_root / "scaler.joblib")
        pca_path = model_root / "pca.joblib"
        self.pca = joblib.load(pca_path) if pca_path.is_file() else None
        self.model = joblib.load(model_root / "kmeans.joblib")
        mapping_payload = _load_json(model_root / "cluster_to_zone.json")
        self.mapping = {int(key): int(value) for key, value in mapping_payload["mapping"].items()}
        if getattr(self.model, "n_clusters", None) != 4 or set(self.mapping) != {0, 1, 2, 3}:
            raise ValueError("invalid four-cluster K-Means display artifacts")

    def predict(self, image: Image.Image) -> dict[str, Any]:
        rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        vector = np.asarray(
            extract_feature_vector(bgr, self.feature_configuration), dtype=np.float32
        ).reshape(1, -1)
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


def _centroid_confidence(distances: np.ndarray) -> np.ndarray:
    """Return the preserved bounded centroid-margin confidence without plotting imports."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("centroid distance matrix must have at least two columns")
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    return np.clip((second - nearest) / np.maximum(second, 1e-12), 0.0, 1.0)
