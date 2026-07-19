from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import kmeans_display
from src.inference.kmeans_display import (
    PORTABLE_SCHEMA_VERSION,
    KMeansRotationDisplay,
    VersionNeutralKMeans,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_version_neutral_model(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    parameter_path = root / "inference_params.npz"
    np.savez_compressed(
        parameter_path,
        scaler_mean=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        scaler_scale=np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
        pca_components=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        pca_mean=np.zeros(3, dtype=np.float32),
        pca_whiten=np.asarray([0], dtype=np.uint8),
        pca_explained_variance=np.ones(2, dtype=np.float32),
        cluster_centers=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    (root / "inference_params.json").write_text(
        json.dumps(
            {
                "schema_version": PORTABLE_SCHEMA_VERSION,
                "parameters_file": parameter_path.name,
                "parameters_sha256": _sha256(parameter_path),
            }
        ),
        encoding="utf-8",
    )
    (root / "feature_config.json").write_text(
        json.dumps({"configuration": {}}), encoding="utf-8"
    )
    (root / "cluster_to_zone.json").write_text(
        json.dumps({"mapping": {"0": 1, "1": 2, "2": 3, "3": 4}}),
        encoding="utf-8",
    )


def test_version_neutral_transform_and_prediction(tmp_path: Path) -> None:
    _write_version_neutral_model(tmp_path)
    model = VersionNeutralKMeans(tmp_path)

    transformed = model.transform(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    labels, distances = model.predict_features(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    )

    assert transformed.tolist() == [[0.0, 0.0]]
    assert labels.tolist() == [0]
    assert distances.shape == (1, 4)
    assert distances[0, 0] == pytest.approx(0.0)


def test_display_prefers_version_neutral_parameters_without_joblib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "models"
    _write_version_neutral_model(model_root)
    monkeypatch.setattr(
        kmeans_display.joblib,
        "load",
        lambda *_args, **_kwargs: pytest.fail("joblib fallback must not load"),
    )
    monkeypatch.setattr(
        kmeans_display,
        "extract_feature_vector",
        lambda *_args, **_kwargs: np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    predictor = KMeansRotationDisplay(
        {
            "paths": {
                "project_root": str(tmp_path),
                "rotation_models": str(model_root),
            }
        }
    )
    result = predictor.predict(Image.new("RGB", (8, 8), "white"))

    assert result["cluster_id"] == 0
    assert result["zone"] == 1
    assert result["confidence"] == pytest.approx(1.0)
    assert result["purpose"] == "display_only"


def test_version_neutral_parameters_fail_closed_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    _write_version_neutral_model(tmp_path)
    metadata_path = tmp_path / "inference_params.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parameters_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        VersionNeutralKMeans(tmp_path)
