from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.orientation_features import extract_feature_vector, feature_cache_is_valid, load_feature_split
from src.rotation_common import ArtifactMismatchError, BOUNDARY_ANGLES, LeakageError, read_csv_rows
from src.rotation_dataset import ROTATION_COLUMNS
from src.rotation_model import (
    _evaluation_metrics,
    centroid_confidence,
    evaluate_kmeans_rotation,
    fit_rotation_preprocessing,
    hungarian_cluster_mapping,
    train_kmeans_rotation,
)
from tests.rotation_test_helpers import make_rotation_config, write_csv, write_feature_cache


def test_combined_orientation_features_are_fixed_length_finite_and_deterministic(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)["rotation_features"]
    blank = np.full((45, 160), 255, dtype=np.uint8)
    document = blank.copy()
    document[5:8, 15:145] = 0
    document[18:21, 30:130] = 0
    document[31:34, 10:100] = 0

    blank_vector, blank_meta = extract_feature_vector(blank, cfg, return_metadata=True)
    first, metadata = extract_feature_vector(document, cfg, return_metadata=True)
    second = extract_feature_vector(document, cfg)

    assert first.dtype == np.float32
    assert len(first) == len(blank_vector) == metadata["dimension"] == blank_meta["dimension"]
    assert len(first) > 1_700
    assert np.isfinite(first).all()
    assert np.isfinite(blank_vector).all()
    np.testing.assert_array_equal(first, second)
    assert list(metadata["group_slices"]) == ["hog", "hough", "projection", "directional_edges", "geometry"]
    assert metadata["group_slices"]["hog"][0] == 0
    assert metadata["group_slices"]["geometry"][1] == len(first)
    assert metadata["group_slices"]["hog"][1] - metadata["group_slices"]["hog"][0] == 1_764
    assert metadata["group_slices"]["hough"][1] - metadata["group_slices"]["hough"][0] == 48
    assert len(first) == 1_893  # 32-bin projection fixture; production config uses 1,957.


def test_feature_cache_validation_detects_hash_and_dimension_changes(tmp_path: Path):
    summary_path = tmp_path / "feature_summary.json"
    cache_root = tmp_path / "features"
    cache_root.mkdir()
    config_hash = "cfg123"
    manifest_hash = "manifest123"
    summary_path.write_text(
        json.dumps({
            "configuration_hash": config_hash,
            "rotation_manifest_hash": manifest_hash,
            "feature_dimension": 3,
        }),
        encoding="utf-8",
    )
    np.savez_compressed(
        cache_root / "train.npz",
        X=np.ones((4, 3), dtype=np.float32),
        configuration_hash=np.asarray([config_hash]),
        rotation_manifest_hash=np.asarray([manifest_hash]),
    )
    assert feature_cache_is_valid(
        summary_path,
        cache_root,
        config_hash=config_hash,
        rotation_manifest_hash=manifest_hash,
        expected_splits={"train"},
    )
    assert not feature_cache_is_valid(
        summary_path,
        cache_root,
        config_hash=config_hash,
        rotation_manifest_hash="changed",
        expected_splits={"train"},
    )

    np.savez_compressed(
        cache_root / "train.npz",
        X=np.ones((4, 2), dtype=np.float32),
        configuration_hash=np.asarray([config_hash]),
        rotation_manifest_hash=np.asarray([manifest_hash]),
    )
    assert not feature_cache_is_valid(
        summary_path,
        cache_root,
        config_hash=config_hash,
        rotation_manifest_hash=manifest_hash,
        expected_splits={"train"},
    )


def _build_model_feature_artifacts(root: Path, cfg: dict, *, private_in_train: bool = False):
    metadata = root / "data/metadata"
    models = root / "models/rotation"
    features = root / "data/processed/features/smoke/feature_hash"
    metadata.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    feature_config = {
        "profile": "smoke",
        "configuration_hash": "feature_hash",
        "rotation_manifest_hash": "rotation_hash",
        "feature_dimension": 4,
    }
    (models / "feature_config.json").write_text(json.dumps(feature_config), encoding="utf-8")

    centers = np.asarray([
        [-12.0, 0.0, 0.0, 0.0],
        [0.0, -12.0, 0.0, 0.0],
        [0.0, 0.0, 12.0, 0.0],
        [0.0, 0.0, 0.0, 12.0],
    ], dtype=np.float32)
    offsets = np.asarray([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=np.float32)
    train_X = np.vstack([center + offset for center in centers for offset in offsets])
    train_zones = np.repeat(np.arange(1, 5), len(offsets))
    train_ids = write_feature_cache(
        features / "train.npz",
        train_X,
        train_zones,
        split="train",
        config_hash="feature_hash",
        rotation_hash="rotation_hash",
        private=private_in_train,
    )
    boundary_angles = np.asarray(BOUNDARY_ANGLES, dtype=np.float32)
    boundary_zones = (boundary_angles // 90).astype(np.int8) + 1
    validation_X = centers[boundary_zones - 1] + 1_000.0
    write_feature_cache(
        features / "validation.npz",
        validation_X,
        boundary_zones,
        split="validation",
        config_hash="feature_hash",
        rotation_hash="rotation_hash",
        angles=boundary_angles,
    )
    test_X = centers[boundary_zones - 1] + 500.0
    write_feature_cache(
        features / "test.npz",
        test_X,
        boundary_zones,
        split="test",
        config_hash="feature_hash",
        rotation_hash="rotation_hash",
        angles=boundary_angles,
    )
    private_X = centers - 1_000.0
    write_feature_cache(
        features / "private_test.npz",
        private_X,
        np.arange(1, 5),
        split="private_test",
        config_hash="feature_hash",
        rotation_hash="rotation_hash",
        private=True,
    )
    (metadata / "feature_summary.json").write_text(
        json.dumps({
            "counts_per_split": {"train": 20, "validation": 16, "test": 16, "private_test": 4},
            "missing_vectors": 0,
        }),
        encoding="utf-8",
    )
    return train_X, train_zones, train_ids


def test_preprocessing_fits_train_only_records_provenance_and_kmeans_maps_clusters(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    train_X, train_zones, train_ids = _build_model_feature_artifacts(tmp_path, cfg)

    preprocessing = fit_rotation_preprocessing(cfg)
    assert preprocessing["skipped"] is False
    summary = preprocessing["summary"]
    assert summary["fit_splits"] == ["train"]
    assert summary["fit_private_sample_count"] == 0
    assert summary["fit_rotation_ids"] == train_ids
    assert summary["fit_rotation_count"] == len(train_X)
    assert summary["artifact_reload_verified"] is True

    scaler = joblib.load(tmp_path / "models/rotation/scaler.joblib")
    np.testing.assert_allclose(scaler.mean_, train_X.mean(axis=0), rtol=0, atol=5e-7)
    # Evaluation values are deliberately extreme; their means must not affect the scaler fit.
    assert np.max(np.abs(scaler.mean_)) < 10
    transformed_private = load_feature_split(cfg, "private_test", transformed=True)
    assert np.isfinite(transformed_private["X"]).all()
    assert np.all(transformed_private["private"] == 1)
    assert str(transformed_private["preprocessing_hash"][0]) == summary["preprocessing_hash"]
    assert load_feature_split(cfg, "validation", transformed=True)["X"].shape[0] == 16
    assert load_feature_split(cfg, "test", transformed=True)["X"].shape[0] == 16
    assert fit_rotation_preprocessing(cfg)["skipped"] is True

    trained = train_kmeans_rotation(cfg)
    training = trained["summary"]
    assert training["fit_splits"] == ["train"]
    assert training["fit_private_sample_count"] == 0
    assert training["fit_rotation_id_hash"] == summary["fit_rotation_id_hash"]
    assert training["n_clusters"] == 4
    assert set(training["cluster_to_zone"].values()) == {1, 2, 3, 4}
    assert training["artifact_reload_verified"] is True
    assert training["training_mapped_accuracy"] == pytest.approx(1.0)

    model = joblib.load(tmp_path / "models/rotation/kmeans.joblib")
    transformed_train = load_feature_split(cfg, "train", transformed=True)
    assert len(np.unique(model.predict(transformed_train["X"]))) == 4

    # The standalone assignment is one-to-one even when raw cluster IDs are permuted.
    raw_clusters = np.repeat([2, 0, 3, 1], 5)
    mapping, matrix = hungarian_cluster_mapping(raw_clusters, train_zones)
    assert mapping == {0: 2, 1: 4, 2: 1, 3: 3}
    assert matrix.shape == (4, 4)


def test_preprocessing_rejects_private_rows_in_training_cache(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    _build_model_feature_artifacts(tmp_path, cfg, private_in_train=True)
    with pytest.raises(LeakageError, match="private samples"):
        fit_rotation_preprocessing(cfg)


def test_pca_fits_train_only_and_reload_transform_matches_all_transform_only_splits(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    cfg["pca"] = {"enabled": True, "n_components": 3, "random_state": 42, "solver": "full"}
    train_X, _, _ = _build_model_feature_artifacts(tmp_path, cfg)
    result = fit_rotation_preprocessing(cfg)
    summary = result["summary"]
    assert summary["pca_enabled"] is True
    assert summary["pca_effective_components"] == 3
    assert summary["output_dimension"] == 3
    assert summary["fit_splits"] == ["train"]
    assert summary["fit_rotation_count"] == len(train_X)

    scaler = joblib.load(tmp_path / "models/rotation/scaler.joblib")
    pca = joblib.load(tmp_path / "models/rotation/pca.joblib")
    assert pca.n_samples_ == len(train_X)
    for split, expected_count, private in (
        ("train", 20, False),
        ("validation", 16, False),
        ("test", 16, False),
        ("private_test", 4, True),
    ):
        raw = load_feature_split(cfg, split)
        transformed = load_feature_split(cfg, split, transformed=True)
        assert transformed["X"].shape == (expected_count, 3)
        np.testing.assert_allclose(
            transformed["X"],
            pca.transform(scaler.transform(raw["X"])),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(transformed["rotation_ids"], raw["rotation_ids"])
        np.testing.assert_array_equal(transformed["true_zones"], raw["true_zones"])
        assert bool(np.all(transformed["private"] != 0)) is private


def test_stale_transformed_cache_and_missing_artifact_errors_are_clear(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    _build_model_feature_artifacts(tmp_path, cfg)
    fit_rotation_preprocessing(cfg)
    transformed_path = tmp_path / "data/processed/features/smoke/feature_hash/transformed_train.npz"
    with np.load(transformed_path, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files if key != "preprocessing_hash"}
    np.savez_compressed(transformed_path, **arrays, preprocessing_hash=np.asarray(["stale_hash"]))
    with pytest.raises(ArtifactMismatchError, match="stale preprocessing hash"):
        train_kmeans_rotation(cfg)

    (tmp_path / "models/rotation/preprocessing_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="preprocessing_summary.json"):
        train_kmeans_rotation(cfg)


def test_centroid_confidence_is_bounded_and_zero_for_a_tie():
    nearest, second, confidence = centroid_confidence(np.asarray([
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
        [1.0, 1.0, 2.0, 3.0],
    ]))
    np.testing.assert_array_equal(nearest, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(second, [1.0, 1.0, 1.0])
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
    assert confidence[2] == 0.0


def _write_evaluation_rotation_manifest(root: Path, cfg: dict) -> None:
    rows = []
    for split in ("train", "validation", "test", "private_test"):
        values = load_feature_split(cfg, split)
        for index, rotation_value in enumerate(values["rotation_ids"]):
            private = split == "private_test"
            rows.append({
                "rotation_id": str(rotation_value),
                "document_id": str(values["document_ids"][index]),
                "page_id": str(values["page_ids"][index]),
                "dataset": "gmail" if private else "synthetic",
                "dataset_component": "private" if private else "fixture",
                "document_type": "private_document" if private else "receipt",
                "project_split": split,
                "source_image_path": "<private>" if private else "fixture.png",
                "rotated_image_path": (
                    f"data/processed/private/rotated/{rotation_value}.png"
                    if private else f"data/processed/public/rotated/{rotation_value}.png"
                ),
                "rotation_angle": float(values["true_angles"][index]),
                "normalized_angle": float(values["true_angles"][index]),
                "rotation_zone": int(values["true_zones"][index]),
                "rotation_direction": "counterclockwise",
                "source_width": 100,
                "source_height": 60,
                "output_width": 120,
                "output_height": 120,
                "background_fill": "white",
                "interpolation": "bicubic",
                "private_status": "private" if private else "public",
                "generation_profile": "smoke",
                "configuration_hash": "rotation_hash",
                "generation_status": "success",
                "error_message": "",
            })
    write_csv(root / "data/metadata/rotation_manifest.csv", rows, ROTATION_COLUMNS)


def test_evaluation_reports_4x4_classification_raw_clustering_boundaries_and_private_aggregates(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    _build_model_feature_artifacts(tmp_path, cfg)
    fit_rotation_preprocessing(cfg)
    train_kmeans_rotation(cfg)
    _write_evaluation_rotation_manifest(tmp_path, cfg)
    secret_filename = "Real Customer Account 918273.pdf"
    write_csv(
        tmp_path / "data/metadata/private_file_inventory.csv",
        [{"file_id": "private", "original_filename": secret_filename}],
        ["file_id", "original_filename"],
    )

    result = evaluate_kmeans_rotation(cfg)
    assert set(result["metrics"]) == {"train", "validation", "test", "private_test"}
    for values in result["metrics"].values():
        assert np.asarray(values["confusion_matrix"]).shape == (4, 4)
        assert set(values["per_zone"]) == {"1", "2", "3", "4"}
        assert -1.0 <= values["adjusted_rand_index"] <= 1.0
        assert 0.0 <= values["normalized_mutual_information"] <= 1.0
        assert 0.0 <= values["cluster_purity"] <= 1.0
    assert result["private_test_aggregate"]["sample_count"] == 4
    assert result["private_test_aggregate"]["reliable_identity_exposure"] is False

    report_root = tmp_path / "reports/kmeans_evaluation"
    predictions = read_csv_rows(report_root / "predictions.csv")
    assert predictions
    assert {row["project_split"] for row in predictions} == {"train", "validation", "test"}
    assert not any(row["dataset"] == "gmail" for row in predictions)
    boundary = read_csv_rows(report_root / "boundary_case_results.csv")
    for split in ("validation", "test"):
        assert {int(float(row["true_angle"])) for row in boundary if row["project_split"] == split} == set(BOUNDARY_ANGLES)
    for path in report_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            assert secret_filename not in path.read_text(encoding="utf-8", errors="replace")

    zones = np.repeat([1, 2, 3, 4], 2)
    clusters = np.repeat([2, 0, 3, 1], 2)
    direct = _evaluation_metrics(zones, zones, clusters, np.eye(8), cfg)
    assert direct["confusion_matrix"] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    assert direct["accuracy"] == 1.0
    assert direct["adjusted_rand_index"] == 1.0
    assert direct["normalized_mutual_information"] == 1.0
