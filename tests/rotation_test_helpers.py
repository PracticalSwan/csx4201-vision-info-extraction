"""Small synthetic builders shared by the rotation-pipeline tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.page_preparation import PAGE_COLUMNS, page_preparation_hash
from src.rotation_common import rotation_filename, sha256_file, stable_id
from src.rotation_dataset import ROTATION_COLUMNS, SPLIT_COLUMNS, rotation_configuration_hash


def make_rotation_config(root: Path) -> dict[str, Any]:
    """Return a complete minimal rotation configuration rooted in ``root``."""
    return {
        "paths": {
            "project_root": str(root),
            "metadata": "data/metadata",
            "processed": "data/processed",
            "page_images": "data/processed/public/page_images",
            "private_page_images": "data/processed/private/page_images",
            "rotated_images": "data/processed/rotated_images",
            "features": "data/processed/features",
            "splits": "data/processed/splits",
            "rotation_models": "models/rotation",
            "reports": "reports",
        },
        "page_selection": {
            "max_pages_per_public_dataset": 100,
            "group_fatura_template_families": True,
            "group_coru_shared_stems": True,
            "coru_components": [
                "Receipt Images & Key Information Detection",
                "Receipt Question Answering",
            ],
            "exclude_coru_components": {
                "OCR Dataset": "line-crop OCR data is outside page-rotation scope",
                "Item Information Extraction": "tabular data is not a page image",
            },
        },
        "page_preparation": {
            "pdf_dpi": 72,
            "normalize_exif": True,
            "materialize_existing_images": False,
            "supported_image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        },
        "rotation_splits": {
            "seed": 42,
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "rotation_generation": {
            "smoke_angles": {
                "zone_1": [45],
                "zone_2": [135],
                "zone_3": [225],
                "zone_4": [315],
            },
            "train_angles": {
                "zone_1": [0, 45],
                "zone_2": [90, 135],
                "zone_3": [180, 225],
                "zone_4": [270, 315],
            },
            "boundary_angles": [0, 1, 45, 89, 90, 91, 135, 179, 180, 181, 225, 269, 270, 271, 315, 359],
            "private_test_angles": [45, 135, 225, 315],
            "max_variants_per_page": 20,
            "smoke_pages_per_dataset_per_split": 1,
            "output_max_dimension": 256,
            "default_profile": "smoke",
            "full_profile_scope": "synthetic bounded test corpus",
        },
        "rotation_features": {
            "strategy": "hog_hough",
            "resize_width": 128,
            "resize_height": 128,
            "padding_value": 255,
            "contrast_normalization": "equalize_hist",
            "hog": {
                "orientations": 9,
                "pixels_per_cell": [16, 16],
                "cells_per_block": [2, 2],
                "block_norm": "L2-Hys",
            },
            "hough": {
                "angle_bins": 36,
                "canny_threshold_1": 50,
                "canny_threshold_2": 150,
                "threshold": 12,
                "min_line_length": 8,
                "max_line_gap": 4,
            },
            "projection_profiles": {"enabled": True, "summary_bins": 32},
            "directional_edges": {"enabled": True},
            "geometric_features": {"enabled": True},
        },
        "feature_preprocessing": {"standardize": True},
        "pca": {"enabled": False, "n_components": 4, "random_state": 42, "solver": "full"},
        "kmeans": {
            "n_clusters": 4,
            "random_state": 42,
            "n_init": 10,
            "max_iter": 100,
            "tolerance": 0.0001,
            "small_cluster_fraction_warning": 0.01,
        },
        "evaluation": {"silhouette_max_samples": 100},
        "angle_estimation": {
            "coarse_step_degrees": 10.0,
            "fine_step_degrees": 2.0,
            "fine_window_degrees": 4.0,
            "scoring_size": 64,
            "reliability_threshold": 0.0,
            "minimum_ink_fraction": 0.001,
            "minimum_edge_fraction": 0.001,
        },
        "runtime": {
            "random_seed": 42,
            "workers": 1,
            "minimum_free_space_gb": 0,
            "maximum_disk_usage_fraction": 1.0,
            "disk_estimate_safety_multiplier": 1.0,
        },
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def page_row(
    page_id: str,
    *,
    document_id: str | None = None,
    dataset: str = "sroie",
    private: bool = False,
    sha256: str | None = None,
    prepared_image_path: str | None = None,
    selection_status: str = "selected",
    preparation_hash_value: str = "",
) -> dict[str, Any]:
    document_id = document_id or f"doc_{page_id}"
    safe_path = prepared_image_path or (
        f"data/processed/private/page_images/{page_id}.png"
        if private
        else f"data/processed/public/page_images/{page_id}.png"
    )
    return {
        "page_id": page_id,
        "document_id": document_id,
        "source_document_id": document_id,
        "source_file_id": f"file_{page_id}",
        "dataset": "gmail" if private else dataset,
        "dataset_component": "private" if private else "synthetic",
        "source_type": "pdf_page" if private else "image",
        "document_type": "private_document" if private else "receipt",
        "language": "mixed" if private else "en",
        "source_file_path": "<private>" if private else f"{dataset}/{page_id}.png",
        "prepared_image_path": safe_path,
        "source_file_format": ".pdf" if private else ".png",
        "source_page_number": 1,
        "source_page_count": 1,
        "source_width": 80,
        "source_height": 60,
        "prepared_width": 80,
        "prepared_height": 60,
        "prepared_format": "png",
        "materialization_mode": "converted_pdf_page" if private else "referenced_image",
        "preparation_hash": preparation_hash_value,
        "annotation_availability": "none",
        "annotation_path": "",
        "private_status": "private" if private else "public",
        "usability_status": "usable",
        "sha256": sha256 or f"sha_{page_id}",
        "original_dataset_split": "synthetic",
        "logical_document_key": document_id,
        "template_family": "",
        "selection_status": selection_status,
        "exclusion_reason": "",
        "notes": "",
    }


def split_row(page: Mapping[str, Any], split: str, group: str | None = None) -> dict[str, Any]:
    return {
        "page_id": page["page_id"],
        "document_id": page["document_id"],
        "source_file_id": page["source_file_id"],
        "dataset": page["dataset"],
        "dataset_component": page["dataset_component"],
        "document_type": page["document_type"],
        "original_dataset_split": page["original_dataset_split"],
        "project_split": split,
        "split_group_id": group or f"group_{page['page_id']}",
        "exact_duplicate_group": "",
        "near_duplicate_group": "",
        "template_family": page["template_family"],
        "private_status": page["private_status"],
        "prepared_image_path": page["prepared_image_path"],
        "selection_status": page["selection_status"],
        "exclusion_reason": page["exclusion_reason"],
    }


def build_valid_verification_artifacts(root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Create the smallest verifier-clean smoke-profile artifact set."""
    (root / "data/raw").mkdir(parents=True, exist_ok=True)
    metadata = root / "data/metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    prepared = root / "data/processed/public/page_images/page_a.png"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), "white").save(prepared)
    preparation_hash_value = page_preparation_hash(cfg)
    source_fingerprint = sha256_file(prepared)
    page = page_row(
        "page_a",
        prepared_image_path=prepared.relative_to(root).as_posix(),
        sha256=source_fingerprint,
        preparation_hash_value=preparation_hash_value,
    )
    split = split_row(page, "train")
    write_csv(metadata / "page_manifest.csv", [page], PAGE_COLUMNS)
    write_csv(metadata / "split_manifest.csv", [split], SPLIT_COLUMNS)
    # The baseline verifier loads both inventories even when its sample set is empty.
    write_csv(metadata / "file_inventory.csv", [], ["file_id"])
    write_csv(metadata / "private_file_inventory.csv", [], ["file_id", "original_filename"])

    verification = root / "reports/verification"
    verification.mkdir(parents=True, exist_ok=True)
    (verification / "raw_baseline.json").write_text(
        json.dumps({"raw_file_count": 0, "raw_size_bytes": 0, "samples": []}),
        encoding="utf-8",
    )

    rotations: list[dict[str, Any]] = []
    rotation_hash = rotation_configuration_hash(cfg, "smoke")
    for angle in (0, 90, 180, 270):
        zone = angle // 90 + 1
        rotation_id = stable_id("rotation", page["page_id"], angle, "smoke", length=16)
        target = root / "data/processed/rotated_images/smoke/train" / f"zone_{zone}" / rotation_filename(page["page_id"], angle, zone)
        target.parent.mkdir(parents=True, exist_ok=True)
        pnginfo = PngInfo()
        for key, value in {
            "rotation_pipeline_artifact": "rotated_page",
            "configuration_hash": rotation_hash,
            "rotation_id": rotation_id,
            "source_sha256": source_fingerprint,
        }.items():
            pnginfo.add_text(key, str(value))
        Image.new("RGB", (80, 60), "white").save(target, pnginfo=pnginfo)
        rotations.append({
            "rotation_id": rotation_id,
            "document_id": page["document_id"],
            "page_id": page["page_id"],
            "dataset": page["dataset"],
            "dataset_component": page["dataset_component"],
            "document_type": page["document_type"],
            "project_split": "train",
            "source_image_path": page["prepared_image_path"],
            "rotated_image_path": target.relative_to(root).as_posix(),
            "rotation_angle": angle,
            "normalized_angle": angle,
            "rotation_zone": zone,
            "rotation_direction": "counterclockwise",
            "source_width": 80,
            "source_height": 60,
            "output_width": 80,
            "output_height": 60,
            "background_fill": "white",
            "interpolation": "bicubic",
            "private_status": "public",
            "generation_profile": "smoke",
            "configuration_hash": rotation_hash,
            "generation_status": "success",
            "error_message": "",
        })
    write_csv(metadata / "rotation_manifest.csv", rotations, ROTATION_COLUMNS)
    return {"page": page, "split": split, "rotations": rotations}


def write_feature_cache(
    path: Path,
    X: np.ndarray,
    zones: np.ndarray,
    *,
    split: str,
    config_hash: str,
    rotation_hash: str,
    private: bool = False,
    angles: np.ndarray | None = None,
) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"{split}_rotation_{index:03d}" for index in range(len(X))]
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        rotation_ids=np.asarray(ids),
        document_ids=np.asarray([f"{split}_doc_{index:03d}" for index in range(len(X))]),
        page_ids=np.asarray([f"{split}_page_{index:03d}" for index in range(len(X))]),
        datasets=np.asarray(["gmail" if private else "synthetic"] * len(X)),
        true_angles=np.asarray(
            angles if angles is not None else [(int(zone) - 1) * 90 + 45 for zone in zones],
            dtype=np.float32,
        ),
        true_zones=np.asarray(zones, dtype=np.int8),
        private=np.asarray([private] * len(X), dtype=np.int8),
        configuration_hash=np.asarray([config_hash]),
        rotation_manifest_hash=np.asarray([rotation_hash]),
    )
    return ids
