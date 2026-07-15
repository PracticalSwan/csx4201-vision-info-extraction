"""Deterministic fixed-length orientation features and split NPZ caches."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from . import config as cfgmod
from .rotation_common import (
    ArtifactMismatchError,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    configuration_hash,
    manifest_digest,
    parse_dataset_filter,
    read_csv_rows,
)

FEATURE_VERSION = "orientation-v1"
FEATURE_MANIFEST_COLUMNS = [
    "feature_id",
    "rotation_id",
    "document_id",
    "page_id",
    "dataset",
    "project_split",
    "true_angle",
    "true_zone",
    "image_path",
    "feature_strategy",
    "feature_version",
    "feature_dimension",
    "feature_cache_path",
    "feature_row_index",
    "configuration_hash",
    "extraction_status",
    "error_message",
    "private_status",
]
FEATURE_ERROR_COLUMNS = [
    "timestamp",
    "rotation_id",
    "dataset",
    "project_split",
    "error_type",
    "error_message",
]


def resize_with_padding(
    gray: np.ndarray,
    width: int,
    height: int,
    padding_value: int = 255,
) -> np.ndarray:
    """Resize without distortion, center on a fixed white canvas, and never crop."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("expected a non-empty grayscale image")
    source_height, source_width = gray.shape
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((height, width), int(padding_value), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def extract_feature_vector(
    image: np.ndarray,
    feature_cfg: Mapping[str, Any],
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Extract HOG or the default combined orientation feature vector."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    original_height, original_width = gray.shape
    width = int(feature_cfg.get("resize_width", 128))
    height = int(feature_cfg.get("resize_height", 128))
    padded = resize_with_padding(gray, width, height, int(feature_cfg.get("padding_value", 255)))
    if feature_cfg.get("contrast_normalization") == "equalize_hist":
        normalized = cv2.equalizeHist(padded)
    else:
        normalized = padded
    hog_values = _hog_features(normalized, feature_cfg["hog"])
    strategy = str(feature_cfg.get("strategy", "hog_hough"))
    groups: list[tuple[str, np.ndarray]] = [("hog", hog_values)]
    diagnostics: dict[str, Any] = {}
    if strategy == "hog_hough":
        hough_values, edges, hough_diag = _hough_features(normalized, feature_cfg["hough"])
        groups.append(("hough", hough_values))
        diagnostics.update(hough_diag)
        if feature_cfg.get("projection_profiles", {}).get("enabled", True):
            groups.append(("projection", _projection_features(normalized, feature_cfg["projection_profiles"])))
        if feature_cfg.get("directional_edges", {}).get("enabled", True):
            groups.append(("directional_edges", _directional_edge_features(normalized)))
        if feature_cfg.get("geometric_features", {}).get("enabled", True):
            groups.append((
                "geometry",
                _geometric_features(gray, edges, original_width, original_height),
            ))
    elif strategy != "hog":
        raise ValueError(f"unsupported feature strategy: {strategy}")
    slices: dict[str, list[int]] = {}
    offset = 0
    vectors = []
    for name, values in groups:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        slices[name] = [offset, offset + len(values)]
        offset += len(values)
        vectors.append(values)
    vector = np.concatenate(vectors).astype(np.float32, copy=False)
    if not np.isfinite(vector).all():
        raise ValueError("feature vector contains NaN or infinity")
    metadata = {
        "dimension": int(vector.size),
        "group_slices": slices,
        "strategy": strategy,
        "version": FEATURE_VERSION,
        **diagnostics,
    }
    return (vector, metadata) if return_metadata else vector


def extract_rotation_features(
    cfg: Mapping[str, Any],
    *,
    profile: str | None = None,
    force: bool = False,
    limit: int = 0,
    datasets: str | list[str] | None = None,
    splits: str | list[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    rotation_manifest_path = metadata / "rotation_manifest.csv"
    if not rotation_manifest_path.is_file():
        raise FileNotFoundError("rotation_manifest.csv is required before feature extraction")
    rotations = [
        row for row in read_csv_rows(rotation_manifest_path)
        if row.get("generation_status") == "success"
    ]
    if not rotations:
        raise ValueError("rotation manifest has no successful rows")
    active_profile = profile or rotations[0]["generation_profile"]
    rotations = [row for row in rotations if row["generation_profile"] == active_profile]
    dataset_filter = parse_dataset_filter(datasets)
    split_filter = parse_dataset_filter(splits)
    if dataset_filter is not None:
        rotations = [row for row in rotations if row["dataset"].lower() in dataset_filter]
    if split_filter is not None:
        rotations = [row for row in rotations if row["project_split"].lower() in split_filter]
    rotations.sort(key=lambda row: row["rotation_id"])
    if limit:
        rotations = rotations[:limit]
    feature_cfg = dict(cfg["rotation_features"])
    config_hash = configuration_hash(feature_cfg)
    rotation_hash = manifest_digest(rotation_manifest_path)
    cache_root = cfgmod.resolve_path(cfg, "features") / active_profile / config_hash
    summary_path = metadata / "feature_summary.json"
    if not force and feature_cache_is_valid(
        summary_path,
        cache_root,
        config_hash=config_hash,
        rotation_manifest_hash=rotation_hash,
        expected_splits={row["project_split"] for row in rotations},
    ):
        return {
            "skipped": True,
            "profile": active_profile,
            "configuration_hash": config_hash,
            "summary": _load_json(summary_path),
        }

    max_workers = max(1, int(workers or cfg.get("runtime", {}).get("workers", 2)))
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rotations:
        by_split[row["project_split"]].append(row)
    manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    group_slices: dict[str, list[int]] | None = None
    storage_bytes = 0
    split_counts: dict[str, int] = {}

    for split, rows in sorted(by_split.items()):
        results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_row = {
                pool.submit(_extract_one, root / row["rotated_image_path"], feature_cfg): row
                for row in rows
            }
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    vector, info = future.result()
                    results[row["rotation_id"]] = (vector, info)
                except Exception as exc:
                    errors.append({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "rotation_id": row["rotation_id"],
                        "dataset": row["dataset"],
                        "project_split": split,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    })
        successful_rows = [row for row in rows if row["rotation_id"] in results]
        if not successful_rows:
            raise ValueError(f"no features extracted for split {split}")
        vectors = []
        for row in successful_rows:
            vector, info = results[row["rotation_id"]]
            vectors.append(vector)
            dimensions.add(int(vector.size))
            if group_slices is None:
                group_slices = info["group_slices"]
            elif group_slices != info["group_slices"]:
                raise ValueError("feature group slices changed within one extraction run")
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite feature values in split {split}")
        cache_path = cache_root / f"{split}.npz"
        atomic_save_npz(
            cache_path,
            X=matrix,
            rotation_ids=np.asarray([row["rotation_id"] for row in successful_rows]),
            document_ids=np.asarray([row["document_id"] for row in successful_rows]),
            page_ids=np.asarray([row["page_id"] for row in successful_rows]),
            datasets=np.asarray([row["dataset"] for row in successful_rows]),
            true_angles=np.asarray([float(row["normalized_angle"]) for row in successful_rows], dtype=np.float32),
            true_zones=np.asarray([int(row["rotation_zone"]) for row in successful_rows], dtype=np.int8),
            private=np.asarray([row["private_status"] == "private" for row in successful_rows], dtype=np.int8),
            configuration_hash=np.asarray([config_hash]),
            rotation_manifest_hash=np.asarray([rotation_hash]),
        )
        storage_bytes += cache_path.stat().st_size
        split_counts[split] = len(successful_rows)
        for index, row in enumerate(successful_rows):
            manifest_rows.append({
                "feature_id": f"feature_{row['rotation_id']}",
                "rotation_id": row["rotation_id"],
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "dataset": row["dataset"],
                "project_split": split,
                "true_angle": row["normalized_angle"],
                "true_zone": row["rotation_zone"],
                "image_path": row["rotated_image_path"],
                "feature_strategy": feature_cfg.get("strategy", "hog_hough"),
                "feature_version": FEATURE_VERSION,
                "feature_dimension": matrix.shape[1],
                "feature_cache_path": cache_path.relative_to(root).as_posix(),
                "feature_row_index": index,
                "configuration_hash": config_hash,
                "extraction_status": "success",
                "error_message": "",
                "private_status": row["private_status"],
            })
    dimension = next(iter(dimensions))
    manifest_rows.sort(key=lambda row: row["rotation_id"])
    error_ids = {row["rotation_id"] for row in errors}
    for row in rotations:
        if row["rotation_id"] in error_ids:
            manifest_rows.append({
                "feature_id": f"feature_{row['rotation_id']}",
                "rotation_id": row["rotation_id"],
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "dataset": row["dataset"],
                "project_split": row["project_split"],
                "true_angle": row["normalized_angle"],
                "true_zone": row["rotation_zone"],
                "image_path": row["rotated_image_path"],
                "feature_strategy": feature_cfg.get("strategy", "hog_hough"),
                "feature_version": FEATURE_VERSION,
                "feature_dimension": "",
                "feature_cache_path": "",
                "feature_row_index": "",
                "configuration_hash": config_hash,
                "extraction_status": "failed",
                "error_message": next(item["error_message"] for item in errors if item["rotation_id"] == row["rotation_id"]),
                "private_status": row["private_status"],
            })
    manifest_rows.sort(key=lambda row: row["rotation_id"])
    summary = {
        "feature_strategy": feature_cfg.get("strategy", "hog_hough"),
        "feature_version": FEATURE_VERSION,
        "profile": active_profile,
        "configuration_hash": config_hash,
        "rotation_manifest_hash": rotation_hash,
        "feature_dimension": dimension,
        "feature_group_slices": group_slices,
        "counts_per_split": split_counts,
        "counts_per_zone": dict(Counter(
            str(row["true_zone"]) for row in manifest_rows if row["extraction_status"] == "success"
        )),
        "missing_vectors": len(errors),
        "invalid_vectors": 0,
        "nan_count": 0,
        "infinity_count": 0,
        "extraction_time_seconds": time.perf_counter() - started,
        "storage_size_bytes": storage_bytes,
        "cache_root": cache_root.relative_to(root).as_posix(),
    }
    atomic_write_csv(metadata / "feature_manifest.csv", manifest_rows, FEATURE_MANIFEST_COLUMNS)
    atomic_write_csv(metadata / "feature_extraction_errors.csv", errors, FEATURE_ERROR_COLUMNS)
    atomic_write_json(summary_path, summary)
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    atomic_write_json(model_root / "feature_config.json", {
        "feature_version": FEATURE_VERSION,
        "profile": active_profile,
        "configuration": feature_cfg,
        "configuration_hash": config_hash,
        "rotation_manifest_hash": rotation_hash,
        "feature_dimension": dimension,
        "feature_group_slices": group_slices,
        "opencv_version": cv2.__version__,
    })
    atomic_write_text(
        cfgmod.resolve_path(cfg, "reports") / "feature_analysis" / "feature_extraction.md",
        _feature_report(summary),
    )
    return {
        "skipped": False,
        "profile": active_profile,
        "configuration_hash": config_hash,
        "summary": summary,
        "errors": errors,
    }


def feature_cache_is_valid(
    summary_path: Path,
    cache_root: Path,
    *,
    config_hash: str,
    rotation_manifest_hash: str,
    expected_splits: set[str],
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = _load_json(summary_path)
    except Exception:
        return False
    if summary.get("configuration_hash") != config_hash:
        return False
    if summary.get("rotation_manifest_hash") != rotation_manifest_hash:
        return False
    dimension = int(summary.get("feature_dimension", 0))
    if dimension <= 0:
        return False
    for split in expected_splits:
        path = cache_root / f"{split}.npz"
        if not path.is_file():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            if data["X"].ndim != 2 or data["X"].shape[1] != dimension:
                return False
            if str(data["configuration_hash"][0]) != config_hash:
                return False
            if str(data["rotation_manifest_hash"][0]) != rotation_manifest_hash:
                return False
        except Exception:
            return False
    return True


def load_feature_split(
    cfg: Mapping[str, Any],
    split: str,
    *,
    transformed: bool = False,
) -> dict[str, np.ndarray]:
    feature_config_path = cfgmod.resolve_path(cfg, "rotation_models") / "feature_config.json"
    if not feature_config_path.is_file():
        raise FileNotFoundError(feature_config_path)
    feature_config = _load_json(feature_config_path)
    cache_root = (
        cfgmod.resolve_path(cfg, "features")
        / feature_config["profile"]
        / feature_config["configuration_hash"]
    )
    name = f"transformed_{split}.npz" if transformed else f"{split}.npz"
    path = cache_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    result = {key: data[key] for key in data.files}
    if not transformed:
        if result["X"].shape[1] != int(feature_config["feature_dimension"]):
            raise ArtifactMismatchError(f"feature dimension mismatch in {path}")
        if str(result["configuration_hash"][0]) != feature_config["configuration_hash"]:
            raise ArtifactMismatchError(f"feature configuration mismatch in {path}")
    return result


def _extract_one(path: Path, feature_cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    vector, metadata = extract_feature_vector(image, feature_cfg, return_metadata=True)
    return vector, metadata


def _hog_features(gray: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    orientations = int(cfg.get("orientations", 9))
    cell = tuple(int(value) for value in cfg.get("pixels_per_cell", (16, 16)))
    cells_per_block = tuple(int(value) for value in cfg.get("cells_per_block", (2, 2)))
    height, width = gray.shape
    block_size = (cell[0] * cells_per_block[0], cell[1] * cells_per_block[1])
    block_stride = cell
    if width % cell[0] or height % cell[1]:
        raise ValueError("HOG window dimensions must be divisible by pixels_per_cell")
    descriptor = cv2.HOGDescriptor(
        _winSize=(width, height),
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell,
        _nbins=orientations,
        _derivAperture=1,
        _winSigma=-1.0,
        _histogramNormType=cv2.HOGDescriptor_L2Hys,
        _L2HysThreshold=0.2,
        _gammaCorrection=True,
        _nlevels=64,
        _signedGradient=False,
    )
    values = descriptor.compute(gray)
    if values is None:
        raise ValueError("OpenCV HOG returned no values")
    return values.reshape(-1).astype(np.float32)


def _hough_features(gray: np.ndarray, cfg: Mapping[str, Any]):
    threshold_1 = int(cfg.get("canny_threshold_1", 50))
    threshold_2 = int(cfg.get("canny_threshold_2", 150))
    edges = cv2.Canny(gray, threshold_1, threshold_2)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=int(cfg.get("threshold", 25)),
        minLineLength=int(cfg.get("min_line_length", 20)),
        maxLineGap=int(cfg.get("max_line_gap", 8)),
    )
    bins = int(cfg.get("angle_bins", 36))
    if lines is None:
        angles = np.empty(0, dtype=np.float32)
        lengths = np.empty(0, dtype=np.float32)
    else:
        segments = lines.reshape(-1, 4).astype(np.float32)
        dx = segments[:, 2] - segments[:, 0]
        dy = segments[:, 3] - segments[:, 1]
        lengths = np.hypot(dx, dy)
        angles = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)
    histogram, _ = np.histogram(angles, bins=bins, range=(0.0, 180.0), weights=lengths if len(lengths) else None)
    histogram = histogram.astype(np.float32)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    diagonal = math.hypot(gray.shape[1], gray.shape[0])
    horizontal = np.mean((angles < 15) | (angles >= 165)) if len(angles) else 0.0
    vertical = np.mean((angles >= 75) & (angles < 105)) if len(angles) else 0.0
    positive_diag = np.mean((angles >= 15) & (angles < 75)) if len(angles) else 0.0
    negative_diag = np.mean((angles >= 105) & (angles < 165)) if len(angles) else 0.0
    doubled = np.radians(angles * 2.0)
    dominant_bin = int(np.argmax(histogram)) if histogram.sum() else 0
    stats = np.asarray([
        min(len(angles) / 100.0, 1.0),
        dominant_bin / max(1, bins - 1),
        horizontal,
        vertical,
        positive_diag,
        negative_diag,
        float(lengths.mean() / diagonal) if len(lengths) else 0.0,
        float(lengths.std() / diagonal) if len(lengths) else 0.0,
        float(lengths.max() / diagonal) if len(lengths) else 0.0,
        float(np.cos(doubled).mean()) if len(doubled) else 0.0,
        float(np.sin(doubled).mean()) if len(doubled) else 0.0,
        float(np.count_nonzero(edges) / edges.size),
    ], dtype=np.float32)
    diagnostics = {"hough_line_count": int(len(angles)), "hough_dominant_bin": dominant_bin}
    return np.concatenate([histogram, stats]), edges, diagnostics


def _projection_features(gray: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = binary.mean(axis=1).astype(np.float32)
    vertical = binary.mean(axis=0).astype(np.float32)
    bins = int(cfg.get("summary_bins", 64))
    horizontal_resampled = _resample_vector(horizontal, bins)
    vertical_resampled = _resample_vector(vertical, bins)
    stats = np.asarray([
        horizontal.mean(), horizontal.std(), horizontal.max(initial=0), _entropy(horizontal),
        vertical.mean(), vertical.std(), vertical.max(initial=0), _entropy(vertical),
    ], dtype=np.float32)
    return np.concatenate([horizontal_resampled, vertical_resampled, stats])


def _directional_edge_features(gray: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    total = float(magnitude.sum()) + 1e-8
    masks = [
        (angle < 22.5) | (angle >= 157.5),
        (angle >= 67.5) & (angle < 112.5),
        (angle >= 22.5) & (angle < 67.5),
        (angle >= 112.5) & (angle < 157.5),
    ]
    return np.asarray([float(magnitude[mask].sum() / total) for mask in masks], dtype=np.float32)


def _geometric_features(gray, edges, width, height):
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    maximum = max(width, height, 1)
    return np.asarray([
        float(width / max(height, 1)),
        float(width / maximum),
        float(height / maximum),
        float(ink.mean()),
        float(np.count_nonzero(edges) / edges.size),
    ], dtype=np.float32)


def _resample_vector(values: np.ndarray, bins: int) -> np.ndarray:
    if len(values) == bins:
        return values.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=bins, dtype=np.float32)
    return np.interp(new_x, old_x, values).astype(np.float32)


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log2(positive)).sum() / max(1.0, math.log2(len(values))))


def _feature_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Rotation Feature Extraction",
        "",
        f"- Strategy: {summary['feature_strategy']}",
        f"- Version: {summary['feature_version']}",
        f"- Fixed feature dimension: {summary['feature_dimension']}",
        f"- Configuration hash: {summary['configuration_hash']}",
        f"- NaN values: {summary['nan_count']}",
        f"- Infinite values: {summary['infinity_count']}",
        f"- Failed vectors: {summary['missing_vectors']}",
        "",
        "The default vector concatenates spatial OpenCV HOG, a length-weighted",
        "Hough orientation histogram and line statistics, horizontal and vertical",
        "projection profiles, directional edge densities, and page geometry.",
        "Feature extraction performs no fitting and does not use validation, test,",
        "or private labels to alter the representation.",
        "",
        "## Counts by split",
        "",
    ]
    for split, count in sorted(summary["counts_per_split"].items()):
        lines.append(f"- {split}: {count}")
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))
