"""Prepare PaddleOCR-aligned, public-only LayoutXLM training examples."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from src import config as cfgmod
from src.information_extraction.alignment import align_ocr_to_annotations
from src.ocr.cache import OCRCache, OCRCacheKey
from src.ocr.model_registry import ModelRegistry
from src.ocr.paddleocr_adapter import PaddleOCRAdapter
from src.rotation_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    deterministic_rank,
    read_csv_rows,
)

PROFILE_LIMITS = {"smoke": 16, "development": 200, "final": 0}
MODEL_MANIFEST_COLUMNS = (
    "example_id", "document_id", "page_id", "dataset", "document_type", "language",
    "project_split", "image_path", "normalized_annotation_path", "model_example_path",
    "ocr_route", "ocr_token_count", "label_count", "relation_count", "alignment_coverage",
    "is_private", "is_usable", "exclusion_reason", "profile",
)


def prepare_model_dataset(
    cfg: Mapping[str, Any],
    registry: ModelRegistry,
    *,
    profile: str = "smoke",
    device: str = "cpu",
    limit: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    if profile not in PROFILE_LIMITS:
        raise ValueError(f"unsupported model-data profile: {profile}")
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    source_manifest = metadata / "information_extraction_manifest.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    all_rows = read_csv_rows(source_manifest)
    if any(row.get("is_private") == "true" and row.get("project_split") != "private_test" for row in all_rows):
        raise ValueError("private row is assigned to a public model split")
    candidates = [
        row for row in all_rows
        if row.get("is_private") == "false"
        and row.get("is_usable") == "true"
        and row.get("normalized_annotation_path")
        and row.get("project_split") in {"train", "validation", "test"}
    ]
    cap = limit or PROFILE_LIMITS[profile]
    candidates = _balanced_selection(candidates, cap)
    output_root = cfgmod.resolve_path(cfg, "model_datasets") / profile
    output_root.mkdir(parents=True, exist_ok=True)
    cache = OCRCache(cfgmod.resolve_path(cfg, "ocr_cache"))
    adapters: dict[str, PaddleOCRAdapter] = {}
    manifest_rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    minimum_coverage = float(cfg.get("information_extraction", {}).get("minimum_alignment_coverage", 0.7))
    for row in candidates:
        annotation_path = root / row["normalized_annotation_path"]
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifest_rows.append(_excluded_row(row, profile, f"normalized_annotation_read_error:{type(exc).__name__}"))
            exclusions["normalized_annotation_read_error"] += 1
            continue
        if not annotation.get("tokens"):
            manifest_rows.append(_excluded_row(row, profile, "requires_ocr_alignment_without_source_token_geometry"))
            exclusions["requires_ocr_alignment_without_source_token_geometry"] += 1
            continue
        image_path = root / row["image_path"]
        route = "thai" if str(row.get("language", "")).lower() in {"th", "thai"} else "general"
        adapter = adapters.get(route)
        if adapter is None:
            adapter = PaddleOCRAdapter(registry, route, device=device)
            adapters[route] = adapter
        ocr_result = _cached_predict(cache, adapter, image_path, route)
        alignment = align_ocr_to_annotations(ocr_result["words"], annotation["tokens"])
        if alignment["alignment_coverage"] < minimum_coverage:
            manifest_rows.append(_excluded_row(
                row, profile, f"alignment_below_{minimum_coverage:.2f}",
                route=route, token_count=len(ocr_result["words"]), coverage=alignment["alignment_coverage"],
            ))
            exclusions["alignment_below_threshold"] += 1
            continue
        example = _build_example(row, annotation, ocr_result, alignment, route, registry)
        output_path = output_root / row["project_split"] / f"{row['page_id']}.json"
        if force or not output_path.is_file():
            atomic_write_json(output_path, example)
        manifest_rows.append({
            "example_id": row["page_id"], "document_id": row["document_id"], "page_id": row["page_id"],
            "dataset": row["dataset"], "document_type": row["document_type"], "language": row["language"],
            "project_split": row["project_split"], "image_path": row["image_path"],
            "normalized_annotation_path": row["normalized_annotation_path"],
            "model_example_path": str(output_path), "ocr_route": route,
            "ocr_token_count": len(example["tokens"]), "label_count": len(example["labels"]),
            "relation_count": len(example["relations"]),
            "alignment_coverage": alignment["alignment_coverage"],
            "is_private": "false", "is_usable": "true", "exclusion_reason": "", "profile": profile,
        })
        counts[f"dataset:{row['dataset']}"] += 1
        counts[f"split:{row['project_split']}"] += 1
    manifest_path = metadata / "model_dataset_manifest.csv"
    atomic_write_csv(manifest_path, manifest_rows, MODEL_MANIFEST_COLUMNS)
    summary = {
        "schema_version": "1.0", "profile": profile, "candidate_count": len(candidates),
        "usable_example_count": sum(row["is_usable"] == "true" for row in manifest_rows),
        "excluded_example_count": sum(row["is_usable"] != "true" for row in manifest_rows),
        "counts_by_dataset": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("dataset:")},
        "counts_by_split": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("split:")},
        "exclusion_counts": dict(exclusions),
        "minimum_alignment_coverage": minimum_coverage,
        "gmail_fit_rows": 0,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
    }
    report_root = cfgmod.resolve_path(cfg, "reports") / "information_extraction"
    atomic_write_json(report_root / "model_dataset_summary.json", summary)
    atomic_write_text(report_root / "model_dataset_report.md", _report(summary))
    return summary


def _cached_predict(
    cache: OCRCache, adapter: PaddleOCRAdapter, image_path: Path, route: str
) -> dict[str, Any]:
    provenance = adapter.provenance()
    key = OCRCacheKey.from_image(
        image_path,
        detector_model=str(provenance["detector_model"]),
        detector_artifact_hash=str(provenance["detector_artifact_hash"]),
        recognizer_model=str(provenance["recognizer_model"]),
        recognizer_artifact_hash=str(provenance["recognizer_artifact_hash"]),
        language_route_configuration={"route": route},
        orientation_configuration={"angles": [0], "purpose": "model_data_alignment"},
        paddleocr_version=str(provenance["paddleocr_version"]),
        preprocessing_version="1.0",
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    with Image.open(image_path) as image:
        result = adapter.predict(image.convert("RGB"), orientation=0.0)
    cache.put(key, result)
    return result


def _build_example(
    row: Mapping[str, str], annotation: Mapping[str, Any], ocr: Mapping[str, Any],
    alignment: Mapping[str, Any], route: str, registry: ModelRegistry,
) -> dict[str, Any]:
    tokens = [
        {
            "id": token["id"], "text": token["text"], "polygon": token["polygon"],
            "bbox": token["bbox"], "confidence": token["confidence"],
        }
        for token in ocr["words"]
    ]
    annotation_index_by_id = {token["id"]: index for index, token in enumerate(annotation["tokens"])}
    annotation_to_ocr: dict[int, list[int]] = defaultdict(list)
    for match in alignment["matches"]:
        annotation_to_ocr[int(match["annotation_index"])].append(int(match["ocr_index"]))
    entities = []
    for entity in annotation.get("entities", []):
        annotation_indices = [annotation_index_by_id[token_id] for token_id in entity.get("token_ids", []) if token_id in annotation_index_by_id]
        ocr_indices = sorted({index for annotation_index in annotation_indices for index in annotation_to_ocr.get(annotation_index, [])})
        if not ocr_indices:
            continue
        entities.append({
            "id": entity["id"], "label": entity["label"], "text": entity["text"],
            "ocr_token_indices": ocr_indices, "bbox": entity["bbox"],
        })
    entity_ids = {entity["id"] for entity in entities}
    relations = [
        dict(relation) for relation in annotation.get("relations", [])
        if relation.get("source_id") in entity_ids and relation.get("target_id") in entity_ids
    ]
    return {
        "schema_version": "1.0", "example_id": row["page_id"],
        "document_id": row["document_id"], "page_id": row["page_id"],
        "dataset": row["dataset"], "document_type": row["document_type"],
        "language": row["language"], "project_split": row["project_split"],
        "image_path": row["image_path"], "width": annotation["page"]["width"],
        "height": annotation["page"]["height"], "tokens": tokens,
        "labels": list(alignment["ocr_labels"]), "entities": entities, "relations": relations,
        "canonical_fields": annotation.get("canonical_fields", {}),
        "alignment": {key: value for key, value in alignment.items() if key != "ocr_labels"},
        "ocr_provenance": {
            "route": route, "detector": ocr["detector_model"], "recognizer": ocr["recognizer_model"],
            "result_hash": ocr["provenance_hash"],
            "detector_artifact_hash": registry.require(ocr["detector_model"]).artifact_hash,
            "recognizer_artifact_hash": registry.require(ocr["recognizer_model"]).artifact_hash,
        },
        "is_private": False,
    }


def _balanced_selection(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: deterministic_rank(row["page_id"], 42))
    if not limit or len(ordered) <= limit:
        return ordered
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        buckets[(row["dataset"], row["project_split"])].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def _excluded_row(
    row: Mapping[str, str], profile: str, reason: str, *, route: str = "",
    token_count: int = 0, coverage: float = 0.0,
) -> dict[str, Any]:
    return {
        "example_id": row["page_id"], "document_id": row["document_id"], "page_id": row["page_id"],
        "dataset": row["dataset"], "document_type": row["document_type"], "language": row["language"],
        "project_split": row["project_split"], "image_path": row.get("image_path", ""),
        "normalized_annotation_path": row.get("normalized_annotation_path", ""),
        "model_example_path": "", "ocr_route": route, "ocr_token_count": token_count,
        "label_count": 0, "relation_count": 0, "alignment_coverage": coverage,
        "is_private": "false", "is_usable": "false", "exclusion_reason": reason, "profile": profile,
    }


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Model dataset report", "",
        f"Profile: `{summary['profile']}`.",
        f"Usable PaddleOCR-aligned examples: {summary['usable_example_count']}.",
        f"Excluded examples: {summary['excluded_example_count']}.",
        f"Minimum source-label alignment coverage: {summary['minimum_alignment_coverage']:.2f}.",
        f"Counts by dataset: `{json.dumps(summary['counts_by_dataset'], sort_keys=True)}`.",
        f"Counts by split: `{json.dumps(summary['counts_by_split'], sort_keys=True)}`.",
        f"Exclusions: `{json.dumps(summary['exclusion_counts'], sort_keys=True)}`.",
        "Gmail fit rows: **0**.", "",
        "OCR result cache keys include image and exact model artifact hashes, route/orientation configuration, PaddleOCR version, and preprocessing version.",
    ]) + "\n"
