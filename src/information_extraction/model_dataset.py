"""Prepare PaddleOCR-aligned, public-only LayoutXLM training examples."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src import config as cfgmod
from src.information_extraction.alignment import align_ocr_to_annotations
from src.ocr.cache import OCRCache
from src.ocr.model_registry import ModelRegistry, REQUIRED_MODEL_NAMES
from src.ocr.pipeline import MultilingualOCR
from src.rotation_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    deterministic_rank,
    read_csv_rows,
    sha256_file,
    UnionFind,
)

PROFILE_LIMITS = {"smoke": 32, "development": 1_500, "final": 0}
MODEL_DATA_PREPROCESSING_VERSION = "2.0-cardinal-polygon-fine"
PUBLIC_MODEL_SPLITS = (
    ("train", 0.70),
    ("dev_select", 0.10),
    ("dev_calibration", 0.05),
    ("test_in_domain", 0.15),
)
MODEL_MANIFEST_COLUMNS = (
    "example_id", "build_id", "document_id", "page_id", "dataset", "document_type",
    "language", "project_split", "split_group_id", "token_source", "image_path",
    "normalized_annotation_path", "model_example_path", "ocr_route", "token_count",
    "label_count", "entity_count", "relation_count", "canonical_field_count",
    "alignment_coverage", "entity_retention_rate", "relation_retention_rate",
    "canonical_retention_rate", "data_quality_score", "inference_realistic",
    "is_private", "is_usable", "exclusion_reason", "profile",
)
IE_SPLIT_MANIFEST_COLUMNS = (
    "document_id", "page_id", "dataset", "dataset_component", "document_type", "language",
    "image_path", "normalized_annotation_path", "project_split", "split_group_id",
    "duplicate_group_id", "sha256", "is_private", "is_usable",
)


def profile_manifest_path(metadata_root: str | Path, profile: str) -> Path:
    """Return the non-overlapping manifest path for a model-data profile."""
    names = {
        "smoke": "model_dataset_smoke_manifest.csv",
        "development": "model_dataset_development_manifest.csv",
        "final": "final_model_dataset_manifest.csv",
    }
    try:
        name = names[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported model-data profile: {profile}") from exc
    return Path(metadata_root) / name


def validate_manifest_profile(
    rows: list[Mapping[str, Any]],
    *,
    expected_profile: str,
    expected_build_id: str | None = None,
) -> None:
    """Refuse empty, mixed, stale, or incorrectly selected build manifests."""
    if not rows:
        raise ValueError("model-data manifest is empty")
    profiles = {str(row.get("profile", "")) for row in rows}
    if profiles != {expected_profile}:
        raise ValueError(
            f"manifest profile mismatch: expected {expected_profile!r}, found {sorted(profiles)!r}"
        )
    build_ids = {str(row.get("build_id", "")) for row in rows}
    if len(build_ids) != 1 or "" in build_ids:
        raise ValueError(f"manifest build IDs are missing or mixed: {sorted(build_ids)!r}")
    if expected_build_id is not None and build_ids != {expected_build_id}:
        raise ValueError(
            f"manifest build mismatch: expected {expected_build_id!r}, found {sorted(build_ids)!r}"
        )


def validate_profile_requirements(
    rows: list[Mapping[str, str]],
    *,
    profile: str,
    requested_limit: int = 0,
) -> None:
    """Enforce the public-corpus minimums before any expensive model work."""
    if profile not in PROFILE_LIMITS:
        raise ValueError(f"unsupported model-data profile: {profile}")
    if requested_limit < 0:
        raise ValueError("model-data limit must be non-negative")
    if any(str(row.get("is_private", "false")).casefold() != "false" for row in rows):
        raise ValueError("private or unmarked rows are refused from public model profiles")
    if profile == "final":
        if requested_limit:
            raise ValueError("final profile refuses any page limit")
        if len(rows) < 2_000:
            raise ValueError("final profile requires at least 2,000 leakage-safe public pages")
        datasets = {str(row.get("dataset", "")).casefold() for row in rows}
        if len(datasets) < 3:
            raise ValueError("final profile requires at least three public datasets")
    if profile == "development" and len(rows) < 500:
        raise ValueError("development profile requires at least 500 leakage-safe public pages")


def compute_dataset_build_id(
    rows: list[Mapping[str, str]],
    *,
    profile: str,
    streams: tuple[str, ...],
    split_manifest_sha256: str,
    build_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Bind a build directory to its inventory, split, profile, and streams."""
    inventory = [
        {
            key: str(row.get(key, ""))
            for key in (
                "page_id",
                "document_id",
                "dataset",
                "image_path",
                "normalized_annotation_path",
                "sha256",
                "project_split",
                "split_group_id",
            )
        }
        for row in sorted(rows, key=lambda value: str(value.get("page_id", "")))
    ]
    material = {
        "schema_version": "1.0",
        "profile": profile,
        "streams": sorted(set(streams)),
        "split_manifest_sha256": split_manifest_sha256,
        "build_provenance": dict(build_provenance or {}),
        "inventory": inventory,
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{profile}-{digest[:16]}"


def validate_reusable_example(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> None:
    """Reject reuse when freshly derived model-example content has drifted."""
    if dict(existing) != dict(expected):
        raise ValueError(
            f"stale or content-mismatched model example refused: {output_path}; "
            "rerun with --force after reviewing the source change"
        )


def select_ocr_variant_rows(
    rows: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Select a deterministic dataset/split-balanced OCR-variant subset."""
    if limit < 0:
        raise ValueError("OCR variant limit must be non-negative")
    return _balanced_selection(rows, limit)


def assign_leakage_safe_splits(
    rows: list[Mapping[str, str]],
    *,
    seed: int = 42,
    unseen_datasets: set[str] | None = None,
) -> list[dict[str, str]]:
    """Assign deterministic splits after unioning all known leakage identities."""
    unseen = {value.casefold() for value in (unseen_datasets or set())}
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["page_id"])
    page_ids = [row["page_id"] for row in ordered]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("page_id values must be unique before split assignment")

    groups = UnionFind(page_ids)
    identity_owner: dict[str, str] = {}
    for row in ordered:
        page_id = row["page_id"]
        dataset = str(row.get("dataset", "")).casefold()
        identities = []
        if row.get("document_id"):
            identities.append(f"document:{dataset}:{row['document_id']}")
        if row.get("duplicate_group_id"):
            identities.append(f"duplicate:{dataset}:{row['duplicate_group_id']}")
        if row.get("sha256"):
            identities.append(f"sha256:{str(row['sha256']).casefold()}")
        for identity in identities:
            owner = identity_owner.setdefault(identity, page_id)
            groups.union(page_id, owner)

    rows_by_page = {row["page_id"]: row for row in ordered}
    components: list[dict[str, Any]] = []
    for members in groups.groups().values():
        stable_members = sorted(members)
        group_digest = hashlib.sha256("\n".join(stable_members).encode("utf-8")).hexdigest()
        group_id = f"split_{group_digest[:16]}"
        datasets = {
            str(rows_by_page[page_id].get("dataset", "")).casefold()
            for page_id in stable_members
        }
        components.append({
            "members": stable_members,
            "digest": group_digest,
            "group_id": group_id,
            "datasets": datasets,
            "contribution": Counter(
                str(rows_by_page[page_id].get("dataset", "")).casefold()
                for page_id in stable_members
            ),
        })

    assignment: dict[str, tuple[str, str]] = {}
    assignable = []
    for component in components:
        if component["datasets"] & unseen:
            for page_id in component["members"]:
                assignment[page_id] = ("unseen_domain_test", component["group_id"])
        else:
            assignable.append(component)

    totals: Counter[str] = Counter()
    for component in assignable:
        totals.update(component["contribution"])
    ratios = dict(PUBLIC_MODEL_SPLITS)
    targets = {
        dataset: _target_split_counts(count, ratios)
        for dataset, count in totals.items()
    }
    assigned = {dataset: Counter() for dataset in totals}
    split_order = [name for name, _ in PUBLIC_MODEL_SPLITS]
    for component in sorted(
        assignable,
        key=lambda value: (
            -len(value["members"]),
            deterministic_rank(str(value["digest"]), seed),
        ),
    ):
        costs: list[tuple[float, int, str]] = []
        for order_index, split in enumerate(split_order):
            cost = 0.0
            for dataset, amount in component["contribution"].items():
                target = max(1, targets[dataset][split])
                before = assigned[dataset][split]
                after = before + amount
                cost += ((after - target) / target) ** 2 - ((before - target) / target) ** 2
            costs.append((cost, order_index, split))
        chosen = min(costs)[2]
        for dataset, amount in component["contribution"].items():
            assigned[dataset][chosen] += amount
        for page_id in component["members"]:
            assignment[page_id] = (chosen, component["group_id"])

    result = []
    for row in ordered:
        updated = dict(row)
        updated["project_split"], updated["split_group_id"] = assignment[row["page_id"]]
        result.append(updated)
    return result


def _target_split_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(math.floor(value)) for split, value in exact.items()}
    remaining = total - sum(counts.values())
    order = [name for name, _ in PUBLIC_MODEL_SPLITS]
    for split in sorted(order, key=lambda name: (-(exact[name] - counts[name]), order.index(name))):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    return counts


def build_ground_truth_example(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    *,
    profile: str,
    split_group_id: str,
) -> dict[str, Any]:
    """Build an immutable source-token example without invoking OCR."""
    _refuse_private(row, annotation)
    tokens = copy.deepcopy(list(annotation.get("tokens", [])))
    token_ids = {str(token.get("id", "")) for token in tokens}
    entities = copy.deepcopy(list(annotation.get("entities", [])))
    entity_for_token: dict[str, str] = {}
    for entity in entities:
        for token_id in entity.get("token_ids", []):
            entity_for_token.setdefault(str(token_id), str(entity.get("id", "")))
    labels = [str(token.get("entity_label", "OTHER")) for token in tokens]
    entity_ids = [entity_for_token.get(str(token.get("id", ""))) for token in tokens]
    canonical_fields = copy.deepcopy(annotation.get("canonical_fields", {}))
    for field in canonical_fields.values():
        if isinstance(field, dict):
            field_token_ids = {str(value) for value in field.get("token_ids", [])}
            if not field_token_ids <= token_ids:
                raise ValueError("canonical field references a missing source token")
    return {
        **_example_identity(row, annotation, profile, split_group_id),
        "example_id": f"{row['page_id']}__ground_truth",
        "token_source": "ground_truth",
        "tokens": tokens,
        "labels": labels,
        "entity_ids": entity_ids,
        "entities": entities,
        "relations": copy.deepcopy(list(annotation.get("relations", []))),
        "canonical_fields": canonical_fields,
        "alignment_coverage": 1.0,
        "token_alignment_score": 1.0,
        "ocr_confidence": None,
        "entity_retention_rate": 1.0,
        "relation_retention_rate": 1.0,
        "canonical_retention_rate": 1.0,
        "data_quality_score": 1.0,
        "has_valid_learning_target": bool(tokens and (entities or canonical_fields)),
        "source_targets": _source_targets(annotation),
        "inference_realistic": False,
        "training_only": False,
        "token_loss_mask": [True] * len(tokens),
        "is_private": False,
    }


def build_paddleocr_example(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    ocr: Mapping[str, Any],
    alignment: Mapping[str, Any],
    *,
    route: str,
    profile: str,
    split_group_id: str,
    model_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build an inference-realistic OCR stream with masked partial supervision."""
    _refuse_private(row, annotation)
    source_tokens = list(annotation.get("tokens", []))
    ocr_words = list(ocr.get("words", []))
    tokens = [
        {
            "id": word["id"],
            "text": word.get("text", ""),
            "polygon": copy.deepcopy(word.get("polygon", [])),
            "bbox": copy.deepcopy(word.get("bbox", [])),
            "confidence": float(word.get("confidence", 0.0)),
        }
        for word in ocr_words
    ]
    source_index_by_id = {
        str(token.get("id", "")): index for index, token in enumerate(source_tokens)
    }
    source_to_ocr: dict[int, set[int]] = defaultdict(set)
    for match in alignment.get("matches", []):
        source_to_ocr[int(match["annotation_index"])].add(int(match["ocr_index"]))
    ocr_to_source: dict[int, set[int]] = defaultdict(set)
    for source_index, ocr_indices in source_to_ocr.items():
        for ocr_index in ocr_indices:
            ocr_to_source[ocr_index].add(source_index)
    for ocr_index, token in enumerate(tokens):
        token["origin"] = "paddleocr"
        token["source_token_ids"] = [
            str(source_tokens[source_index].get("id", ""))
            for source_index in sorted(ocr_to_source.get(ocr_index, set()))
        ]

    entities: list[dict[str, Any]] = []
    entity_by_ocr_index: dict[int, str] = {}
    retained_entity_ids: set[str] = set()
    for source_entity in annotation.get("entities", []):
        source_indices = [
            source_index_by_id[str(token_id)]
            for token_id in source_entity.get("token_ids", [])
            if str(token_id) in source_index_by_id
        ]
        ocr_indices = sorted(
            {ocr_index for source_index in source_indices for ocr_index in source_to_ocr.get(source_index, set())}
        )
        if not ocr_indices:
            continue
        entity = copy.deepcopy(dict(source_entity))
        entity["source_token_ids"] = list(source_entity.get("token_ids", []))
        entity["ocr_token_indices"] = ocr_indices
        entity["token_ids"] = [str(tokens[index]["id"]) for index in ocr_indices]
        entities.append(entity)
        entity_id = str(entity.get("id", ""))
        retained_entity_ids.add(entity_id)
        for ocr_index in ocr_indices:
            entity_by_ocr_index.setdefault(ocr_index, entity_id)

    relations = [
        copy.deepcopy(dict(relation))
        for relation in annotation.get("relations", [])
        if str(relation.get("source_id", "")) in retained_entity_ids
        and str(relation.get("target_id", "")) in retained_entity_ids
    ]

    canonical_fields: dict[str, Any] = {}
    retained_canonical = 0
    for name, source_field in annotation.get("canonical_fields", {}).items():
        field = copy.deepcopy(source_field)
        if not isinstance(field, dict):
            canonical_fields[name] = field
            continue
        source_ids = [str(token_id) for token_id in field.get("token_ids", [])]
        mapped_indices = sorted(
            {
                ocr_index
                for token_id in source_ids
                for ocr_index in source_to_ocr.get(source_index_by_id.get(token_id, -1), set())
            }
        )
        field["source_token_ids"] = source_ids
        field["token_ids"] = [str(tokens[index]["id"]) for index in mapped_indices]
        field["evidence_valid"] = bool(field["token_ids"])
        retained_canonical += int(field["evidence_valid"])
        canonical_fields[name] = field

    source_entity_count = len(annotation.get("entities", []))
    source_relation_count = len(annotation.get("relations", []))
    source_canonical_count = len(annotation.get("canonical_fields", {}))
    entity_retention = len(entities) / max(1, source_entity_count) if source_entity_count else 1.0
    relation_retention = len(relations) / max(1, source_relation_count) if source_relation_count else 1.0
    canonical_retention = (
        retained_canonical / max(1, source_canonical_count) if source_canonical_count else 1.0
    )
    match_scores = [float(match.get("score", 0.0)) for match in alignment.get("matches", [])]
    alignment_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
    ocr_confidence = (
        sum(float(token["confidence"]) for token in tokens) / len(tokens) if tokens else 0.0
    )
    coverage = float(alignment.get("alignment_coverage", 0.0))
    quality = sum(
        (coverage, alignment_score, ocr_confidence, entity_retention, relation_retention, canonical_retention)
    ) / 6.0
    labels = list(alignment.get("ocr_labels", ["OTHER"] * len(tokens)))
    entity_ids = [entity_by_ocr_index.get(index) for index in range(len(tokens))]
    detector = str(ocr.get("detector_model", ""))
    recognizer = str(ocr.get("recognizer_model", ""))
    has_valid_target = bool(
        entities
        or relations
        or any(
            isinstance(field, Mapping) and field.get("evidence_valid")
            for field in canonical_fields.values()
        )
    )
    return {
        **_example_identity(row, annotation, profile, split_group_id),
        "example_id": f"{row['page_id']}__paddleocr",
        "token_source": "paddleocr",
        "inference_realistic": True,
        "training_only": False,
        "tokens": tokens,
        "labels": labels,
        "token_loss_mask": [bool(token["source_token_ids"]) for token in tokens],
        "entity_ids": entity_ids,
        "entities": entities,
        "relations": relations,
        "canonical_fields": canonical_fields,
        "alignment": copy.deepcopy({key: value for key, value in alignment.items() if key != "ocr_labels"}),
        "alignment_coverage": coverage,
        "token_alignment_score": alignment_score,
        "ocr_confidence": ocr_confidence,
        "entity_retention_rate": entity_retention,
        "relation_retention_rate": relation_retention,
        "canonical_retention_rate": canonical_retention,
        "data_quality_score": quality,
        "has_valid_learning_target": has_valid_target,
        "source_targets": _source_targets(annotation),
        "ocr_provenance": _ocr_example_provenance(
            ocr, route=route, model_hashes=model_hashes
        ),
        "is_private": False,
    }


def build_hybrid_example(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    ocr: Mapping[str, Any],
    alignment: Mapping[str, Any],
    *,
    route: str,
    profile: str,
    split_group_id: str,
    model_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a training-only OCR-noisy stream with explicit source fallbacks."""
    _refuse_private(row, annotation)
    source_tokens = list(annotation.get("tokens", []))
    ocr_words = list(ocr.get("words", []))
    source_to_ocr: dict[int, list[int]] = defaultdict(list)
    match_scores = []
    for match in alignment.get("matches", []):
        source_index = int(match["annotation_index"])
        ocr_index = int(match["ocr_index"])
        source_to_ocr[source_index].append(ocr_index)
        match_scores.append(float(match.get("score", 0.0)))

    tokens: list[dict[str, Any]] = []
    labels: list[str] = []
    entity_ids: list[str | None] = []
    source_to_stream_ids: dict[str, list[str]] = defaultdict(list)
    source_entity_by_token: dict[str, str] = {}
    for entity in annotation.get("entities", []):
        for token_id in entity.get("token_ids", []):
            source_entity_by_token.setdefault(str(token_id), str(entity.get("id", "")))

    used_ocr_indices: set[int] = set()
    fallback_count = 0
    for source_index, source_token in enumerate(source_tokens):
        source_id = str(source_token.get("id", ""))
        mapped = [
            index
            for index in sorted(set(source_to_ocr.get(source_index, [])))
            if index not in used_ocr_indices
        ]
        if mapped:
            for ocr_index in mapped:
                used_ocr_indices.add(ocr_index)
                word = ocr_words[ocr_index]
                token_id = str(word["id"])
                tokens.append({
                    "id": token_id,
                    "text": word.get("text", ""),
                    "polygon": copy.deepcopy(word.get("polygon", [])),
                    "bbox": copy.deepcopy(word.get("bbox", [])),
                    "confidence": float(word.get("confidence", 0.0)),
                    "origin": "paddleocr",
                    "source_token_ids": [source_id],
                    "teacher_supplied": False,
                })
                labels.append(str(source_token.get("entity_label", "OTHER")))
                entity_ids.append(source_entity_by_token.get(source_id))
                source_to_stream_ids[source_id].append(token_id)
        else:
            fallback_count += 1
            token_id = f"fallback__{source_id}"
            tokens.append({
                "id": token_id,
                "text": source_token.get("text", ""),
                "polygon": copy.deepcopy(source_token.get("polygon", [])),
                "bbox": copy.deepcopy(source_token.get("bbox", [])),
                "confidence": None,
                "origin": "ground_truth_fallback",
                "source_token_ids": [source_id],
                "teacher_supplied": True,
            })
            labels.append(str(source_token.get("entity_label", "OTHER")))
            entity_ids.append(source_entity_by_token.get(source_id))
            source_to_stream_ids[source_id].append(token_id)

    for ocr_index, word in enumerate(ocr_words):
        if ocr_index in used_ocr_indices:
            continue
        tokens.append({
            "id": str(word["id"]),
            "text": word.get("text", ""),
            "polygon": copy.deepcopy(word.get("polygon", [])),
            "bbox": copy.deepcopy(word.get("bbox", [])),
            "confidence": float(word.get("confidence", 0.0)),
            "origin": "paddleocr_extra",
            "source_token_ids": [],
            "teacher_supplied": False,
        })
        labels.append("OTHER")
        entity_ids.append(None)

    entities: list[dict[str, Any]] = []
    for source_entity in annotation.get("entities", []):
        entity = copy.deepcopy(dict(source_entity))
        source_ids = [str(token_id) for token_id in source_entity.get("token_ids", [])]
        entity["source_token_ids"] = source_ids
        entity["token_ids"] = [
            stream_id for source_id in source_ids for stream_id in source_to_stream_ids[source_id]
        ]
        entities.append(entity)

    canonical_fields: dict[str, Any] = {}
    for name, source_field in annotation.get("canonical_fields", {}).items():
        field = copy.deepcopy(source_field)
        if isinstance(field, dict):
            source_ids = [str(token_id) for token_id in field.get("token_ids", [])]
            field["source_token_ids"] = source_ids
            field["token_ids"] = [
                stream_id for source_id in source_ids for stream_id in source_to_stream_ids[source_id]
            ]
            field["evidence_valid"] = bool(field["token_ids"])
        canonical_fields[name] = field

    ocr_confidences = [
        float(token["confidence"])
        for token in tokens
        if token["confidence"] is not None and token["origin"] == "paddleocr"
    ]
    ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
    coverage = float(alignment.get("alignment_coverage", 0.0))
    alignment_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
    fallback_fraction = fallback_count / max(1, len(source_tokens))
    detector = str(ocr.get("detector_model", ""))
    recognizer = str(ocr.get("recognizer_model", ""))
    return {
        **_example_identity(row, annotation, profile, split_group_id),
        "example_id": f"{row['page_id']}__hybrid",
        "token_source": "hybrid",
        "inference_realistic": False,
        "training_only": True,
        "tokens": tokens,
        "labels": labels,
        "token_loss_mask": [bool(token["source_token_ids"]) for token in tokens],
        "entity_ids": entity_ids,
        "entities": entities,
        "relations": copy.deepcopy(list(annotation.get("relations", []))),
        "canonical_fields": canonical_fields,
        "alignment": copy.deepcopy({key: value for key, value in alignment.items() if key != "ocr_labels"}),
        "alignment_coverage": coverage,
        "token_alignment_score": alignment_score,
        "ocr_confidence": ocr_confidence,
        "ground_truth_fallback_fraction": fallback_fraction,
        "entity_retention_rate": 1.0,
        "relation_retention_rate": 1.0,
        "canonical_retention_rate": 1.0,
        "data_quality_score": sum((coverage, alignment_score, ocr_confidence, 1.0 - fallback_fraction)) / 4.0,
        "has_valid_learning_target": bool(tokens and (entities or canonical_fields)),
        "source_targets": _source_targets(annotation),
        "ocr_provenance": _ocr_example_provenance(
            ocr, route=route, model_hashes=model_hashes
        ),
        "is_private": False,
    }


def _source_targets(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tokens": copy.deepcopy(list(annotation.get("tokens", []))),
        "entities": copy.deepcopy(list(annotation.get("entities", []))),
        "relations": copy.deepcopy(list(annotation.get("relations", []))),
        "canonical_fields": copy.deepcopy(annotation.get("canonical_fields", {})),
    }


def _ocr_example_provenance(
    ocr: Mapping[str, Any],
    *,
    route: str,
    model_hashes: Mapping[str, str],
) -> dict[str, Any]:
    detector = str(ocr.get("detector_model", ""))
    recognizer = str(ocr.get("recognizer_model", ""))
    fine_deskew = ocr.get("fine_deskew")
    return {
        "route": route,
        "selected_route": str(ocr.get("language_route", route)),
        "detector": detector,
        "recognizer": recognizer,
        "result_hash": str(ocr.get("provenance_hash", "")),
        "detector_artifact_hash": str(model_hashes.get(detector, "")),
        "recognizer_artifact_hash": str(model_hashes.get(recognizer, "")),
        "selected_orientation": float(ocr.get("orientation", 0.0) or 0.0),
        "fine_deskew": copy.deepcopy(fine_deskew) if isinstance(fine_deskew, Mapping) else None,
        "orientation_policy": "cardinal_plus_polygon_fine_deskew",
        "kmeans_controls_ocr": False,
    }


def _example_identity(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    profile: str,
    split_group_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "document_id": row["document_id"],
        "page_id": row["page_id"],
        "dataset": row["dataset"],
        "document_type": row["document_type"],
        "language": row["language"],
        "project_split": row["project_split"],
        "split_group_id": split_group_id,
        "profile": profile,
        "image_path": row["image_path"],
        "width": int(annotation["page"]["width"]),
        "height": int(annotation["page"]["height"]),
    }


def _refuse_private(row: Mapping[str, Any], annotation: Mapping[str, Any]) -> None:
    if str(row.get("is_private", "false")).casefold() == "true" or annotation.get("is_private") is not False:
        raise ValueError("private or unmarked annotations are refused for model fitting")


def prepare_model_dataset(
    cfg: Mapping[str, Any],
    registry: ModelRegistry,
    *,
    profile: str = "smoke",
    device: str = "cpu",
    limit: int = 0,
    force: bool = False,
    streams: tuple[str, ...] = ("ground_truth",),
    ocr_pipeline: MultilingualOCR | None = None,
    ocr_variant_limit: int = 0,
) -> dict[str, Any]:
    if profile not in PROFILE_LIMITS:
        raise ValueError(f"unsupported model-data profile: {profile}")
    selected_streams = tuple(sorted(set(streams)))
    supported_streams = {"ground_truth", "paddleocr", "hybrid"}
    if not selected_streams or not set(selected_streams) <= supported_streams:
        raise ValueError(f"unsupported model-data streams: {selected_streams!r}")
    if ocr_variant_limit < 0:
        raise ValueError("OCR variant limit must be non-negative")
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    source_manifest = metadata / "information_extraction_manifest.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    all_rows = read_csv_rows(source_manifest)
    if any(row.get("is_private") == "true" and row.get("project_split") != "private_test" for row in all_rows):
        raise ValueError("private row is assigned to a public model split")
    eligible = [
        row for row in all_rows
        if row.get("is_private") == "false"
        and row.get("is_usable") == "true"
        and row.get("normalized_annotation_path")
    ]
    assigned = assign_leakage_safe_splits(eligible, seed=42, unseen_datasets={"coru"})
    split_manifest_path = metadata / "information_extraction_split_manifest.csv"
    atomic_write_csv(split_manifest_path, assigned, IE_SPLIT_MANIFEST_COLUMNS)
    split_manifest_sha256 = sha256_file(split_manifest_path)
    public_fit_splits = {name for name, _ in PUBLIC_MODEL_SPLITS}
    candidates = [
        row for row in assigned
        if row.get("project_split") in public_fit_splits
        and str(row.get("dataset", "")).casefold() != "coru"
    ]
    cap = limit or PROFILE_LIMITS[profile]
    candidates = _balanced_selection(candidates, cap)
    validate_profile_requirements(candidates, profile=profile, requested_limit=limit)
    requires_ocr = bool({"paddleocr", "hybrid"} & set(selected_streams))
    if ocr_variant_limit and not requires_ocr:
        raise ValueError("OCR variant limit requires a paddleocr or hybrid stream")
    variant_rows = (
        select_ocr_variant_rows(candidates, ocr_variant_limit)
        if requires_ocr
        else []
    )
    variant_page_ids = {row["page_id"] for row in variant_rows}
    variant_selection_sha256 = hashlib.sha256(
        "\n".join(sorted(variant_page_ids)).encode("utf-8")
    ).hexdigest()
    build_provenance: dict[str, Any] = {}
    if requires_ocr:
        build_provenance = {
            "preprocessing_version": MODEL_DATA_PREPROCESSING_VERSION,
            "orientation_policy": {
                "cardinal_angles": [0, 90, 180, 270],
                "polygon_fine_deskew": True,
                "kmeans_controls_ocr": False,
            },
            "ocr_model_artifact_hashes": {
                name: registry.require(name).artifact_hash
                for name in REQUIRED_MODEL_NAMES
            },
            "ocr_variant_selection": {
                "requested_limit": ocr_variant_limit,
                "selected_page_count": len(variant_rows),
                "page_ids_sha256": variant_selection_sha256,
                "balance_keys": ["dataset", "project_split"],
            },
        }
    build_id = compute_dataset_build_id(
        candidates,
        profile=profile,
        streams=selected_streams,
        split_manifest_sha256=split_manifest_sha256,
        build_provenance=build_provenance,
    )
    output_root = cfgmod.resolve_path(cfg, "model_datasets") / profile / build_id
    output_root.mkdir(parents=True, exist_ok=True)
    cache = OCRCache(cfgmod.resolve_path(cfg, "ocr_cache"))
    if requires_ocr and ocr_pipeline is None:
        ocr_pipeline = MultilingualOCR(
            registry=registry,
            device=device,
            cache=cache,
            preprocessing_version=MODEL_DATA_PREPROCESSING_VERSION,
        )
    manifest_rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in candidates:
        annotation_path = root / row["normalized_annotation_path"]
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifest_rows.append(_excluded_row(
                row,
                profile,
                f"normalized_annotation_read_error:{type(exc).__name__}",
                build_id=build_id,
            ))
            exclusions["normalized_annotation_read_error"] += 1
            continue
        if not annotation.get("tokens"):
            manifest_rows.append(_excluded_row(
                row,
                profile,
                "source_token_geometry_required_for_fit_streams",
                build_id=build_id,
            ))
            exclusions["source_token_geometry_required_for_fit_streams"] += 1
            continue
        route = "thai" if str(row.get("language", "")).lower() in {"th", "thai"} else "general"
        ocr_result: dict[str, Any] | None = None
        alignment: dict[str, Any] | None = None
        model_hashes: dict[str, str] = {}
        row_has_ocr_variant = row["page_id"] in variant_page_ids
        if requires_ocr and row_has_ocr_variant:
            image_path = root / row["image_path"]
            assert ocr_pipeline is not None
            ocr_result = predict_model_data_ocr(ocr_pipeline, image_path, route=route)
            alignment = align_ocr_to_annotations(ocr_result["words"], annotation["tokens"])
            for model_name in (
                str(ocr_result.get("detector_model", "")),
                str(ocr_result.get("recognizer_model", "")),
            ):
                if model_name:
                    model_hashes[model_name] = registry.require(model_name).artifact_hash

        for stream in selected_streams:
            if stream != "ground_truth" and not row_has_ocr_variant:
                continue
            if stream == "ground_truth":
                example = build_ground_truth_example(
                    row,
                    annotation,
                    profile=profile,
                    split_group_id=row["split_group_id"],
                )
            elif stream == "paddleocr":
                assert ocr_result is not None and alignment is not None
                example = build_paddleocr_example(
                    row,
                    annotation,
                    ocr_result,
                    alignment,
                    route=route,
                    profile=profile,
                    split_group_id=row["split_group_id"],
                    model_hashes=model_hashes,
                )
            else:
                assert ocr_result is not None and alignment is not None
                example = build_hybrid_example(
                    row,
                    annotation,
                    ocr_result,
                    alignment,
                    route=route,
                    profile=profile,
                    split_group_id=row["split_group_id"],
                    model_hashes=model_hashes,
                )
            example["build_id"] = build_id
            example["split_manifest_sha256"] = split_manifest_sha256
            output_path = output_root / row["project_split"] / stream / f"{row['page_id']}.json"
            if force or not output_path.is_file():
                atomic_write_json(output_path, example)
            else:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                if (
                    existing.get("build_id") != build_id
                    or existing.get("profile") != profile
                    or existing.get("token_source") != stream
                ):
                    raise ValueError(f"stale or mismatched model example refused: {output_path}")
                validate_reusable_example(
                    existing, example, output_path=output_path
                )
            manifest_rows.append(_manifest_row(
                row,
                example,
                output_path,
                profile=profile,
                build_id=build_id,
                route=route if stream != "ground_truth" else "",
            ))
            counts[f"dataset:{row['dataset']}"] += 1
            counts[f"split:{row['project_split']}"] += 1
            counts[f"stream:{stream}"] += 1
    manifest_path = profile_manifest_path(metadata, profile)
    atomic_write_csv(manifest_path, manifest_rows, MODEL_MANIFEST_COLUMNS)
    manifest_sha256 = sha256_file(manifest_path)
    summary = {
        "schema_version": "2.0", "profile": profile, "build_id": build_id,
        "streams": list(selected_streams), "candidate_page_count": len(candidates),
        "ocr_variant_page_count": len(variant_rows),
        "build_provenance": build_provenance,
        "usable_example_count": sum(row["is_usable"] == "true" for row in manifest_rows),
        "excluded_example_count": sum(row["is_usable"] != "true" for row in manifest_rows),
        "counts_by_dataset": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("dataset:")},
        "counts_by_split": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("split:")},
        "counts_by_stream": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("stream:")},
        "exclusion_counts": dict(exclusions),
        "minimum_alignment_coverage": None,
        "whole_page_alignment_rejection_enabled": False,
        "gmail_fit_rows": 0,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha256,
    }
    atomic_write_json(output_root / "build_complete.json", {
        "schema_version": "1.0",
        "build_id": build_id,
        "profile": profile,
        "manifest_sha256": manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "usable_example_count": summary["usable_example_count"],
    })
    report_root = cfgmod.resolve_path(cfg, "reports") / "final_model"
    atomic_write_json(report_root / f"model_dataset_{profile}_summary.json", summary)
    atomic_write_text(report_root / f"model_dataset_{profile}_report.md", _report(summary))
    legacy_report_root = cfgmod.resolve_path(cfg, "reports") / "information_extraction"
    atomic_write_json(legacy_report_root / "model_dataset_summary.json", summary)
    atomic_write_text(legacy_report_root / "model_dataset_report.md", _report(summary))
    return summary


def predict_model_data_ocr(
    pipeline: Any,
    image_path: Path,
    *,
    route: str,
) -> dict[str, Any]:
    """Use the same verified rotation-aware OCR path used by inference."""
    if route not in {"general", "thai"}:
        raise ValueError(f"unsupported OCR route: {route}")
    return pipeline.extract_path(
        image_path,
        language_mode=route,
        private=False,
    )


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


def _manifest_row(
    row: Mapping[str, str],
    example: Mapping[str, Any],
    output_path: Path,
    *,
    profile: str,
    build_id: str,
    route: str,
) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "build_id": build_id,
        "document_id": row["document_id"],
        "page_id": row["page_id"],
        "dataset": row["dataset"],
        "document_type": row["document_type"],
        "language": row["language"],
        "project_split": row["project_split"],
        "split_group_id": row["split_group_id"],
        "token_source": example["token_source"],
        "image_path": row["image_path"],
        "normalized_annotation_path": row["normalized_annotation_path"],
        "model_example_path": str(output_path),
        "ocr_route": route,
        "token_count": len(example.get("tokens", [])),
        "label_count": len(example.get("labels", [])),
        "entity_count": len(example.get("entities", [])),
        "relation_count": len(example.get("relations", [])),
        "canonical_field_count": len(example.get("canonical_fields", {})),
        "alignment_coverage": example.get("alignment_coverage", 0.0),
        "entity_retention_rate": example.get("entity_retention_rate", 0.0),
        "relation_retention_rate": example.get("relation_retention_rate", 0.0),
        "canonical_retention_rate": example.get("canonical_retention_rate", 0.0),
        "data_quality_score": example.get("data_quality_score", 0.0),
        "inference_realistic": str(bool(example.get("inference_realistic"))).lower(),
        "is_private": "false",
        "is_usable": "true",
        "exclusion_reason": "",
        "profile": profile,
    }


def _excluded_row(
    row: Mapping[str, str], profile: str, reason: str, *, route: str = "",
    token_count: int = 0, coverage: float = 0.0, build_id: str = "",
    token_source: str = "",
) -> dict[str, Any]:
    return {
        "example_id": row["page_id"], "build_id": build_id,
        "document_id": row["document_id"], "page_id": row["page_id"],
        "dataset": row["dataset"], "document_type": row["document_type"], "language": row["language"],
        "project_split": row["project_split"], "split_group_id": row.get("split_group_id", ""),
        "token_source": token_source, "image_path": row.get("image_path", ""),
        "normalized_annotation_path": row.get("normalized_annotation_path", ""),
        "model_example_path": "", "ocr_route": route, "token_count": token_count,
        "label_count": 0, "entity_count": 0, "relation_count": 0,
        "canonical_field_count": 0, "alignment_coverage": coverage,
        "entity_retention_rate": 0.0, "relation_retention_rate": 0.0,
        "canonical_retention_rate": 0.0, "data_quality_score": 0.0,
        "inference_realistic": "false",
        "is_private": "false", "is_usable": "false", "exclusion_reason": reason, "profile": profile,
    }


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Model dataset report", "",
        f"Profile: `{summary['profile']}`.",
        f"Build: `{summary['build_id']}`.",
        f"Streams: `{json.dumps(summary['streams'])}`.",
        f"Candidate source pages: {summary['candidate_page_count']}.",
        f"Usable examples: {summary['usable_example_count']}.",
        f"Excluded examples: {summary['excluded_example_count']}.",
        "Whole-page alignment rejection: **disabled**; partial targets use explicit masks.",
        f"Counts by dataset: `{json.dumps(summary['counts_by_dataset'], sort_keys=True)}`.",
        f"Counts by split: `{json.dumps(summary['counts_by_split'], sort_keys=True)}`.",
        f"Counts by stream: `{json.dumps(summary['counts_by_stream'], sort_keys=True)}`.",
        f"Exclusions: `{json.dumps(summary['exclusion_counts'], sort_keys=True)}`.",
        "Gmail fit rows: **0**.", "",
        "OCR result cache keys include image and exact model artifact hashes, route/orientation configuration, PaddleOCR version, and preprocessing version.",
    ]) + "\n"
