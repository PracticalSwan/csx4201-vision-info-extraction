"""LayoutXLM word labels, dynamic boxes, normalization, and window inputs."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.information_extraction.geometry import (
    apply_matrix,
    expanded_rotation_transform,
    polygon_to_bbox,
    transform_annotation,
)

BASE_ENTITY_LABELS = ("HEADER", "KEY", "VALUE", "QUESTION", "ANSWER", "TABLE_CELL", "OTHER")
BIO_LABELS = ("O",) + tuple(
    prefix + label
    for label in BASE_ENTITY_LABELS if label != "OTHER"
    for prefix in ("B-", "I-")
)
LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def to_bio_labels(
    labels: Sequence[str], entity_ids: Sequence[str | None] | None = None
) -> list[str]:
    if entity_ids is not None and len(entity_ids) != len(labels):
        raise ValueError("entity_ids and labels must have equal lengths")
    result: list[str] = []
    previous = "OTHER"
    previous_entity_id: str | None = None
    for index, label in enumerate(labels):
        normalized = label if label in BASE_ENTITY_LABELS else "OTHER"
        entity_id = entity_ids[index] if entity_ids is not None else None
        if normalized == "OTHER":
            result.append("O")
        else:
            same_entity = entity_ids is None or (
                entity_id is not None and entity_id == previous_entity_id
            )
            prefix = "I-" if normalized == previous and same_entity else "B-"
            result.append(prefix + normalized)
        previous = normalized
        previous_entity_id = entity_id
    return result


def normalize_bbox(bbox: Sequence[float], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError("valid page dimensions and a four-value bbox are required")
    x0, y0, x1, y1 = map(float, bbox)
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        raise ValueError(f"bbox outside page bounds: {bbox} vs {width}x{height}")
    return [
        max(0, min(1000, round(x0 * 1000 / width))),
        max(0, min(1000, round(y0 * 1000 / height))),
        max(0, min(1000, round(x1 * 1000 / width))),
        max(0, min(1000, round(y1 * 1000 / height))),
    ]


def rotated_word_boxes(
    example: Mapping[str, Any], angle: float
) -> tuple[list[list[int]], int, int, dict[str, Any]]:
    transform = expanded_rotation_transform(int(example["width"]), int(example["height"]), angle)
    boxes = []
    for token in example["tokens"]:
        polygon = apply_matrix(token["polygon"], transform.forward)
        bbox = polygon_to_bbox(polygon)
        bbox = [
            max(0.0, min(float(transform.output_width), bbox[0])),
            max(0.0, min(float(transform.output_height), bbox[1])),
            max(0.0, min(float(transform.output_width), bbox[2])),
            max(0.0, min(float(transform.output_height), bbox[3])),
        ]
        boxes.append(normalize_bbox(bbox, transform.output_width, transform.output_height))
    return boxes, transform.output_width, transform.output_height, transform.as_dict()


def rotate_example_geometry(
    example: Mapping[str, Any],
    angle: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rotate token and entity geometry while preserving IDs and targets."""
    transform = expanded_rotation_transform(
        int(example["width"]), int(example["height"]), angle
    )
    rotated = transform_annotation(example, transform)
    rotated["width"] = transform.output_width
    rotated["height"] = transform.output_height
    return rotated, transform.as_dict()


def encode_layoutxlm_windows(
    tokenizer: Any,
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    labels: Sequence[int],
    *,
    max_length: int = 512,
    stride: int = 64,
) -> dict[str, Any]:
    if not (len(words) == len(boxes) == len(labels)):
        raise ValueError("words, boxes, and labels must have equal lengths")
    encoding = tokenizer(
        list(words), boxes=[list(box) for box in boxes], word_labels=list(labels),
        truncation=True, padding="max_length", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_tensors=None,
    )
    if "input_ids" not in encoding or "bbox" not in encoding or "labels" not in encoding:
        raise ValueError("LayoutXLM tokenizer did not return required model inputs")
    return dict(encoding)


def load_model_examples(
    manifest_path: str | Path,
    split: str,
    *,
    expected_profile: str | None = None,
    expected_build_id: str | None = None,
    token_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    import csv

    all_rows = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            all_rows.append(row)
    if not all_rows:
        raise ValueError(f"model manifest is empty: {manifest_path}")
    if expected_profile is not None:
        profiles = {row.get("profile", "") for row in all_rows}
        if profiles != {expected_profile}:
            raise ValueError(
                f"model manifest profile mismatch: expected {expected_profile!r}, found {sorted(profiles)!r}"
            )
    if expected_build_id is not None:
        build_ids = {row.get("build_id", "") for row in all_rows}
        if build_ids != {expected_build_id}:
            raise ValueError(
                f"model manifest build mismatch: expected {expected_build_id!r}, found {sorted(build_ids)!r}"
            )
    rows = [
        row
        for row in all_rows
        if row.get("project_split") == split
        and row.get("is_usable") == "true"
        and (token_sources is None or row.get("token_source") in token_sources)
    ]
    examples = []
    for row in rows:
        if row.get("is_private") != "false":
            raise ValueError("private or unmarked manifest row refused")
        path = Path(row["model_example_path"])
        example = json.loads(path.read_text(encoding="utf-8"))
        if example.get("is_private") is not False:
            raise ValueError(f"private or unmarked model example refused: {path}")
        expected = {
            "example_id": row.get("example_id"),
            "build_id": row.get("build_id"),
            "profile": row.get("profile"),
            "project_split": row.get("project_split"),
            "token_source": row.get("token_source"),
        }
        mismatches = {
            key: (value, example.get(key))
            for key, value in expected.items()
            if value and example.get(key) != value
        }
        if mismatches:
            mismatch_keys = ", ".join(sorted(mismatches))
            raise ValueError(f"model example manifest binding mismatch ({mismatch_keys}): {path}")
        examples.append(example)
    return examples
