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
)

BASE_ENTITY_LABELS = ("HEADER", "KEY", "VALUE", "QUESTION", "ANSWER", "TABLE_CELL", "OTHER")
BIO_LABELS = ("O",) + tuple(
    prefix + label
    for label in BASE_ENTITY_LABELS if label != "OTHER"
    for prefix in ("B-", "I-")
)
LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def to_bio_labels(labels: Sequence[str]) -> list[str]:
    result: list[str] = []
    previous = "OTHER"
    for label in labels:
        normalized = label if label in BASE_ENTITY_LABELS else "OTHER"
        if normalized == "OTHER":
            result.append("O")
        else:
            prefix = "I-" if normalized == previous else "B-"
            result.append(prefix + normalized)
        previous = normalized
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


def load_model_examples(manifest_path: str | Path, split: str) -> list[dict[str, Any]]:
    import csv

    rows = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("project_split") == split and row.get("is_usable") == "true":
                rows.append(row)
    examples = []
    for row in rows:
        path = Path(row["model_example_path"])
        example = json.loads(path.read_text(encoding="utf-8"))
        if example.get("is_private") is not False:
            raise ValueError(f"private or unmarked model example refused: {path}")
        examples.append(example)
    return examples
