"""Evidence-only generic key/value entities for unseen document types."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from src.information_extraction.geometry import bbox_to_polygon
from src.rotation_common import stable_id


def generic_key_value_entities(
    ocr_result: Mapping[str, Any], *, page_number: int
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for line_index, line in enumerate(ocr_result.get("lines") or []):
        text = str(line.get("text", "")).strip()
        parts = re.split(r"\s*[:：]\s*", text, maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        key_text, value_text = parts
        x0, y0, x1, y1 = map(float, line["bbox"])
        fraction = max(0.2, min(0.8, len(key_text) / max(1, len(text))))
        split_x = x0 + (x1 - x0) * fraction
        confidence = line.get("confidence")
        confidence = 0.5 if confidence is None else float(confidence)
        word_ids = [str(value) for value in line.get("word_ids", [])]
        for role, value, bbox in (
            ("KEY", key_text, [x0, y0, split_x, y1]),
            ("VALUE", value_text, [split_x, y0, x1, y1]),
        ):
            entities.append({
                "id": stable_id("generic_entity", page_number, line_index, role, value),
                "label": role,
                "text": value,
                "word_ids": word_ids,
                "polygon": bbox_to_polygon(bbox),
                "bbox": bbox,
                "confidence": max(0.0, min(1.0, confidence * 0.8)),
                "page_number": int(page_number),
            })
    return entities
