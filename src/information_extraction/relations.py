"""Geometry-aware entity-pair candidate generation and rule inference."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.rotation_common import stable_id

COMPATIBLE_RELATIONS = {
    ("KEY", "VALUE"): "KEY_VALUE",
    ("QUESTION", "ANSWER"): "QUESTION_ANSWER",
    ("HEADER", "SECTION"): "HEADER_SECTION",
    ("TABLE_HEADER", "TABLE_CELL"): "TABLE_HEADER_CELL",
}


def relation_features(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, float]:
    sx0, sy0, sx1, sy1 = map(float, source["bbox"])
    tx0, ty0, tx1, ty1 = map(float, target["bbox"])
    sw, sh = max(1e-6, sx1 - sx0), max(1e-6, sy1 - sy0)
    tw, th = max(1e-6, tx1 - tx0), max(1e-6, ty1 - ty0)
    scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    horizontal_gap = max(0.0, tx0 - sx1, sx0 - tx1)
    vertical_gap = max(0.0, ty0 - sy1, sy0 - ty1)
    intersection_width = max(0.0, min(sx1, tx1) - max(sx0, tx0))
    intersection_height = max(0.0, min(sy1, ty1) - max(sy0, ty0))
    overlap = intersection_width * intersection_height
    union = sw * sh + tw * th - overlap
    same_line = max(sy0, ty0) <= min(sy1, ty1)
    same_column = max(sx0, tx0) <= min(sx1, tx1)
    return {
        "delta_x": tcx - scx,
        "delta_y": tcy - scy,
        "distance": math.hypot(tcx - scx, tcy - scy),
        "horizontal_gap": horizontal_gap,
        "vertical_gap": vertical_gap,
        "iou": overlap / max(1e-6, union),
        "same_line": float(same_line),
        "same_column": float(same_column),
        "target_is_right": float(tcx >= scx),
        "target_is_below": float(tcy >= scy),
    }


def generate_relation_candidates(
    entities: Sequence[Mapping[str, Any]], *, max_normalized_distance: float = 0.6
) -> list[dict[str, Any]]:
    """Generate only type-compatible, same-page, geometrically plausible pairs."""
    candidates: list[dict[str, Any]] = []
    if not entities:
        return candidates
    maximum_extent = max(
        max(float(entity["bbox"][2]), float(entity["bbox"][3])) for entity in entities
    )
    for source in entities:
        for target in entities:
            if source["id"] == target["id"]:
                continue
            relation_type = COMPATIBLE_RELATIONS.get((str(source["label"]), str(target["label"])))
            if relation_type is None:
                continue
            if source.get("page_number", 1) != target.get("page_number", 1):
                continue
            features = relation_features(source, target)
            normalized_distance = features["distance"] / max(1.0, maximum_extent)
            if normalized_distance > max_normalized_distance:
                continue
            candidates.append({
                "source_id": source["id"], "target_id": target["id"],
                "relation_type": relation_type, "features": features,
                "normalized_distance": normalized_distance,
            })
    return candidates


def infer_relations(entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rule baseline: keep the best geometric target per source and relation type."""
    candidates = generate_relation_candidates(entities)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate["source_id"], candidate["relation_type"]), []).append(candidate)
    relations = []
    entity_by_id = {str(entity["id"]): entity for entity in entities}
    for (source_id, relation_type), values in sorted(grouped.items()):
        best = min(values, key=_candidate_cost)
        features = best["features"]
        confidence = max(0.0, min(1.0, 1.0 - best["normalized_distance"]))
        target = entity_by_id[best["target_id"]]
        relations.append({
            "id": stable_id("rel", source_id, best["target_id"], relation_type),
            "type": relation_type,
            "source_id": source_id,
            "target_id": best["target_id"],
            "confidence": confidence,
            "page_number": int(target.get("page_number", 1)),
        })
    return relations


def _candidate_cost(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    features = candidate["features"]
    reading_order_penalty = 0.0
    if not features["target_is_right"] and not features["target_is_below"]:
        reading_order_penalty = 0.25
    line_bonus = -0.15 if features["same_line"] else 0.0
    return (
        float(candidate["normalized_distance"]) + reading_order_penalty + line_bonus,
        float(features["vertical_gap"]),
        float(features["horizontal_gap"]),
    )
