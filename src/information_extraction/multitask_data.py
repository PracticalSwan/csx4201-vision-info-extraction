"""Task labels and real entity-pair supervision for text-layout training."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.information_extraction.layoutxlm_data import LABEL_TO_ID, to_bio_labels
from src.information_extraction.relations import (
    COMPATIBLE_RELATIONS,
    generate_relation_candidates,
    relation_features,
)
from src.rotation_common import deterministic_rank

DOCUMENT_TYPE_LABELS = ("invoice", "receipt", "form", "other")
DOCUMENT_TYPE_TO_ID = {label: index for index, label in enumerate(DOCUMENT_TYPE_LABELS)}

CANONICAL_FIELD_LABELS = (
    "NONE",
    "organization_name",
    "document_title",
    "date",
    "invoice_number",
    "receipt_number",
    "reference_number",
    "subtotal",
    "tax",
    "total_amount",
    "currency",
    "payment_method",
    "address",
    "email",
    "phone_number",
)
CANONICAL_FIELD_TO_ID = {label: index for index, label in enumerate(CANONICAL_FIELD_LABELS)}

ENTITY_TYPE_LABELS = (
    "OTHER",
    "HEADER",
    "KEY",
    "VALUE",
    "QUESTION",
    "ANSWER",
    "TABLE_CELL",
    "SECTION",
    "TABLE_HEADER",
)
ENTITY_TYPE_TO_ID = {label: index for index, label in enumerate(ENTITY_TYPE_LABELS)}

RELATION_LABELS = (
    "NO_RELATION",
    "KEY_VALUE",
    "QUESTION_ANSWER",
    "HEADER_SECTION",
    "TABLE_HEADER_CELL",
    "OTHER_RELATION",
)
RELATION_LABEL_TO_ID = {label: index for index, label in enumerate(RELATION_LABELS)}

RELATION_FEATURE_ORDER = (
    "delta_x",
    "delta_y",
    "distance",
    "horizontal_gap",
    "vertical_gap",
    "iou",
    "same_line",
    "same_column",
    "target_is_right",
    "target_is_below",
)


def build_word_supervision(example: Mapping[str, Any]) -> dict[str, Any]:
    """Create entity, canonical-evidence, and document labels at word level."""
    tokens = list(example.get("tokens", []))
    labels = [str(value) for value in example.get("labels", [])]
    if len(tokens) != len(labels):
        raise ValueError("tokens and entity labels must have equal lengths")
    entity_ids = list(example.get("entity_ids", [None] * len(tokens)))
    if len(entity_ids) != len(tokens):
        raise ValueError("entity_ids and tokens must have equal lengths")
    loss_mask = [bool(value) for value in example.get("token_loss_mask", [True] * len(tokens))]
    if len(loss_mask) != len(tokens):
        raise ValueError("token_loss_mask and tokens must have equal lengths")

    canonical_ids = [0 if enabled else -100 for enabled in loss_mask]
    token_index_by_id = {
        str(token.get("id", "")): index for index, token in enumerate(tokens)
    }
    assigned_field_by_index: dict[int, str] = {}
    for field_name, field in sorted(example.get("canonical_fields", {}).items()):
        if field_name not in CANONICAL_FIELD_TO_ID or not isinstance(field, Mapping):
            continue
        if field.get("evidence_valid") is False:
            continue
        for token_id in field.get("token_ids", []):
            index = token_index_by_id.get(str(token_id))
            if index is None or not loss_mask[index]:
                continue
            previous = assigned_field_by_index.get(index)
            if previous is not None and previous != field_name:
                canonical_ids[index] = -100
                assigned_field_by_index[index] = "AMBIGUOUS"
            elif previous != "AMBIGUOUS":
                canonical_ids[index] = CANONICAL_FIELD_TO_ID[field_name]
                assigned_field_by_index[index] = field_name

    document_type = str(example.get("document_type", "other")).casefold()
    return {
        "entity_bio": to_bio_labels(labels, entity_ids=entity_ids),
        "entity_loss_mask": loss_mask,
        "canonical_label_ids": canonical_ids,
        "document_label_id": DOCUMENT_TYPE_TO_ID.get(
            document_type, DOCUMENT_TYPE_TO_ID["other"]
        ),
    }


def build_relation_supervision(
    example: Mapping[str, Any],
    *,
    word_ids: Sequence[int | None],
    max_negatives_per_positive: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return all real positives plus deterministic hard, type-compatible negatives."""
    if max_negatives_per_positive < 0:
        raise ValueError("max_negatives_per_positive must be non-negative")
    tokens = list(example.get("tokens", []))
    token_index_by_id = {
        str(token.get("id", "")): index for index, token in enumerate(tokens)
    }
    present_word_indices = {int(value) for value in word_ids if value is not None}
    entities: dict[str, dict[str, Any]] = {}
    for source_entity in example.get("entities", []):
        entity = dict(source_entity)
        token_ids = [str(value) for value in entity.get("token_ids", [])]
        word_indices = {
            token_index_by_id[token_id]
            for token_id in token_ids
            if token_id in token_index_by_id
        }
        if not word_indices and entity.get("ocr_token_indices"):
            word_indices = {int(value) for value in entity["ocr_token_indices"]}
        if not word_indices or not word_indices <= present_word_indices:
            continue
        entity["word_indices"] = word_indices
        entities[str(entity.get("id", ""))] = entity
    maximum_x = max((float(entity["bbox"][2]) for entity in entities.values()), default=1.0)
    maximum_y = max((float(entity["bbox"][3]) for entity in entities.values()), default=1.0)
    page_width = max(1.0, float(example.get("width", maximum_x)))
    page_height = max(1.0, float(example.get("height", maximum_y)))

    positives = []
    positive_keys: set[tuple[str, str]] = set()
    for relation in example.get("relations", []):
        source_id = str(relation.get("source_id", ""))
        target_id = str(relation.get("target_id", ""))
        if source_id not in entities or target_id not in entities:
            continue
        relation_type = str(relation.get("type", "OTHER_RELATION"))
        label_id = RELATION_LABEL_TO_ID.get(
            relation_type, RELATION_LABEL_TO_ID["OTHER_RELATION"]
        )
        positives.append(_relation_pair(
            entities[source_id],
            entities[target_id],
            word_ids,
            label_id=label_id,
            relation_id=str(relation.get("id", "")) or None,
            page_width=page_width,
            page_height=page_height,
        ))
        positive_keys.add((source_id, target_id))

    negative_candidates = []
    entity_values = list(entities.values())
    for source in entity_values:
        for target in entity_values:
            source_id = str(source.get("id", ""))
            target_id = str(target.get("id", ""))
            if source_id == target_id or (source_id, target_id) in positive_keys:
                continue
            compatible = (
                str(source.get("label", "")), str(target.get("label", ""))
            ) in COMPATIBLE_RELATIONS
            features = relation_features(source, target)
            negative_candidates.append((
                0 if compatible else 1,
                float(features["distance"]),
                deterministic_rank(
                    f"{example.get('example_id', '')}|{source_id}|{target_id}", seed
                ),
                source,
                target,
            ))
    positive_count = len(positives)
    negative_limit = max_negatives_per_positive * max(1, positive_count)
    negatives = [
        _relation_pair(
            source,
            target,
            word_ids,
            label_id=RELATION_LABEL_TO_ID["NO_RELATION"],
            relation_id=None,
            page_width=page_width,
            page_height=page_height,
        )
        for _, _, _, source, target in sorted(negative_candidates)[:negative_limit]
    ]
    return positives + negatives


def build_inference_relation_pairs(
    entities: Sequence[Mapping[str, Any]],
    *,
    word_ids: Sequence[int | None],
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    """Build type-compatible relation-head inputs for one tokenizer window."""
    present_word_indices = {int(value) for value in word_ids if value is not None}
    eligible = []
    for source in entities:
        entity = dict(source)
        indices = {int(value) for value in entity.get("word_indices", [])}
        if not indices or not indices <= present_word_indices:
            continue
        entity["word_indices"] = indices
        eligible.append(entity)
    pairs = []
    for candidate in generate_relation_candidates(eligible):
        by_id = {str(entity.get("id", "")): entity for entity in eligible}
        source = by_id[candidate["source_id"]]
        target = by_id[candidate["target_id"]]
        pair = _relation_pair(
            source,
            target,
            word_ids,
            label_id=RELATION_LABEL_TO_ID["NO_RELATION"],
            relation_id=None,
            page_width=max(1.0, float(page_width)),
            page_height=max(1.0, float(page_height)),
        )
        pair["candidate_relation_type"] = candidate["relation_type"]
        pairs.append(pair)
    return pairs


def encode_multitask_windows(
    tokenizer: Any,
    example: Mapping[str, Any],
    *,
    boxes: Sequence[Sequence[int]],
    max_length: int = 512,
    stride: int = 64,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Tokenize one example and carry every task's masks into each window."""
    words = [str(token.get("text", "")) for token in example.get("tokens", [])]
    if len(words) != len(boxes):
        raise ValueError("words and boxes must have equal lengths")
    supervision = build_word_supervision(example)
    encoding = tokenizer(
        words,
        boxes=[list(box) for box in boxes],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors=None,
    )
    input_ids = encoding.get("input_ids", [])
    if not input_ids:
        raise ValueError("LayoutXLM tokenizer returned no windows")
    nested = isinstance(input_ids[0], list)
    window_count = len(input_ids) if nested else 1
    result = []
    for window_index in range(window_count):
        word_ids = list(encoding.word_ids(batch_index=window_index))
        entity_labels: list[int] = []
        canonical_labels: list[int] = []
        seen_word_ids: set[int] = set()
        for word_id in word_ids:
            if word_id is None or int(word_id) in seen_word_ids:
                entity_labels.append(-100)
                canonical_labels.append(-100)
                continue
            index = int(word_id)
            seen_word_ids.add(index)
            if not supervision["entity_loss_mask"][index]:
                entity_labels.append(-100)
            else:
                entity_labels.append(LABEL_TO_ID[supervision["entity_bio"][index]])
            canonical_labels.append(int(supervision["canonical_label_ids"][index]))
        window = {
            "example_id": str(example.get("example_id", "")),
            "window_index": window_index,
            "word_ids": word_ids,
            "entity_labels": entity_labels,
            "canonical_labels": canonical_labels,
            "document_label": int(supervision["document_label_id"]),
            "relation_pairs": build_relation_supervision(
                example,
                word_ids=word_ids,
                seed=seed,
            ),
        }
        for key in ("input_ids", "bbox", "attention_mask", "token_type_ids"):
            if key not in encoding:
                continue
            values = encoding[key]
            window[key] = values[window_index] if nested else values
        result.append(window)
    return result


def _relation_pair(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    word_ids: Sequence[int | None],
    *,
    label_id: int,
    relation_id: str | None,
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    features = relation_features(source, target)
    normalized_features = dict(features)
    for name in ("delta_x", "horizontal_gap"):
        normalized_features[name] = float(features[name]) / page_width
    for name in ("delta_y", "vertical_gap"):
        normalized_features[name] = float(features[name]) / page_height
    normalized_features["distance"] = float(features["distance"]) / (
        page_width**2 + page_height**2
    ) ** 0.5
    source_indices = set(source["word_indices"])
    target_indices = set(target["word_indices"])
    return {
        "source_id": str(source.get("id", "")),
        "target_id": str(target.get("id", "")),
        "relation_id": relation_id,
        "label_id": label_id,
        "source_type_id": ENTITY_TYPE_TO_ID.get(
            str(source.get("label", "OTHER")), ENTITY_TYPE_TO_ID["OTHER"]
        ),
        "target_type_id": ENTITY_TYPE_TO_ID.get(
            str(target.get("label", "OTHER")), ENTITY_TYPE_TO_ID["OTHER"]
        ),
        "source_mask": [
            1.0 if word_id is not None and int(word_id) in source_indices else 0.0
            for word_id in word_ids
        ],
        "target_mask": [
            1.0 if word_id is not None and int(word_id) in target_indices else 0.0
            for word_id in word_ids
        ],
        "geometry": [float(normalized_features[name]) for name in RELATION_FEATURE_ORDER],
    }
