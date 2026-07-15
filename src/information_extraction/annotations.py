"""Public-dataset adapters for the normalized information-extraction schema."""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from src.information_extraction.geometry import bbox_to_polygon, polygon_to_bbox
from src.rotation_common import stable_id

CONVERTER_VERSION = "1.1"
ENTITY_LABELS = {"HEADER", "KEY", "VALUE", "QUESTION", "ANSWER", "TABLE_CELL", "OTHER"}


class AnnotationConversionError(ValueError):
    """Raised for a malformed or unsupported public source annotation."""


def load_annotation_schema(project_root: Path) -> dict[str, Any]:
    path = project_root / "data" / "metadata" / "annotation_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _annotation_validator(project_root: Path) -> Draft202012Validator:
    """Compile one validator per project root instead of once per page."""
    return Draft202012Validator(load_annotation_schema(project_root))


def validate_annotation(record: Mapping[str, Any], project_root: Path) -> None:
    validator = _annotation_validator(project_root.resolve())
    errors = sorted(validator.iter_errors(dict(record)), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
            for error in errors[:8]
        )
        raise AnnotationConversionError(detail)
    _validate_geometry(record)
    _validate_relations(record)


def normalize_public_page(
    row: Mapping[str, str], project_root: Path, project_split: str
) -> dict[str, Any]:
    dataset = row["dataset"].lower()
    image_path = project_root / row["prepared_image_path"]
    if not image_path.is_file():
        raise AnnotationConversionError(f"image does not exist: {image_path}")
    with Image.open(image_path) as image:
        width, height = image.size
    base = {
        "schema_version": "1.0",
        "document_id": row["document_id"],
        "page_id": row["page_id"],
        "dataset": dataset,
        "dataset_component": row.get("dataset_component", ""),
        "document_type": row.get("document_type") or "unknown",
        "language": row.get("language") or "unknown",
        "image_path": Path(row["prepared_image_path"]).as_posix(),
        "page": {
            "number": int(row.get("source_page_number") or 1),
            "width": int(width),
            "height": int(height),
        },
        "tokens": [],
        "entities": [],
        "relations": [],
        "canonical_fields": {},
        "source_qa": [],
        "annotation_provenance": {
            "source_paths": [],
            "source_format": "unknown",
            "converter_version": CONVERTER_VERSION,
        },
        "alignment_status": "source_tokens",
        "project_split": project_split,
        "is_private": False,
    }
    if dataset == "sroie":
        _normalize_sroie(base, image_path)
    elif dataset == "funsd":
        _normalize_funsd(base, image_path)
    elif dataset == "fatura":
        _normalize_fatura(base, image_path)
    elif dataset == "coru":
        _normalize_coru(base, image_path)
    else:
        raise AnnotationConversionError(f"unsupported dataset: {dataset}")
    validate_annotation(base, project_root)
    return base


def _normalize_sroie(record: dict[str, Any], image_path: Path) -> None:
    split_root = image_path.parent.parent
    box_path = split_root / "box" / f"{image_path.stem}.txt"
    entity_path = split_root / "entities" / f"{image_path.stem}.txt"
    if not box_path.is_file() or not entity_path.is_file():
        raise AnnotationConversionError(f"SROIE annotation pair missing for {image_path.name}")
    entities = json.loads(entity_path.read_text(encoding="utf-8-sig"))
    canonical = {
        "company": "organization_name",
        "address": "address",
        "date": "date",
        "total": "total_amount",
    }
    token_rows: list[tuple[list[list[float]], str]] = []
    skipped_source_rows: list[dict[str, Any]] = []
    box_text, box_encoding = _read_source_text(box_path)
    for line_number, values in enumerate(csv.reader(io.StringIO(box_text)), start=1):
        if len(values) < 9:
            skipped_source_rows.append({"line": line_number, "reason": "fewer_than_9_values"})
            continue
        try:
            polygon = [[float(values[i]), float(values[i + 1])] for i in range(0, 8, 2)]
        except ValueError:
            skipped_source_rows.append({"line": line_number, "reason": "invalid_coordinate"})
            continue
        text = ",".join(values[8:]).strip()
        bbox = polygon_to_bbox(polygon)
        if not text:
            skipped_source_rows.append({"line": line_number, "reason": "empty_text"})
            continue
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            skipped_source_rows.append({"line": line_number, "reason": "degenerate_geometry"})
            continue
        token_rows.append((polygon, text))
    if not token_rows:
        raise AnnotationConversionError("SROIE annotation has no usable token rows")
    normalized_fields = {key: _normalized_text(str(value)) for key, value in entities.items()}
    for index, (polygon, text) in enumerate(token_rows):
        matched = [
            key for key, value in normalized_fields.items()
            if value and (_normalized_text(text) in value or value in _normalized_text(text))
        ]
        label = "VALUE" if matched else "OTHER"
        record["tokens"].append(_token(record["page_id"], index, text, polygon, label, index))
    for source_key, field_name in canonical.items():
        value = entities.get(source_key)
        if value in (None, ""):
            continue
        token_ids = [
            token["id"] for token in record["tokens"]
            if _normalized_text(token["text"]) in _normalized_text(str(value))
        ]
        record["canonical_fields"][field_name] = {
            "value": str(value), "raw_text": str(value), "token_ids": token_ids, "source_key": source_key
        }
        matched_tokens = [token for token in record["tokens"] if token["id"] in token_ids]
        if matched_tokens:
            record["entities"].append(
                _entity_from_tokens(record["page_id"], f"sroie-{source_key}", "VALUE", str(value), matched_tokens)
            )
    record["annotation_provenance"] = {
        "source_paths": [_relative_public(box_path), _relative_public(entity_path)],
        "source_format": "sroie_polygon_csv_plus_entity_json",
        "converter_version": CONVERTER_VERSION,
        "box_text_encoding": box_encoding,
        "skipped_source_rows": skipped_source_rows,
    }


def _normalize_funsd(record: dict[str, Any], image_path: Path) -> None:
    annotation_path = image_path.parent.parent / "annotations" / f"{image_path.stem}.json"
    if not annotation_path.is_file():
        raise AnnotationConversionError(f"FUNSD annotation missing: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    forms = payload.get("form")
    if not isinstance(forms, list):
        raise AnnotationConversionError("FUNSD form must be a list")
    source_to_entity: dict[int, str] = {}
    source_labels: dict[int, str] = {}
    links: set[tuple[int, int]] = set()
    for form_index, form in enumerate(forms):
        if not isinstance(form, Mapping):
            raise AnnotationConversionError("FUNSD form item must be an object")
        source_id = int(form.get("id", form_index))
        source_label = str(form.get("label", "other")).lower()
        label = {"header": "HEADER", "question": "QUESTION", "answer": "ANSWER"}.get(source_label, "OTHER")
        source_labels[source_id] = label
        form_tokens: list[dict[str, Any]] = []
        words = form.get("words") or []
        for word_index, word in enumerate(words):
            text = str(word.get("text", "")).strip()
            box = word.get("box")
            if not text or not _valid_bbox(box):
                continue
            token = _token(
                record["page_id"], len(record["tokens"]), text, bbox_to_polygon(box), label,
                f"{source_id}:{word_index}",
            )
            record["tokens"].append(token)
            form_tokens.append(token)
        if not form_tokens:
            text = str(form.get("text", "")).strip()
            box = form.get("box")
            if text and _valid_bbox(box):
                token = _token(
                    record["page_id"], len(record["tokens"]), text, bbox_to_polygon(box), label, source_id
                )
                record["tokens"].append(token)
                form_tokens.append(token)
        if form_tokens:
            entity = _entity_from_tokens(
                record["page_id"], f"funsd-{source_id}", label,
                str(form.get("text") or " ".join(token["text"] for token in form_tokens)), form_tokens,
            )
            record["entities"].append(entity)
            source_to_entity[source_id] = entity["id"]
        for link in form.get("linking") or []:
            if isinstance(link, Sequence) and len(link) == 2:
                links.add((int(link[0]), int(link[1])))
    for index, (left, right) in enumerate(sorted(links)):
        if left not in source_to_entity or right not in source_to_entity:
            continue
        pair = {source_labels.get(left), source_labels.get(right)}
        relation_type = "QUESTION_ANSWER" if pair == {"QUESTION", "ANSWER"} else "GENERIC"
        record["relations"].append({
            "id": stable_id("rel", record["page_id"], left, right),
            "type": relation_type,
            "source_id": source_to_entity[left],
            "target_id": source_to_entity[right],
        })
    record["annotation_provenance"] = {
        "source_paths": [_relative_public(annotation_path)],
        "source_format": "funsd_form_json",
        "converter_version": CONVERTER_VERSION,
    }


def _normalize_fatura(record: dict[str, Any], image_path: Path) -> None:
    root = image_path.parent.parent
    hf_matches = sorted((root / "Annotations" / "layoutlm_HF_format").glob(f"{image_path.stem}_hugg_*.json"))
    original_path = root / "Annotations" / "Original_Format" / f"{image_path.stem}.json"
    if len(hf_matches) != 1 or not original_path.is_file():
        raise AnnotationConversionError(f"FATURA annotations missing or ambiguous for {image_path.name}")
    hf_path = hf_matches[0]
    payload = json.loads(hf_path.read_text(encoding="utf-8-sig"))
    words, boxes, tags = payload.get("words"), payload.get("bboxes"), payload.get("ner_tags")
    if not isinstance(words, list) or not (len(words) == len(boxes or []) == len(tags or [])):
        raise AnnotationConversionError("FATURA words, bboxes, and ner_tags lengths differ")
    clipped_source_boxes = 0
    width, height = int(record["page"]["width"]), int(record["page"]["height"])
    for index, (text, box, tag) in enumerate(zip(words, boxes, tags, strict=True)):
        text = str(text).strip()
        if not text or not _valid_bbox(box):
            continue
        clipped_box = _clip_bbox_to_page(box, width, height)
        if clipped_box is None:
            continue
        if list(map(float, box)) != clipped_box:
            clipped_source_boxes += 1
        label = "TABLE_CELL" if int(tag) == 10 else ("OTHER" if int(tag) == 13 else "VALUE")
        record["tokens"].append(
            _token(record["page_id"], index, text, bbox_to_polygon(clipped_box), label, int(tag))
        )
    for group_index, group in enumerate(_contiguous_token_groups(record["tokens"])):
        if group[0]["entity_label"] == "OTHER":
            continue
        record["entities"].append(
            _entity_from_tokens(
                record["page_id"], f"fatura-{group_index}", group[0]["entity_label"],
                " ".join(token["text"] for token in group), group,
            )
        )
    original = json.loads(original_path.read_text(encoding="utf-8-sig"))
    field_map = {
        "SELLER_NAME": "organization_name", "DATE": "date", "NUMBER": "invoice_number",
        "SUB_TOTAL": "subtotal", "TAX": "tax", "TOTAL": "total_amount",
        "SELLER_ADDRESS": "address", "SELLER_EMAIL": "email", "TITLE": "document_title",
    }
    for source_key, field_name in field_map.items():
        item = original.get(source_key)
        if isinstance(item, Mapping) and item.get("text"):
            record["canonical_fields"][field_name] = {
                "value": str(item["text"]), "raw_text": str(item["text"]),
                "token_ids": _matching_token_ids(record["tokens"], str(item["text"])),
                "source_key": source_key,
            }
    record["annotation_provenance"] = {
        "source_paths": [_relative_public(hf_path), _relative_public(original_path)],
        "source_format": "fatura_layoutlm_hf_plus_original_json",
        "converter_version": CONVERTER_VERSION,
        "clipped_source_boxes": clipped_source_boxes,
    }


def _normalize_coru(record: dict[str, Any], image_path: Path) -> None:
    component = record["dataset_component"]
    if component == "Receipt Question Answering":
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.is_file():
            raise AnnotationConversionError(f"CORU QA annotation missing: {annotation_path}")
        payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise AnnotationConversionError("CORU QA annotation must be a list")
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                record["source_qa"].append({"question": question, "answer": answer})
                canonical = _coru_question_field(question)
                if canonical and canonical not in record["canonical_fields"]:
                    record["canonical_fields"][canonical] = {
                        "value": answer, "raw_text": answer, "token_ids": [], "source_key": question
                    }
        record["alignment_status"] = "requires_ocr_alignment"
        record["annotation_provenance"] = {
            "source_paths": [_relative_public(annotation_path)],
            "source_format": "coru_question_answer_json_without_geometry",
            "converter_version": CONVERTER_VERSION,
        }
        return
    if component == "Receipt Images & Key Information Detection":
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise AnnotationConversionError(f"CORU KIE label missing: {label_path}")
        record["alignment_status"] = "unsupported"
        record["annotation_provenance"] = {
            "source_paths": [_relative_public(label_path)],
            "source_format": "coru_yolo_regions_without_text_or_class_map",
            "converter_version": CONVERTER_VERSION,
        }
        return
    raise AnnotationConversionError(f"unsupported CORU component: {component}")


def _token(
    page_id: str,
    index: int,
    text: str,
    polygon: Sequence[Sequence[float]],
    label: str,
    source_id: Any,
) -> dict[str, Any]:
    if label not in ENTITY_LABELS:
        raise AnnotationConversionError(f"invalid entity label: {label}")
    clean_polygon = [[float(x), float(y)] for x, y in polygon]
    return {
        "id": stable_id("tok", page_id, index),
        "text": unicodedata.normalize("NFC", text.strip()),
        "polygon": clean_polygon,
        "bbox": polygon_to_bbox(clean_polygon),
        "entity_label": label,
        "source_id": source_id,
    }


def _entity_from_tokens(
    page_id: str, source_id: str, label: str, text: str, tokens: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    boxes = [token["bbox"] for token in tokens]
    bbox = [
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    ]
    return {
        "id": stable_id("ent", page_id, source_id),
        "label": label,
        "text": unicodedata.normalize("NFC", text.strip()),
        "token_ids": [str(token["id"]) for token in tokens],
        "polygon": bbox_to_polygon(bbox),
        "bbox": bbox,
    }


def _contiguous_token_groups(tokens: Sequence[Mapping[str, Any]]) -> Iterable[list[Mapping[str, Any]]]:
    group: list[Mapping[str, Any]] = []
    previous: Any = object()
    for token in tokens:
        current = token.get("source_id")
        if group and current != previous:
            yield group
            group = []
        group.append(token)
        previous = current
    if group:
        yield group


def _matching_token_ids(tokens: Sequence[Mapping[str, Any]], value: str) -> list[str]:
    normalized = _normalized_text(value)
    return [str(token["id"]) for token in tokens if _normalized_text(str(token["text"])) in normalized]


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return False
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return False
    return x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0


def _clip_bbox_to_page(value: Sequence[Any], width: int, height: int) -> list[float] | None:
    x0, y0, x1, y1 = map(float, value)
    clipped = [
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    ]
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


def _read_source_text(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return payload.decode("windows-1252"), "windows-1252"


def _validate_geometry(record: Mapping[str, Any]) -> None:
    width, height = record["page"]["width"], record["page"]["height"]
    ids: set[str] = set()
    for collection in ("tokens", "entities"):
        for item in record[collection]:
            if item["id"] in ids:
                raise AnnotationConversionError(f"duplicate annotation id: {item['id']}")
            ids.add(item["id"])
            x0, y0, x1, y1 = item["bbox"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise AnnotationConversionError(
                    f"{collection} geometry outside image bounds: {item['id']} {item['bbox']} vs {width}x{height}"
                )


def _validate_relations(record: Mapping[str, Any]) -> None:
    entity_ids = {entity["id"] for entity in record["entities"]}
    for relation in record["relations"]:
        if relation["source_id"] not in entity_ids or relation["target_id"] not in entity_ids:
            raise AnnotationConversionError(f"relation references missing entity: {relation['id']}")


def _normalized_text(value: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", value).casefold(), flags=re.UNICODE)


def _relative_public(path: Path) -> str:
    parts = list(path.parts)
    try:
        index = [part.lower() for part in parts].index("public")
    except ValueError:
        return path.name
    return Path(*parts[index - 2:]).as_posix()


def _coru_question_field(question: str) -> str | None:
    normalized = question.casefold()
    patterns = (
        (("store name", "merchant", "company"), "organization_name"),
        (("invoice number",), "invoice_number"),
        (("receipt number", "transaction number"), "receipt_number"),
        (("reference number", "account number"), "reference_number"),
        (("due date", " date", "date?"), "date"),
        (("subtotal",), "subtotal"),
        (("tax", "vat"), "tax"),
        (("total", "amount due"), "total_amount"),
        (("address",), "address"),
        (("email",), "email"),
        (("phone", "telephone"), "phone_number"),
    )
    for keywords, field in patterns:
        if any(keyword in normalized for keyword in keywords):
            return field
    return None
