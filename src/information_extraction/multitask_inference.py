"""Calibrated inference and deterministic post-processing for the multi-task pre-model."""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.information_extraction.entity_inference import entities_from_word_predictions
from src.information_extraction.geometry import bbox_to_polygon
from src.information_extraction.layoutxlm_data import normalize_bbox
from src.information_extraction.multitask_data import (
    CANONICAL_FIELD_LABELS,
    DOCUMENT_TYPE_LABELS,
    RELATION_LABELS,
    build_inference_relation_pairs,
)
from src.rotation_common import sha256_file, stable_id

DEFAULT_THRESHOLDS = {
    "entity": 0.50,
    "canonical": 0.50,
    "document": 0.50,
    "relation": 0.50,
}
DEFAULT_TEMPERATURES = {
    "entity": 1.0,
    "canonical": 1.0,
    "document": 1.0,
    "relation": 1.0,
}


def aggregate_word_probabilities(
    window_probabilities: Sequence[Sequence[Sequence[float]]],
    window_word_ids: Sequence[Sequence[int | None]],
    *,
    word_count: int,
) -> list[list[float]]:
    """Average the first subtoken score for each word across overlapping windows."""
    if len(window_probabilities) != len(window_word_ids):
        raise ValueError("window probabilities and word IDs must have equal lengths")
    class_count = 0
    for window in window_probabilities:
        if window:
            class_count = len(window[0])
            break
    if class_count < 1:
        return [[] for _ in range(word_count)]
    sums = [[0.0] * class_count for _ in range(word_count)]
    counts = [0] * word_count
    for probabilities, word_ids in zip(
        window_probabilities, window_word_ids, strict=True
    ):
        if len(probabilities) != len(word_ids):
            raise ValueError("token probabilities and word IDs must have equal lengths")
        seen: set[int] = set()
        for vector, word_id in zip(probabilities, word_ids, strict=True):
            if word_id is None:
                continue
            index = int(word_id)
            if index in seen:
                continue
            seen.add(index)
            if index < 0 or index >= word_count or len(vector) != class_count:
                raise ValueError("invalid word probability alignment")
            sums[index] = [left + float(right) for left, right in zip(sums[index], vector)]
            counts[index] += 1
    return [
        [value / counts[index] for value in vector]
        if counts[index]
        else [1.0] + [0.0] * (class_count - 1)
        for index, vector in enumerate(sums)
    ]


def decode_canonical_fields(
    words: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]],
    *,
    page_number: int,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Decode canonical evidence spans and abstain below calibrated thresholds."""
    if len(words) != len(probabilities):
        raise ValueError("canonical probabilities must align with OCR words")
    threshold_values = dict(thresholds or {})
    tagged: list[tuple[str | None, float]] = []
    for vector in probabilities:
        if len(vector) != len(CANONICAL_FIELD_LABELS):
            raise ValueError("canonical probability class count mismatch")
        label_id = max(range(len(vector)), key=lambda index: float(vector[index]))
        field = CANONICAL_FIELD_LABELS[label_id]
        confidence = float(vector[label_id])
        threshold = float(
            threshold_values.get(field, threshold_values.get("default", DEFAULT_THRESHOLDS["canonical"]))
        )
        tagged.append((field if label_id and confidence >= threshold else None, confidence))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_field: str | None = None
    current_indices: list[int] = []
    for index, (field, _) in enumerate(tagged):
        if field != current_field:
            if current_field and current_indices:
                groups[current_field].append(_field_candidate(words, tagged, current_indices, page_number))
            current_field, current_indices = field, []
        if field:
            current_indices.append(index)
    if current_field and current_indices:
        groups[current_field].append(_field_candidate(words, tagged, current_indices, page_number))
    return {
        field: max(values, key=lambda value: (float(value["confidence"]), len(value["raw_text"])))
        for field, values in groups.items()
    }


def decode_document_type(
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> tuple[str, float]:
    if len(probabilities) != len(DOCUMENT_TYPE_LABELS):
        raise ValueError("document probability class count mismatch")
    label_id = max(range(len(probabilities)), key=lambda index: float(probabilities[index]))
    confidence = float(probabilities[label_id])
    label = DOCUMENT_TYPE_LABELS[label_id]
    if confidence < threshold or label == "other":
        return "unknown", confidence
    return label, confidence


def decode_relations(
    pairs: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]],
    *,
    page_number: int,
    threshold: float,
) -> list[dict[str, Any]]:
    if len(pairs) != len(probabilities):
        raise ValueError("relation probabilities must align with candidate pairs")
    result = []
    for pair, vector in zip(pairs, probabilities, strict=True):
        if len(vector) != len(RELATION_LABELS):
            raise ValueError("relation probability class count mismatch")
        label_id = max(range(len(vector)), key=lambda index: float(vector[index]))
        confidence = float(vector[label_id])
        label = RELATION_LABELS[label_id]
        expected = str(pair.get("candidate_relation_type", ""))
        if label == "NO_RELATION" or confidence < threshold:
            continue
        if label != expected and label != "OTHER_RELATION":
            continue
        relation_type = "GENERIC" if label == "OTHER_RELATION" else label
        source_id = str(pair["source_id"])
        target_id = str(pair["target_id"])
        result.append({
            "id": stable_id("model_relation", source_id, target_id, relation_type),
            "type": relation_type,
            "source_id": source_id,
            "target_id": target_id,
            "confidence": max(0.0, min(1.0, confidence)),
            "page_number": int(page_number),
        })
    return result


def reconstruct_tables(
    entities: Sequence[Mapping[str, Any]],
    *,
    page_number: int,
    source: str = "model_entities",
) -> list[dict[str, Any]]:
    """Reconstruct a deterministic grid from model-predicted TABLE_CELL entities."""
    cells = [dict(entity) for entity in entities if entity.get("label") == "TABLE_CELL"]
    if len(cells) < 2:
        return []
    heights = [max(1.0, float(cell["bbox"][3]) - float(cell["bbox"][1])) for cell in cells]
    tolerance = max(2.0, statistics.median(heights) * 0.7)
    rows: list[list[dict[str, Any]]] = []
    for cell in sorted(cells, key=lambda value: (
        (float(value["bbox"][1]) + float(value["bbox"][3])) / 2.0,
        float(value["bbox"][0]),
    )):
        center = (float(cell["bbox"][1]) + float(cell["bbox"][3])) / 2.0
        target = next((row for row in rows if abs(center - statistics.fmean(
            (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0 for item in row
        )) <= tolerance), None)
        (target if target is not None else rows.append([]) or rows[-1]).append(cell)
    output_cells = []
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(sorted(row, key=lambda value: float(value["bbox"][0]))):
            output_cells.append({
                "row_index": row_index,
                "column_index": column_index,
                "text": str(cell.get("text", "")),
                "bbox": [float(value) for value in cell["bbox"]],
                "polygon": [
                    [float(x), float(y)]
                    for x, y in (cell.get("polygon") or bbox_to_polygon(cell["bbox"]))
                ],
                "entity_id": str(cell.get("id", "")),
                "confidence": float(cell.get("confidence", 0.0)),
                "semantic_type": "unknown",
            })
    header_row_index, column_semantics = _table_header_semantics(output_cells)
    for cell in output_cells:
        cell["semantic_type"] = column_semantics.get(cell["column_index"], "unknown")
    structured_rows = []
    for row_index in range(len(rows)):
        row_cells = [cell for cell in output_cells if cell["row_index"] == row_index]
        structured_rows.append({
            "row_index": row_index,
            "row_type": (
                "header" if row_index == header_row_index
                else "item" if header_row_index is not None and row_index > header_row_index
                else "unknown"
            ),
            "cells": row_cells,
        })
    bbox = _union_bbox([cell["bbox"] for cell in cells])
    warnings = [] if header_row_index is not None else [
        "table header semantics were not confidently identified; cells remain geometry-only"
    ]
    return [{
        "id": stable_id("table", page_number, *[cell["id"] for cell in cells]),
        "page_number": int(page_number),
        "method": "geometry:table_cell_grid",
        "source": source,
        "confidence": statistics.fmean(float(cell.get("confidence", 0.0)) for cell in cells),
        "row_count": len(rows),
        "column_count": max(len(row) for row in rows),
        "bbox": bbox,
        "cells": output_cells,
        "header_row_index": header_row_index,
        "rows": structured_rows,
        "source_polygons": [cell["polygon"] for cell in output_cells],
        "warnings": warnings,
        "raw_ocr_fallback": "\n".join(
            " | ".join(cell["text"] for cell in row["cells"])
            for row in structured_rows
        ),
    }]


def reconstruct_ocr_tables(
    words: Sequence[Mapping[str, Any]], *, page_number: int
) -> list[dict[str, Any]]:
    """Conservative repeated-row fallback for layouts without TABLE_CELL predictions."""
    entities = []
    for index, word in enumerate(words):
        bbox = word.get("bbox")
        text = str(word.get("text", "")).strip()
        if not text or not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        entities.append({
            "id": str(word.get("id") or stable_id("ocr_table_cell", page_number, index, text)),
            "label": "TABLE_CELL",
            "text": text,
            "bbox": list(bbox),
            "polygon": word.get("polygon") or bbox_to_polygon(bbox),
            "confidence": float(word.get("confidence", 0.0) or 0.0),
        })
    tables = reconstruct_tables(entities, page_number=page_number, source="ocr_geometry")
    if not tables:
        return []
    table = tables[0]
    row_sizes = Counter(cell["row_index"] for cell in table["cells"])
    repeated_rows = [row for row, count in row_sizes.items() if count >= 2]
    if len(repeated_rows) < 3 or table["column_count"] < 2:
        return []
    aligned_columns = 0
    for column in range(table["column_count"]):
        column_cells = [
            cell for cell in table["cells"]
            if cell["row_index"] in repeated_rows and cell["column_index"] == column
        ]
        if len(column_cells) < 3:
            continue
        centers = [(cell["bbox"][0] + cell["bbox"][2]) / 2.0 for cell in column_cells]
        widths = [max(1.0, cell["bbox"][2] - cell["bbox"][0]) for cell in column_cells]
        if max(abs(center - statistics.median(centers)) for center in centers) <= max(
            20.0, statistics.median(widths) * 1.5
        ):
            aligned_columns += 1
    if aligned_columns < 2:
        return []
    table["confidence"] = min(float(table["confidence"]), 0.75)
    table["warnings"].append("table reconstructed from repeated OCR geometry without a model TABLE_CELL label")
    return tables


def _table_header_semantics(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[int | None, dict[int, str]]:
    keywords = {
        "description": ("description", "item", "product", "details", "açıklama", "ürün", "รายการ", "สินค้า"),
        "quantity": ("qty", "quantity", "adet", "miktar", "จำนวน"),
        "unit_price": ("unit price", "price", "birim fiyat", "fiyat", "ราคาต่อหน่วย", "ราคา"),
        "line_total": ("line total", "amount", "total", "tutar", "toplam", "จำนวนเงิน", "รวม"),
    }
    by_row: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_row[int(cell["row_index"])].append(cell)
    for row_index in sorted(by_row)[:2]:
        semantics: dict[int, str] = {}
        for cell in by_row[row_index]:
            folded = str(cell.get("text", "")).casefold()
            for semantic, values in keywords.items():
                if any(value.casefold() in folded for value in values):
                    semantics[int(cell["column_index"])] = semantic
                    break
        if len(semantics) >= 2:
            return row_index, semantics
    return None, {}


class MultiTaskLayoutExtractor:
    """Run all four trained heads and calibrated abstention in the layout process."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        max_length: int = 512,
        stride: int = 64,
        calibration_path: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"multi-task checkpoint is missing: {checkpoint}")
        import torch
        from transformers import LayoutXLMTokenizerFast

        from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA layout inference requested but PyTorch CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = LayoutXLMTokenizerFast.from_pretrained(
            checkpoint, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.model = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(self.device).eval()
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.id_to_entity_label = {
            int(key): str(value) for key, value in self.model.config.id2label.items()
        }
        self.calibration, self.calibration_warnings = _load_calibration(
            calibration_path, checkpoint
        )
        if confidence_threshold is not None:
            self.calibration = apply_confidence_floor(
                self.calibration, confidence_threshold
            )
            self.calibration_warnings.append(
                f"CLI confidence floor {float(confidence_threshold):.3f} applied to emitted entities, relations, and canonical fields"
            )

    def extract(
        self,
        ocr_result: Mapping[str, Any],
        *,
        page_number: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        words = list(ocr_result.get("words") or [])
        if not words:
            return {
                "entities": [], "relations": [], "canonical_fields": {}, "tables": [],
                "document_type": {"label": "unknown", "confidence": None},
                "warnings": [*self.calibration_warnings, "multi-task model received no OCR words"],
            }
        texts = [str(word.get("text", "")) for word in words]
        boxes = [
            normalize_bbox(_bounded_bbox(word["bbox"], width, height), width, height)
            for word in words
        ]
        encoding = self.tokenizer(
            texts,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in {"input_ids", "bbox", "attention_mask", "token_type_ids"}
        }
        with self.torch.inference_mode():
            output = self.model(**model_inputs)
        entity_windows = _temperature_softmax(
            output.entity_logits, self.calibration["temperatures"]["entity"], self.torch
        ).cpu().tolist()
        canonical_windows = _temperature_softmax(
            output.canonical_logits, self.calibration["temperatures"]["canonical"], self.torch
        ).cpu().tolist()
        document_windows = _temperature_softmax(
            output.document_logits, self.calibration["temperatures"]["document"], self.torch
        ).cpu().tolist()
        word_ids = [
            list(encoding.word_ids(batch_index=index))
            for index in range(len(entity_windows))
        ]
        entity_scores = aggregate_word_probabilities(entity_windows, word_ids, word_count=len(words))
        canonical_scores = aggregate_word_probabilities(canonical_windows, word_ids, word_count=len(words))
        entity_predictions = []
        entity_threshold = float(self.calibration["thresholds"]["entity"])
        for scores in entity_scores:
            label_id = max(range(len(scores)), key=lambda index: scores[index])
            confidence = float(scores[label_id])
            entity_predictions.append({
                "label": self.id_to_entity_label.get(label_id, "O") if confidence >= entity_threshold else "O",
                "confidence": confidence,
            })
        entities = entities_from_word_predictions(words, entity_predictions, page_number=page_number)
        word_index_by_id = {str(word.get("id", "")): index for index, word in enumerate(words)}
        relation_entities = []
        for entity in entities:
            item = dict(entity)
            item["word_indices"] = [
                word_index_by_id[word_id]
                for word_id in item.get("word_ids", [])
                if word_id in word_index_by_id
            ]
            relation_entities.append(item)
        pairs, relation_scores = self._relation_probabilities(
            relation_entities, encoding, model_inputs, word_ids, width, height
        )
        document_scores = [
            statistics.fmean(window[index] for window in document_windows)
            for index in range(len(DOCUMENT_TYPE_LABELS))
        ]
        document_label, document_confidence = decode_document_type(
            document_scores,
            threshold=float(self.calibration["thresholds"]["document"]),
        )
        return {
            "entities": entities,
            "relations": decode_relations(
                pairs,
                relation_scores,
                page_number=page_number,
                threshold=float(self.calibration["thresholds"]["relation"]),
            ),
            "canonical_fields": decode_canonical_fields(
                words,
                canonical_scores,
                page_number=page_number,
                thresholds=self.calibration.get("canonical_field_thresholds")
                or {"default": float(self.calibration["thresholds"]["canonical"])},
            ),
            "tables": reconstruct_tables(entities, page_number=page_number),
            "document_type": {"label": document_label, "confidence": document_confidence},
            "warnings": list(self.calibration_warnings),
        }

    def _relation_probabilities(
        self,
        entities: Sequence[Mapping[str, Any]],
        encoding: Any,
        model_inputs: Mapping[str, Any],
        window_word_ids: Sequence[Sequence[int | None]],
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], list[list[float]]]:
        sums: dict[tuple[str, str], list[float]] = {}
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        metadata: dict[tuple[str, str], dict[str, Any]] = {}
        for window_index, word_ids in enumerate(window_word_ids):
            pairs = build_inference_relation_pairs(
                entities,
                word_ids=word_ids,
                page_width=width,
                page_height=height,
            )
            if not pairs:
                continue
            batch = {key: value[window_index : window_index + 1] for key, value in model_inputs.items()}
            batch.update({
                "relation_source_masks": self.torch.tensor(
                    [[pair["source_mask"] for pair in pairs]], dtype=self.torch.float32, device=self.device
                ),
                "relation_target_masks": self.torch.tensor(
                    [[pair["target_mask"] for pair in pairs]], dtype=self.torch.float32, device=self.device
                ),
                "relation_geometry": self.torch.tensor(
                    [[pair["geometry"] for pair in pairs]], dtype=self.torch.float32, device=self.device
                ),
                "relation_source_types": self.torch.tensor(
                    [[pair["source_type_id"] for pair in pairs]], dtype=self.torch.long, device=self.device
                ),
                "relation_target_types": self.torch.tensor(
                    [[pair["target_type_id"] for pair in pairs]], dtype=self.torch.long, device=self.device
                ),
            })
            with self.torch.inference_mode():
                logits = self.model(**batch).relation_logits
            vectors = _temperature_softmax(
                logits, self.calibration["temperatures"]["relation"], self.torch
            )[0].cpu().tolist()
            for pair, vector in zip(pairs, vectors, strict=True):
                key = (str(pair["source_id"]), str(pair["target_id"]))
                if key not in sums:
                    sums[key] = [0.0] * len(vector)
                    metadata[key] = dict(pair)
                sums[key] = [left + float(right) for left, right in zip(sums[key], vector)]
                counts[key] += 1
        keys = sorted(sums)
        return [metadata[key] for key in keys], [
            [value / counts[key] for value in sums[key]] for key in keys
        ]


def _field_candidate(
    words: Sequence[Mapping[str, Any]],
    tagged: Sequence[tuple[str | None, float]],
    indices: Sequence[int],
    page_number: int,
) -> dict[str, Any]:
    field = str(tagged[indices[0]][0])
    raw_text = " ".join(str(words[index].get("text", "")) for index in indices).strip()
    bbox = _union_bbox([words[index]["bbox"] for index in indices])
    return {
        "value": _normalize_field_value(field, raw_text),
        "raw_text": raw_text,
        "polygon": bbox_to_polygon(bbox),
        "bbox": bbox,
        "confidence": statistics.fmean(tagged[index][1] for index in indices),
        "method": "model:canonical_evidence",
        "extraction_source": "model",
        "validation_status": "unvalidated",
        "page_number": int(page_number),
    }


def _normalize_field_value(field: str, raw_text: str) -> str:
    value = raw_text.strip()
    if field in {"subtotal", "discount", "service_charge", "tax", "total_amount", "paid_amount", "balance"}:
        cleaned = re.sub(r"(?:THB|TRY|USD|EUR|GBP|[$€£฿₺]|\s)", "", value, flags=re.I)
        if cleaned.count(",") and cleaned.count("."):
            cleaned = cleaned.replace(".", "").replace(",", ".") if cleaned.rfind(",") > cleaned.rfind(".") else cleaned.replace(",", "")
        elif cleaned.count(",") == 1 and len(cleaned.rsplit(",", 1)[1]) in {2, 3}:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            return format(Decimal(cleaned).quantize(Decimal("0.01")), "f")
        except InvalidOperation:
            return value
    if field == "currency":
        folded = value.casefold()
        for marker, code in (("thb", "THB"), ("฿", "THB"), ("try", "TRY"), ("₺", "TRY"), ("usd", "USD"), ("$", "USD"), ("eur", "EUR"), ("€", "EUR"), ("gbp", "GBP"), ("£", "GBP")):
            if marker in folded:
                return code
    return value


def _union_bbox(boxes: Sequence[Sequence[Any]]) -> list[float]:
    values = [list(map(float, box)) for box in boxes]
    return [
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    ]


def _bounded_bbox(value: Sequence[Any], width: int, height: int) -> list[float]:
    if len(value) != 4:
        raise ValueError("OCR bbox must have four values")
    x0, y0, x1, y1 = map(float, value)
    bounded = [
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    ]
    if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
        raise ValueError(f"invalid OCR bbox after clipping: {value}")
    return bounded


def _temperature_softmax(logits: Any, temperature: float, torch: Any) -> Any:
    value = max(0.05, min(10.0, float(temperature)))
    return torch.softmax(logits / value, dim=-1)


def _load_calibration(
    calibration_path: str | Path | None,
    checkpoint: Path,
) -> tuple[dict[str, Any], list[str]]:
    default = {
        "temperatures": dict(DEFAULT_TEMPERATURES),
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "canonical_field_thresholds": {},
    }
    if calibration_path is None:
        return default, ["calibration unavailable; conservative default thresholds active"]
    path = Path(calibration_path)
    if not path.is_file():
        raise FileNotFoundError(f"required calibration file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(payload.get("checkpoint_model_sha256", ""))
    model_path = checkpoint / "model.safetensors"
    if not expected_hash or not model_path.is_file() or sha256_file(model_path) != expected_hash:
        raise ValueError("calibration artifact is not bound to this checkpoint")
    temperatures = {**DEFAULT_TEMPERATURES, **dict(payload.get("temperatures") or {})}
    thresholds = {**DEFAULT_THRESHOLDS, **dict(payload.get("thresholds") or {})}
    for name, value in temperatures.items():
        if not math.isfinite(float(value)) or not 0.05 <= float(value) <= 10.0:
            raise ValueError(f"invalid calibration temperature for {name}")
    for name, value in thresholds.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid calibration threshold for {name}")
    return {
        "temperatures": temperatures,
        "thresholds": thresholds,
        "canonical_field_thresholds": dict(payload.get("canonical_field_thresholds") or {}),
    }, []


def apply_confidence_floor(
    calibration: Mapping[str, Any], confidence_threshold: float
) -> dict[str, Any]:
    """Apply an inference-only minimum without weakening fitted calibration."""
    floor = float(confidence_threshold)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    result = copy.deepcopy(dict(calibration))
    thresholds = result.setdefault("thresholds", {})
    for head in ("entity", "canonical", "relation"):
        thresholds[head] = max(floor, float(thresholds.get(head, 0.0)))
    field_thresholds = result.setdefault("canonical_field_thresholds", {})
    for field, value in list(field_thresholds.items()):
        field_thresholds[field] = max(floor, float(value))
    if not field_thresholds:
        field_thresholds["default"] = thresholds["canonical"]
    elif "default" in field_thresholds:
        field_thresholds["default"] = max(floor, float(field_thresholds["default"]))
    return result
