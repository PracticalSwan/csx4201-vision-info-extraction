"""Checkpoint-backed multilingual text+2D-layout entity inference."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.information_extraction.geometry import bbox_to_polygon
from src.information_extraction.layoutxlm_data import normalize_bbox
from src.rotation_common import stable_id


class LayoutEntityExtractor:
    """Run the trained Detectron2-free LayoutXLM token classifier."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        max_length: int = 512,
        stride: int = 64,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"layout model checkpoint is missing: {checkpoint}")
        import torch
        from transformers import LayoutXLMTokenizerFast

        from src.information_extraction.layoutxlm_model import (
            LayoutXLMTextLayoutForTokenClassification,
        )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA layout inference requested but PyTorch CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = LayoutXLMTokenizerFast.from_pretrained(
            checkpoint, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.model = LayoutXLMTextLayoutForTokenClassification.from_pretrained(checkpoint)
        self.model.to(self.device).eval()
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.id_to_label = {
            int(key): str(value) for key, value in self.model.config.id2label.items()
        }

    def extract(
        self,
        ocr_result: Mapping[str, Any],
        *,
        page_number: int,
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        words = list(ocr_result.get("words") or [])
        if not words:
            return [], ["layout model received no OCR words"]
        texts = [str(word.get("text", "")) for word in words]
        boxes = [normalize_bbox(_bounded_bbox(word["bbox"], width, height), width, height) for word in words]
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
        score_sums = [None] * len(words)
        score_counts = [0] * len(words)
        with self.torch.inference_mode():
            probabilities = self.model(**model_inputs).logits.softmax(dim=-1).cpu()
        for batch_index in range(probabilities.shape[0]):
            word_ids = encoding.word_ids(batch_index=batch_index)
            seen_in_window: set[int] = set()
            for token_index, word_index in enumerate(word_ids):
                if word_index is None or word_index in seen_in_window:
                    continue
                seen_in_window.add(word_index)
                vector = probabilities[batch_index, token_index]
                score_sums[word_index] = vector if score_sums[word_index] is None else score_sums[word_index] + vector
                score_counts[word_index] += 1
        predictions = []
        for index, word in enumerate(words):
            if score_sums[index] is None:
                predictions.append({"label": "O", "confidence": 0.0})
                continue
            scores = score_sums[index] / max(1, score_counts[index])
            label_id = int(scores.argmax())
            predictions.append({
                "label": self.id_to_label.get(label_id, "O"),
                "confidence": float(scores[label_id]),
            })
        return entities_from_word_predictions(words, predictions, page_number=page_number), []


def entities_from_word_predictions(
    words: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    page_number: int,
) -> list[dict[str, Any]]:
    if len(words) != len(predictions):
        raise ValueError("word and prediction counts differ")
    groups: list[tuple[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]] = []
    current_label = ""
    current: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for word, prediction in zip(words, predictions, strict=True):
        raw_label = str(prediction.get("label", "O"))
        if raw_label == "O" or "-" not in raw_label:
            if current:
                groups.append((current_label, current))
                current = []
            current_label = ""
            continue
        prefix, label = raw_label.split("-", 1)
        if prefix == "B" or label != current_label:
            if current:
                groups.append((current_label, current))
            current_label, current = label, []
        current.append((word, prediction))
    if current:
        groups.append((current_label, current))

    entities: list[dict[str, Any]] = []
    for index, (label, members) in enumerate(groups):
        member_words = [member[0] for member in members]
        boxes = [list(map(float, word["bbox"])) for word in member_words]
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        word_ids = [str(word["id"]) for word in member_words]
        entities.append({
            "id": stable_id("entity", page_number, index, label, *word_ids),
            "label": label,
            "text": " ".join(str(word["text"]) for word in member_words).strip(),
            "word_ids": word_ids,
            "polygon": bbox_to_polygon(bbox),
            "bbox": bbox,
            "confidence": sum(float(member[1].get("confidence", 0.0)) for member in members) / len(members),
            "page_number": int(page_number),
        })
    return entities


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
