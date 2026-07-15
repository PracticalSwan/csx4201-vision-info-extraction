#!/usr/bin/env python3
"""Evaluate cached public OCR-only and OCR-plus-rules smoke baselines."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.evaluation.metrics import normalized_text, ocr_text_metrics  # noqa: E402
from src.information_extraction.rules import extract_rule_fields  # noqa: E402
from src.rotation_common import atomic_write_json, read_csv_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    manifest = read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "model_dataset_manifest.csv")
    rows = [row for row in manifest if row.get("is_usable") == "true" and row.get("is_private") == "false"]
    ocr_totals: Counter[str] = Counter()
    field_totals: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    warning_count = 0
    for row in rows:
        example = json.loads(Path(row["model_example_path"]).read_text(encoding="utf-8"))
        annotation = json.loads((PROJECT_ROOT / row["normalized_annotation_path"]).read_text(encoding="utf-8"))
        words = example["tokens"]
        ocr = {
            "words": words,
            "lines": [
                {
                    "id": f"line-{index}", "text": word["text"], "word_ids": [word["id"]],
                    "polygon": word["polygon"], "bbox": word["bbox"], "confidence": word["confidence"],
                }
                for index, word in enumerate(words)
            ],
        }
        reference = " ".join(token["text"] for token in annotation.get("tokens") or [])
        prediction = " ".join(word["text"] for word in words)
        metrics = ocr_text_metrics(reference, prediction)
        ocr_totals["character_errors"] += metrics["character_errors"]
        ocr_totals["reference_characters"] += metrics["reference_characters"]
        ocr_totals["word_errors"] += metrics["word_errors"]
        ocr_totals["reference_words"] += len(normalized_text(reference).split())
        ocr_totals["empty_outputs"] += metrics["empty_output"]
        fields, warnings = extract_rule_fields(ocr)
        warning_count += len(warnings)
        for name, expected in (annotation.get("canonical_fields") or {}).items():
            if not isinstance(expected, dict) or expected.get("value") in (None, ""):
                continue
            field_totals["applicable"] += 1
            by_dataset[row["dataset"]]["applicable"] += 1
            by_language[row["language"]]["applicable"] += 1
            predicted = fields.get(name)
            if predicted and normalized_text(predicted["value"]) == normalized_text(expected["value"]):
                field_totals["correct"] += 1
                by_dataset[row["dataset"]]["correct"] += 1
                by_language[row["language"]]["correct"] += 1
        field_totals["predicted"] += len(fields)
    ocr_report = {
        "schema_version": "1.0", "profile": "smoke", "public_only": True,
        "examples": len(rows), "gmail_fit_rows": 0,
        "cer": ocr_totals["character_errors"] / max(1, ocr_totals["reference_characters"]),
        "wer": ocr_totals["word_errors"] / max(1, ocr_totals["reference_words"]),
        "empty_output_rate": ocr_totals["empty_outputs"] / max(1, len(rows)),
        "limitations": ["Uses the aligned smoke subset and cached upright PaddleOCR tokens."],
    }
    rule_report = {
        "schema_version": "1.0", "profile": "smoke", "public_only": True,
        "examples": len(rows), "gmail_fit_rows": 0,
        "field_applicable": field_totals["applicable"], "field_correct": field_totals["correct"],
        "field_accuracy": field_totals["correct"] / max(1, field_totals["applicable"]),
        "predicted_field_count": field_totals["predicted"], "conflict_warning_count": warning_count,
        "by_dataset": _field_groups(by_dataset), "by_language": _field_groups(by_language),
        "limitations": ["Rules are evidence-only and are not a substitute for the layout model."],
    }
    reports = cfgmod.resolve_path(cfg, "reports")
    atomic_write_json(reports / "ocr" / "ocr_only_baseline.json", ocr_report)
    atomic_write_json(reports / "information_extraction" / "rule_baseline.json", rule_report)
    print(json.dumps({"ocr": ocr_report, "rules": rule_report}, indent=2))
    return 0 if rows else 1


def _field_groups(groups: dict[str, Counter[str]]) -> dict[str, Any]:
    return {
        key: {
            "applicable": values["applicable"], "correct": values["correct"],
            "accuracy": values["correct"] / max(1, values["applicable"]),
        }
        for key, values in sorted(groups.items())
    }


if __name__ == "__main__":
    raise SystemExit(main())
