from __future__ import annotations

import csv
import json
from pathlib import Path

from src.information_extraction.dataset_reporting import generate_final_dataset_reports


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_final_dataset_reports_are_aggregate_and_profile_bound(tmp_path: Path) -> None:
    example_path = tmp_path / "example.json"
    example_path.write_text(json.dumps({
        "is_private": False,
        "source_targets": {"tokens": [{}, {}, {}]},
        "alignment": {"unmatched_labels": [{}], "unmatched_ocr_tokens": [{}, {}]},
        "alignment_coverage": 2 / 3,
        "entity_retention_rate": 0.5,
        "relation_retention_rate": 1.0,
        "canonical_retention_rate": 1.0,
        "data_quality_score": 0.75,
        "canonical_fields": {"total_amount": {"evidence_valid": True}},
    }), encoding="utf-8")
    manifest_path = tmp_path / "final.csv"
    common = {
        "example_id": "e", "build_id": "final-build", "document_id": "d",
        "page_id": "p", "dataset": "sroie", "project_split": "train",
        "is_private": "false", "is_usable": "true", "exclusion_reason": "",
        "profile": "final", "model_example_path": str(example_path),
    }
    _write_csv(manifest_path, [
        {**common, "token_source": "ground_truth"},
        {**common, "example_id": "e2", "token_source": "paddleocr"},
    ])
    split_path = tmp_path / "splits.csv"
    _write_csv(split_path, [{
        "page_id": "p", "dataset": "sroie", "project_split": "train",
        "is_private": "false",
    }])

    summary = generate_final_dataset_reports(manifest_path, split_path, tmp_path / "reports")

    assert summary["usable_public_fit_pages"] == 1
    assert summary["private_fit_rows"] == 0
    metrics = json.loads((tmp_path / "reports" / "alignment_metrics.json").read_text())
    paddle = metrics["by_dataset"]["sroie"]["paddleocr"]
    assert paddle["matched_tokens"] == 2
    assert paddle["unmatched_tokens"] == 1
    assert "private" not in (tmp_path / "reports" / "dataset_quality.md").read_text().casefold() or "private/gmail fit rows" in (tmp_path / "reports" / "dataset_quality.md").read_text().casefold()
