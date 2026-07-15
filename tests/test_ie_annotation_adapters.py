from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from src.information_extraction.annotations import normalize_public_page


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _schema(project: Path) -> None:
    target = project / "data" / "metadata" / "annotation_schema.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "data" / "metadata" / "annotation_schema.json", target)


def _row(path: str, dataset: str, component: str = "") -> dict[str, str]:
    return {
        "document_id": f"{dataset}_doc_test",
        "page_id": f"{dataset}_page_test",
        "dataset": dataset,
        "dataset_component": component,
        "document_type": "invoice" if dataset == "fatura" else "receipt",
        "language": "tr" if dataset == "fatura" else "en",
        "prepared_image_path": path,
        "source_page_number": "1",
    }


def test_fatura_clips_source_boxes_to_real_image_bounds(tmp_path: Path) -> None:
    _schema(tmp_path)
    base = tmp_path / "data" / "raw" / "public" / "fatura" / "invoices_dataset_final"
    image = base / "images" / "invoice.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image)
    hf = base / "Annotations" / "layoutlm_HF_format" / "invoice_hugg_train.json"
    original = base / "Annotations" / "Original_Format" / "invoice.json"
    hf.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    hf.write_text(
        json.dumps({"words": ["TOTAL"], "bboxes": [[90, 10, 120, 20]], "ner_tags": [3]}),
        encoding="utf-8",
    )
    original.write_text(json.dumps({"TOTAL": {"text": "TOTAL"}}), encoding="utf-8")

    record = normalize_public_page(
        _row(image.relative_to(tmp_path).as_posix(), "fatura"), tmp_path, "train"
    )

    assert record["tokens"][0]["bbox"] == [90.0, 10.0, 100.0, 20.0]
    assert record["annotation_provenance"]["clipped_source_boxes"] == 1


def test_sroie_falls_back_to_windows_1252_and_skips_bad_rows(tmp_path: Path) -> None:
    _schema(tmp_path)
    base = tmp_path / "data" / "raw" / "public" / "sroie" / "SROIE2019" / "test"
    image = base / "img" / "receipt.jpg"
    box = base / "box" / "receipt.txt"
    entity = base / "entities" / "receipt.txt"
    image.parent.mkdir(parents=True)
    box.parent.mkdir(parents=True)
    entity.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image)
    box.write_bytes(
        "0,0,50,0,50,10,0,10,TOTAL £1.00\nmalformed\n10,10,10,10,10,10,10,10,bad".encode(
            "windows-1252"
        )
    )
    entity.write_text(json.dumps({"total": "£1.00"}), encoding="utf-8")

    record = normalize_public_page(
        _row(image.relative_to(tmp_path).as_posix(), "sroie"), tmp_path, "test"
    )

    assert [token["text"] for token in record["tokens"]] == ["TOTAL £1.00"]
    assert record["annotation_provenance"]["box_text_encoding"] == "windows-1252"
    assert [row["reason"] for row in record["annotation_provenance"]["skipped_source_rows"]] == [
        "fewer_than_9_values",
        "degenerate_geometry",
    ]
