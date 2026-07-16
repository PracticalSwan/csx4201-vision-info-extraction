from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import extract_document

from src.information_extraction.entity_inference import entities_from_word_predictions
from src.information_extraction.entity_worker_client import SubprocessLayoutEntityExtractor
from src.inference.document_io import DocumentInputError, DocumentPage, load_document_pages
from src.inference.document_pipeline import DocumentPipeline, merge_document_fields
from src.inference.output_writer import require_private_output_root


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_general_cli_rejects_configured_private_input_without_private_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    private_root = tmp_path / "private" / "gmail" / "invoices"
    cfg = {
        "paths": {
            "project_root": str(tmp_path),
            "gmail_invoices": str(private_root),
            "external_assets": str(tmp_path / "external"),
            "private_outputs": str(tmp_path / "private-output"),
        }
    }
    monkeypatch.setattr(extract_document.cfgmod, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        extract_document,
        "configure_external_environment",
        lambda *_: pytest.fail("environment setup ran before private-input validation"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_document.py",
            "--input",
            str(private_root / "synthetic-private.pdf"),
            "--output",
            str(tmp_path / "public-output"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        extract_document.main()

    assert exc.value.code == 2
    assert "requires --private-output" in capsys.readouterr().err


class FakeOCR:
    def extract_page(self, image, **kwargs):
        word = {
            "id": "word-1",
            "text": "Custom Key: Value",
            "confidence": 0.95,
            "polygon": [[5.0, 5.0], [95.0, 5.0], [95.0, 20.0], [5.0, 20.0]],
            "bbox": [5.0, 5.0, 95.0, 20.0],
        }
        return {
            "full_text": word["text"],
            "words": [word],
            "lines": [{
                "id": "line-1", "text": word["text"], "word_ids": [word["id"]],
                "polygon": word["polygon"], "bbox": word["bbox"], "confidence": 0.95,
            }],
            "mean_confidence": 0.95,
            "detector_model": "PP-OCRv6_medium_det",
            "recognizer_model": "PP-OCRv6_medium_rec",
            "language_route": "general",
            "orientation": 90.0,
            "duration_seconds": 0.01,
            "warnings": [],
            "provenance_hash": "12345678abcdef",
            "candidate_scores": [{"orientation": 90.0, "total": 1.0}],
            "candidate_transform": {"forward": IDENTITY, "inverse": IDENTITY},
        }


class BlankOCR:
    def extract_page(self, image, **kwargs):
        del image, kwargs
        return {
            "full_text": "", "words": [], "lines": [], "mean_confidence": 0.0,
            "detector_model": "PP-OCRv6_medium_det",
            "recognizer_model": "PP-OCRv6_medium_rec",
            "language_route": "general", "orientation": 0.0,
            "duration_seconds": 0.01,
            "warnings": ["no text detected"],
            "provenance_hash": "blank-public-fixture",
            "candidate_scores": [{"orientation": 0.0, "total": 0.0}],
            "candidate_transform": {"forward": IDENTITY, "inverse": IDENTITY},
        }


class FirstPageFailsOCR(FakeOCR):
    def __init__(self) -> None:
        self.calls = 0

    def extract_page(self, image, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic first-page failure")
        return super().extract_page(image, **kwargs)


class WrongKMeans:
    def predict(self, image):
        return {"cluster_id": 3, "zone": 4, "confidence": 0.01}


class BrokenKMeans:
    def predict(self, image):
        raise FileNotFoundError("artifact missing")


class FakeMultiTaskExtractor:
    def extract(self, ocr_result, *, page_number: int, width: int, height: int):
        del ocr_result, width, height
        key = {
            "id": "model-key", "label": "KEY", "text": "Total",
            "word_ids": ["word-1"], "polygon": [[5, 5], [45, 5], [45, 20], [5, 20]],
            "bbox": [5, 5, 45, 20], "confidence": 0.91, "page_number": page_number,
        }
        value = {
            "id": "model-value", "label": "VALUE", "text": "12.50",
            "word_ids": ["word-1"], "polygon": [[50, 5], [95, 5], [95, 20], [50, 20]],
            "bbox": [50, 5, 95, 20], "confidence": 0.93, "page_number": page_number,
        }
        return {
            "entities": [key, value],
            "relations": [{
                "id": "model-relation", "type": "KEY_VALUE", "source_id": "model-key",
                "target_id": "model-value", "confidence": 0.88, "page_number": page_number,
            }],
            "canonical_fields": {
                "total_amount": {
                    "value": "12.50", "raw_text": "12.50",
                    "polygon": value["polygon"], "bbox": value["bbox"],
                    "confidence": 0.92, "method": "model:canonical_evidence",
                    "extraction_source": "model", "validation_status": "unvalidated",
                    "page_number": page_number,
                }
            },
            "tables": [{
                "id": "table-1", "page_number": page_number,
                "method": "geometry:table_cell_grid", "source": "model_entities",
                "confidence": 0.8,
                "row_count": 1, "column_count": 2, "bbox": [5, 5, 95, 20],
                "cells": [],
                "header_row_index": None, "rows": [], "source_polygons": [],
                "warnings": ["fixture"], "raw_ocr_fallback": "",
            }],
            "document_type": {"label": "receipt", "confidence": 0.87},
            "warnings": [],
        }


def test_unknown_document_returns_generic_pair_and_wrong_kmeans_does_not_control_ocr() -> None:
    pipeline = DocumentPipeline(
        ocr=FakeOCR(), device="cpu", kmeans_predictor=WrongKMeans(), entity_extractor=None
    )
    result = pipeline.extract_pages(
        document_id="doc", source_type="image",
        pages=[DocumentPage(1, Image.new("RGB", (100, 100), "white"))],
    )

    assert result["document_type"]["label"] == "unknown"
    assert result["pages"][0]["selected_ocr_orientation"] == 90.0
    assert result["rotation_display"]["zone"] == 4
    assert result["rotation_display"]["purpose"] == "display_only"
    assert {entity["label"] for entity in result["pages"][0]["entities"]} == {"KEY", "VALUE"}
    assert result["pages"][0]["key_value_pairs"][0]["type"] == "KEY_VALUE"


def test_kmeans_failure_is_schema_valid_and_non_blocking() -> None:
    pipeline = DocumentPipeline(
        ocr=FakeOCR(), device="cpu", kmeans_predictor=BrokenKMeans(), entity_extractor=None
    )
    result = pipeline.extract_pages(
        document_id="doc", source_type="image",
        pages=[DocumentPage(1, Image.new("RGB", (100, 100), "white"))],
    )
    assert result["pages"][0]["full_text"] == "Custom Key: Value"
    assert result["rotation_display"]["cluster_id"] is None
    assert "failed independently" in result["rotation_display"]["warning"]


def test_blank_document_returns_schema_valid_null_safe_result() -> None:
    result = DocumentPipeline(
        ocr=BlankOCR(), device="cpu", entity_extractor=None,
        enable_kmeans_display=False,
    ).extract_pages(
        document_id="blank", source_type="image",
        pages=[DocumentPage(1, Image.new("RGB", (100, 100), "white"))],
    )

    assert result["pages"][0]["full_text"] == ""
    assert result["pages"][0]["entities"] == []
    assert result["pages"][0]["key_value_pairs"] == []
    assert all(value is None for value in result["fields"].values())


def test_multipage_failure_isolated_when_continue_is_enabled() -> None:
    result = DocumentPipeline(
        ocr=FirstPageFailsOCR(), device="cpu", entity_extractor=None,
        enable_kmeans_display=False,
    ).extract_pages(
        document_id="partial", source_type="pdf",
        pages=[
            DocumentPage(1, Image.new("RGB", (100, 100), "white")),
            DocumentPage(2, Image.new("RGB", (100, 100), "white")),
        ],
        continue_on_page_error=True,
    )

    assert result["pages"][0]["ocr"]["detector_model"] == "unavailable"
    assert result["pages"][1]["full_text"] == "Custom Key: Value"
    assert any("page 1 failed" in warning for warning in result["warnings"])


def test_document_pipeline_uses_all_trained_heads_and_model_tables() -> None:
    pipeline = DocumentPipeline(
        ocr=FakeOCR(),
        device="cpu",
        entity_extractor=FakeMultiTaskExtractor(),
        enable_kmeans_display=False,
    )
    result = pipeline.extract_pages(
        document_id="doc",
        source_type="image",
        pages=[DocumentPage(1, Image.new("RGB", (100, 100), "white"))],
    )

    assert result["document_type"] == {"label": "receipt", "confidence": 0.87}
    assert result["fields"]["total_amount"]["value"] == "12.50"
    assert result["pages"][0]["key_value_pairs"][0]["id"] == "model-relation"
    assert result["pages"][0]["tables"][0]["method"] == "geometry:table_cell_grid"


def test_multipage_field_conflict_abstains_when_confidences_are_tied() -> None:
    candidates = [
        {"total_amount": {
            "value": value, "raw_text": value,
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "bbox": [0, 0, 10, 10], "confidence": confidence,
            "method": "model:canonical_evidence", "extraction_source": "model",
            "validation_status": "unvalidated", "page_number": page,
        }}
        for value, confidence, page in (("10.00", 0.80, 1), ("20.00", 0.82, 2))
    ]

    fields, warnings = merge_document_fields(candidates)

    assert "total_amount" not in fields
    assert any("abstained" in warning for warning in warnings)


def test_entity_predictions_group_bio_spans() -> None:
    words = [
        {"id": "a", "text": "Invoice", "bbox": [0, 0, 20, 10]},
        {"id": "b", "text": "Number", "bbox": [21, 0, 45, 10]},
        {"id": "c", "text": "123", "bbox": [50, 0, 70, 10]},
    ]
    predictions = [
        {"label": "B-KEY", "confidence": 0.9},
        {"label": "I-KEY", "confidence": 0.8},
        {"label": "B-VALUE", "confidence": 0.95},
    ]
    entities = entities_from_word_predictions(words, predictions, page_number=2)
    assert [(entity["label"], entity["text"]) for entity in entities] == [
        ("KEY", "Invoice Number"), ("VALUE", "123")
    ]
    assert all(entity["page_number"] == 2 for entity in entities)


def test_image_loader_accepts_uppercase_extension_and_flattens_alpha(tmp_path: Path) -> None:
    path = tmp_path / "alpha.PNG"
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path)
    _, source_type, pages = load_document_pages(path)
    assert source_type == "image"
    assert pages[0].image.mode == "RGB"
    assert pages[0].image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("suffix", [".tiff", ".bmp", ".webp"])
def test_image_loader_accepts_declared_safe_image_formats(
    tmp_path: Path, suffix: str,
) -> None:
    path = tmp_path / f"document{suffix}"
    Image.new("RGB", (20, 20), "white").save(path)

    _, source_type, pages = load_document_pages(path)

    assert source_type == "image"
    assert pages[0].image.size == (20, 20)


def test_input_errors_are_actionable(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.touch()
    with pytest.raises(DocumentInputError, match="empty"):
        load_document_pages(empty)
    unsupported = tmp_path / "value.txt"
    unsupported.write_text("data", encoding="utf-8")
    with pytest.raises(DocumentInputError, match="unsupported"):
        load_document_pages(unsupported)


def test_invalid_layout_checkpoint_fails_before_worker_launch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        SubprocessLayoutEntityExtractor(tmp_path / "missing-checkpoint")


def test_private_output_must_remain_under_private_root(tmp_path: Path) -> None:
    private = tmp_path / "private"
    require_private_output_root(private / "result", private)
    with pytest.raises(ValueError, match="private root"):
        require_private_output_root(tmp_path / "public", private)


def test_multipage_pdf_loader(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "two-pages.pdf"
    document = fitz.open()
    document.new_page(width=100, height=100)
    document.new_page(width=100, height=100)
    document.save(path)
    document.close()
    _, source_type, pages = load_document_pages(path, pdf_dpi=72)
    assert source_type == "pdf"
    assert [page.page_number for page in pages] == [1, 2]


def test_pdf_page_limit_and_encryption_error(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    plain = tmp_path / "three.pdf"
    document = fitz.open()
    for _ in range(3):
        document.new_page(width=100, height=100)
    document.save(plain)
    document.close()
    assert len(load_document_pages(plain, pdf_dpi=72, max_pages=2)[2]) == 2

    encrypted = tmp_path / "protected.pdf"
    document = fitz.open()
    document.new_page(width=100, height=100)
    document.save(
        encrypted,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()
    with pytest.raises(DocumentInputError, match="password-protected"):
        load_document_pages(encrypted, pdf_dpi=72)


def test_corrupt_inputs_small_and_pixel_gate(tmp_path: Path) -> None:
    corrupt_image = tmp_path / "bad.jpg"
    corrupt_image.write_bytes(b"not an image")
    with pytest.raises(DocumentInputError, match="cannot be decoded"):
        load_document_pages(corrupt_image)
    corrupt_pdf = tmp_path / "bad.pdf"
    corrupt_pdf.write_bytes(b"not a pdf")
    with pytest.raises(DocumentInputError, match="cannot be opened"):
        load_document_pages(corrupt_pdf)
    tiny = tmp_path / "tiny.png"
    Image.new("L", (1, 1), 0).save(tiny)
    with pytest.raises(DocumentInputError, match="too small"):
        load_document_pages(tiny)
    large = tmp_path / "large.png"
    Image.new("L", (20, 20), 255).save(large)
    with pytest.raises(DocumentInputError, match="pixel"):
        load_document_pages(large, max_pixels=100)


def test_cmyk_exif_unicode_and_read_only_images(tmp_path: Path) -> None:
    cmyk = tmp_path / "เอกสาร CMYK.JPG"
    Image.new("CMYK", (12, 8), (0, 0, 0, 0)).save(cmyk)
    cmyk.chmod(0o444)
    assert load_document_pages(cmyk)[2][0].image.mode == "RGB"
    cmyk.chmod(0o666)

    oriented = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (10, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(oriented, exif=exif)
    loaded = load_document_pages(oriented)[2][0].image
    assert loaded.size == (20, 10)
