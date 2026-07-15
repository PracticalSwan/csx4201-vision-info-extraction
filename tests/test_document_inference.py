from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.information_extraction.entity_inference import entities_from_word_predictions
from src.inference.document_io import DocumentInputError, DocumentPage, load_document_pages
from src.inference.document_pipeline import DocumentPipeline
from src.inference.output_writer import require_private_output_root


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


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


class WrongKMeans:
    def predict(self, image):
        return {"cluster_id": 3, "zone": 4, "confidence": 0.01}


class BrokenKMeans:
    def predict(self, image):
        raise FileNotFoundError("artifact missing")


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


def test_input_errors_are_actionable(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.touch()
    with pytest.raises(DocumentInputError, match="empty"):
        load_document_pages(empty)
    unsupported = tmp_path / "value.txt"
    unsupported.write_text("data", encoding="utf-8")
    with pytest.raises(DocumentInputError, match="unsupported"):
        load_document_pages(unsupported)


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
