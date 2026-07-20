from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.portable.gui import (
    APP_CSS,
    _clean_log_line,
    _on_document_change,
    _preview_document,
)


def test_preview_document_renders_uploaded_image(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    Image.new("RGB", (48, 32), "navy").save(source)

    preview, note = _preview_document(str(source))

    assert isinstance(preview, Image.Image)
    assert preview.size == (48, 32)
    assert "sample.png" in note
    assert "Previewing the image locally" in note


def test_preview_document_renders_only_first_pdf_page(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page(width=144, height=72)
    document.new_page(width=72, height=144)
    document.save(source)
    document.close()

    preview, note = _preview_document(str(source))

    assert isinstance(preview, Image.Image)
    assert preview.width > preview.height
    assert "first PDF page" in note


def test_preview_document_clears_when_no_file_is_selected() -> None:
    preview, note = _preview_document(None)

    assert preview is None
    assert note == "Upload an image or PDF to preview it here."


def test_document_change_clears_stale_results(tmp_path: Path) -> None:
    source = tmp_path / "replacement.png"
    Image.new("RGB", (12, 12), "white").save(source)

    outputs = _on_document_change(str(source))

    assert len(outputs) == 9
    assert "Ready to extract **replacement.png**" in outputs[2]
    assert outputs[3:] == ([], "", None, [], None, "")


def test_result_panes_have_bounded_independent_scroll_contract() -> None:
    assert "#ocr-text textarea" in APP_CSS
    assert "#run-log textarea" in APP_CSS
    assert "overflow-y: scroll !important" in APP_CSS
    assert "scrollbar-gutter: stable" in APP_CSS


def test_run_log_strips_terminal_color_sequences() -> None:
    assert _clean_log_line("\x1b[32mCreating model\x1b[0m") == "Creating model"
