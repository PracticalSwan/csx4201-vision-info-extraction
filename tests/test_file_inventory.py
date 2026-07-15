"""Tests for file inventory: columns, validation, and error recording."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src import file_inventory as inv
from src.dataset_discovery import DatasetInfo

from .conftest import make_config


def _ds(root: Path, name: str = "sroie") -> DatasetInfo:
    return DatasetInfo(name=name, source_type="public",
                       current_path=root, target_path=root, confidence="high")


def _build(root: Path, cfg):
    return inv.build_inventory([_ds(root)], cfg)


def _save_img(path: Path, size=(4, 4), color=(1, 2, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_inventory_has_required_columns(tmp_path):
    root = tmp_path / "sroie"
    _save_img(root / "img" / "x.jpg")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    assert result.rows, "no rows produced"
    for col in inv.INVENTORY_COLUMNS:
        assert col in result.rows[0], f"missing column {col}"


def test_empty_file_is_recorded(tmp_path):
    root = tmp_path / "sroie"
    (root / "img").mkdir(parents=True)
    (root / "img" / "empty.jpg").write_bytes(b"")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "empty.jpg")
    assert row["is_empty"] is True
    assert row["is_readable"] is False
    assert any(e["error_type"] != "" for e in result.errors) or "empty" in row["notes"]


def test_invalid_json_is_recorded(tmp_path):
    root = tmp_path / "sroie"
    (root / "ann").mkdir(parents=True)
    (root / "ann" / "bad.json").write_text("{not valid json", encoding="utf-8")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "bad.json")
    assert row["is_readable"] is False
    assert "invalid_json" in row["notes"]


def test_corrupted_image_is_recorded(tmp_path):
    root = tmp_path / "sroie"
    (root / "img").mkdir(parents=True)
    (root / "img" / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n this is not a real png")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "broken.png")
    assert row["is_image"] is True
    assert row["is_readable"] is False


def test_corrupted_pdf_is_recorded(tmp_path):
    root = tmp_path / "sroie"
    (root / "doc").mkdir(parents=True)
    (root / "doc" / "broken.pdf").write_bytes(b"not a real pdf at all")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "broken.pdf")
    assert row["is_pdf"] is True
    assert row["is_readable"] is False


def test_valid_image_records_dimensions(tmp_path):
    root = tmp_path / "sroie"
    _save_img(root / "img" / "good.png", size=(16, 9), color=(5, 6, 7))
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "good.png")
    assert row["is_readable"] is True
    assert row["_width"] == 16 and row["_height"] == 9


def test_sha256_populated_when_enabled(tmp_path):
    root = tmp_path / "sroie"
    (root).mkdir(parents=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    cfg = make_config(tmp_path)
    result = _build(root, cfg)
    row = next(r for r in result.rows if r["original_filename"] == "a.txt")
    assert len(row["sha256"]) == 64


def test_document_id_assigned_stably(tmp_path):
    root = tmp_path / "sroie"
    _save_img(root / "x.jpg", size=(2, 2))
    cfg = make_config(tmp_path)
    r1 = _build(root, cfg).rows
    r2 = _build(root, cfg).rows
    assert [r["document_id"] for r in r1] == [r["document_id"] for r in r2]
    assert all(r["document_id"].startswith("sroie_") for r in r1)
