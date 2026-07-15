"""Tests for dataset pair validation and unmatched-file reporting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src import dataset_validation as dv
from src import file_inventory as inv
from src.dataset_discovery import DatasetInfo

from .conftest import make_config


def _inventory_for(root: Path, name: str, cfg):
    ds = DatasetInfo(name=name, source_type="public",
                     current_path=root, target_path=root, confidence="high")
    return inv.build_inventory([ds], cfg).rows


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


def test_unmatched_image_without_annotation_recorded(tmp_path):
    root = tmp_path / "funsd" / "dataset" / "training_data"
    _png(root / "images" / "0000971160.png")
    (root / "annotations").mkdir(parents=True)
    # annotation present
    (root / "annotations" / "0000971160.json").write_text(json.dumps({"form": []}), encoding="utf-8")
    # orphan image with no annotation
    _png(root / "images" / "orphan.png")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "funsd", "funsd", cfg)
    unmatched = dv.find_unmatched(rows)
    paths = [u["file_path"] for u in unmatched]
    assert any("orphan" in p for p in paths)
    assert all(u["match_status"] == "unmatched" for u in unmatched)


def test_unmatched_annotation_without_image_recorded(tmp_path):
    root = tmp_path / "funsd" / "dataset" / "testing_data"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "lonely.json").write_text(json.dumps({"form": []}), encoding="utf-8")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "funsd", "funsd", cfg)
    unmatched = dv.find_unmatched(rows)
    assert any("lonely" in u["file_path"] for u in unmatched)


def test_matched_pairs_not_reported(tmp_path):
    root = tmp_path / "funsd" / "dataset" / "training_data"
    _png(root / "images" / "pair.png")
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "pair.json").write_text(json.dumps({"form": []}), encoding="utf-8")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "funsd", "funsd", cfg)
    unmatched = dv.find_unmatched(rows)
    assert unmatched == []


def test_sroie_triplet_matching(tmp_path):
    root = tmp_path / "sroie" / "SROIE2019" / "test"
    _png(root / "img" / "X1.jpg")
    (root / "box").mkdir(parents=True)
    (root / "box" / "X1.txt").write_text("1,1,2,1,2,2,1,2,t", encoding="utf-8")
    (root / "entities").mkdir(parents=True)
    (root / "entities" / "X1.txt").write_text("{}", encoding="utf-8")
    # orphan image
    _png(root / "img" / "X2.jpg")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "sroie", "sroie", cfg)
    unmatched = dv.find_unmatched(rows)
    assert any("X2" in u["file_path"] for u in unmatched)


def test_unmatched_csv_has_required_columns(tmp_path):
    root = tmp_path / "funsd" / "dataset" / "training_data"
    _png(root / "images" / "orphan.png")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "funsd", "funsd", cfg)
    unmatched = dv.find_unmatched(rows)
    out = tmp_path / "unmatched.csv"
    dv.write_unmatched_csv(unmatched, out)
    with out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == dv.UNMATCHED_COLUMNS


def test_fatura_matches_suffixed_annotation_formats(tmp_path):
    """COCO/HF annotations named <image>_coco_test.json must pair with the image."""
    root = tmp_path / "fatura" / "invoices_dataset_final"
    _png(root / "images" / "Template1_Instance0.jpg")
    for fmt_dir, suffix in (("Annotations/Original_Format", ""),
                            ("Annotations/COCO_compatible_format", "_coco_test"),
                            ("Annotations/layoutlm_HF_format", "_hugg_test")):
        p = root / fmt_dir / f"Template1_Instance0{suffix}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    cfg = make_config(tmp_path)
    rows = _inventory_for(tmp_path / "fatura", "fatura", cfg)
    unmatched = dv.find_unmatched(rows)
    assert unmatched == [], f"expected all FATURA annotations paired, got {unmatched}"
