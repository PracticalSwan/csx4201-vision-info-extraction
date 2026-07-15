"""Tests for exact (SHA-256) and near (perceptual) duplicate detection."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src import duplicate_detection as dup
from src import file_inventory as inv
from src.dataset_discovery import DatasetInfo

from .conftest import make_config


def _build(root: Path, cfg):
    ds = DatasetInfo(name="coru", source_type="public",
                     current_path=root, target_path=root, confidence="high")
    return inv.build_inventory([ds], cfg)


def test_exact_duplicates_grouped_by_sha256(tmp_path):
    root = tmp_path / "coru"
    (root).mkdir(parents=True)
    payload = b"identical bytes"
    (root / "a.txt").write_bytes(payload)
    (root / "b.txt").write_bytes(payload)
    (root / "c.txt").write_bytes(b"different bytes")
    cfg = make_config(tmp_path)
    rows = _build(root, cfg).rows
    groups = dup.find_exact_duplicates(rows)
    assert len(groups) == 1
    assert {r["original_filename"] for r in groups[0]} == {"a.txt", "b.txt"}


def test_no_false_exact_duplicate(tmp_path):
    root = tmp_path / "coru"
    root.mkdir()
    (root / "a.txt").write_bytes(b"one")
    (root / "b.txt").write_bytes(b"two")
    cfg = make_config(tmp_path)
    rows = _build(root, cfg).rows
    assert dup.find_exact_duplicates(rows) == []


def test_near_duplicates_detected_on_synthetic_images(tmp_path):
    """Two near-identical images cluster; a distinct, textured image does not."""
    import random
    random.seed(42)
    root = tmp_path / "coru" / "img"
    root.mkdir(parents=True)
    base = Image.new("RGB", (32, 32), (100, 100, 100))
    base.save(root / "base.jpg", "JPEG")
    near = base.copy()
    near.putpixel((0, 0), (101, 101, 101))  # tiny perturbation
    near.save(root / "near.jpg", "JPEG")
    # A high-frequency textured image: phash differs strongly from uniform base.
    different = Image.new("RGB", (32, 32))
    different.putdata([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                       for _ in range(32 * 32)])
    different.save(root / "different.jpg", "JPEG")

    cfg = make_config(tmp_path)
    rows = _build(root, cfg).rows
    groups = dup.find_near_duplicates(rows, cfg)
    grouped_ids = {r["original_filename"] for g in groups for r in g}
    assert "base.jpg" in grouped_ids and "near.jpg" in grouped_ids
    assert "different.jpg" not in grouped_ids


def test_duplicate_report_has_required_fields(tmp_path):
    root = tmp_path / "coru"
    root.mkdir()
    (root / "a.txt").write_bytes(b"x")
    (root / "b.txt").write_bytes(b"x")
    cfg = make_config(tmp_path)
    rows = _build(root, cfg).rows
    exact = dup.find_exact_duplicates(rows)
    report = dup.build_duplicate_report(exact, [])
    assert report
    for col in dup.DUPLICATE_COLUMNS:
        assert col in report[0]
    assert all(r["duplicate_type"] == "exact" for r in report)


def test_near_duplicate_respects_threshold(tmp_path):
    """With threshold 0 and clearly different textured images, no near-dup group forms."""
    import random
    root = tmp_path / "coru" / "img"
    root.mkdir(parents=True)
    for seed, name in ((1, "p1.jpg"), (2, "p2.jpg")):
        rnd = random.Random(seed)
        img = Image.new("RGB", (32, 32))
        img.putdata([(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))
                     for _ in range(32 * 32)])
        img.save(root / name, "JPEG")
    cfg = make_config(tmp_path)
    cfg["duplicates"]["perceptual_threshold"] = 0
    rows = _build(root, cfg).rows
    groups = dup.find_near_duplicates(rows, cfg)
    flat = {r["original_filename"] for g in groups for r in g}
    assert "p1.jpg" not in flat or "p2.jpg" not in flat
