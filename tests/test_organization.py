"""Tests for organize_data modes and the no-forbidden-outputs invariant.

Calls organize_data functions directly (the test sandbox blocks subprocess).
Scenarios covered (from the project plan):
  15 reference mode does not move files
  16 dry-run mode does not change files
  17 move mode preserves file counts
  18 copy mode preserves file counts
  19 raw file hashes do not change after organization
  22 no training folders created
  23 no model files created
  24 no rotated images generated
  25 no OCR outputs generated
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import organize_data
from src import dataset_discovery as dd

from .conftest import build_sroie, make_config

FORBIDDEN_DIRS = ["models", "rotated_images", "processed_images", "ocr_output", "outputs", "checkpoints"]
FORBIDDEN_MODEL_EXTS = (".pt", ".pth", ".onnx", ".safetensors")


def _setup(tmp_path: Path):
    """Build a legacy sroie dataset and a config that searches the legacy tree."""
    legacy_root = tmp_path / "vision_info_extraction_data" / "public_train"
    legacy_root.mkdir(parents=True)
    build_sroie(legacy_root)  # legacy_root/sroie_receipts/SROIE2019/...
    cfg = make_config(tmp_path, candidate_roots=("vision_info_extraction_data", "data/raw"))
    return cfg


def _sroie_ds(cfg):
    return next(d for d in dd.discover_datasets(cfg) if d.name == "sroie")


def _count(root: Path) -> int:
    return sum(1 for _ in root.rglob("*") if _.is_file())


def _first_file_hash(root: Path):
    f = next(p for p in root.rglob("*") if p.is_file())
    return f, hashlib.sha256(f.read_bytes()).hexdigest()


def _assert_no_forbidden_outputs(tmp_path: Path) -> None:
    for d in FORBIDDEN_DIRS:
        assert not (tmp_path / d).exists(), f"forbidden dir created: {d}"
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in FORBIDDEN_MODEL_EXTS:
            raise AssertionError(f"model file created: {p}")


def test_reference_mode_does_not_move_files(tmp_path):
    cfg = _setup(tmp_path)
    src = _sroie_ds(cfg).current_path
    before = _count(src)
    actions = organize_data.organize_one(_sroie_ds(cfg), "reference", False, False, True)
    assert src.exists(), "reference mode must not remove the source"
    assert _count(src) == before
    assert actions[0]["action"].startswith("referenced")
    _assert_no_forbidden_outputs(tmp_path)


def test_dry_run_mode_does_not_change_files(tmp_path):
    cfg = _setup(tmp_path)
    src = _sroie_ds(cfg).current_path
    before = _count(src)
    actions = organize_data.organize_one(_sroie_ds(cfg), "move", True, False, True)
    assert src.exists(), "dry-run must not move anything"
    assert _count(src) == before
    assert actions[0]["action"].startswith("would-move")
    _assert_no_forbidden_outputs(tmp_path)


def test_move_mode_preserves_file_counts_and_hashes(tmp_path):
    cfg = _setup(tmp_path)
    ds = _sroie_ds(cfg)
    src = ds.current_path
    before = _count(src)
    sample, sample_hash = _first_file_hash(src)
    rel = sample.relative_to(src)
    actions = organize_data.organize_one(ds, "move", False, False, True)
    assert not src.exists(), "move must remove the source"
    target = ds.target_path
    assert target.exists()
    assert _count(target) == before, "file count must be preserved by move"
    moved = target / rel
    assert moved.exists()
    assert hashlib.sha256(moved.read_bytes()).hexdigest() == sample_hash, \
        "raw file hash must not change after move"
    assert actions[0]["action"] == "moved"
    _assert_no_forbidden_outputs(tmp_path)


def test_copy_mode_preserves_file_counts(tmp_path):
    cfg = _setup(tmp_path)
    ds = _sroie_ds(cfg)
    src = ds.current_path
    before = _count(src)
    organize_data.organize_one(ds, "copy", False, False, True)
    assert src.exists(), "copy must keep the source"
    target = ds.target_path
    assert target.exists()
    assert _count(src) == before
    assert _count(target) == before, "file count must be preserved by copy"
    _assert_no_forbidden_outputs(tmp_path)


def test_no_training_or_model_or_rotation_or_ocr_outputs(tmp_path):
    """End-to-end: a full move must not create any later-stage artifacts."""
    cfg = _setup(tmp_path)
    organize_data.organize_one(_sroie_ds(cfg), "move", False, False, True)
    _assert_no_forbidden_outputs(tmp_path)


def test_move_is_idempotent(tmp_path):
    """Re-running move after organization reports already-organized."""
    cfg = _setup(tmp_path)
    ds = _sroie_ds(cfg)
    first = organize_data.organize_one(ds, "move", False, False, True)
    assert first[0]["action"] == "moved"
    target = ds.target_path
    count_after_first = _count(target)
    # Re-discover (now finds at target) and organize again.
    ds2 = _sroie_ds(cfg)
    second = organize_data.organize_one(ds2, "move", False, False, True)
    assert _count(target) == count_after_first, "idempotent re-run must not change counts"
    assert second[0]["action"].startswith("already-organized")
