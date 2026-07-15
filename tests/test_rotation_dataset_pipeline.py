from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.page_preparation import PAGE_COLUMNS
from src.rotation_common import LeakageError
from src.rotation_dataset import (
    ROTATION_COLUMNS,
    SPLIT_COLUMNS,
    _rotate_page,
    _scan_private_name_leaks,
    _validate_split_rows,
    create_rotation_splits,
    verify_rotation_data,
)
from tests.rotation_test_helpers import (
    build_valid_verification_artifacts,
    make_rotation_config,
    page_row,
    split_row,
    write_csv,
)


def test_splits_are_deterministic_group_exact_duplicates_and_isolate_private(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    metadata = tmp_path / "data/metadata"
    pages = [page_row(f"page_{index:02d}", dataset="sroie") for index in range(16)]
    pages[0]["sha256"] = "same-content"
    pages[1]["sha256"] = "same-content"
    secret_name = "Real Person Invoice 2026.pdf"
    private = page_row(
        "private_page_001",
        document_id="private_doc_001",
        private=True,
        prepared_image_path="data/processed/private/page_images/private_page_001.png",
    )
    assert secret_name not in repr(private)
    pages.append(private)
    write_csv(metadata / "page_manifest.csv", pages, PAGE_COLUMNS)

    first = create_rotation_splits(cfg, dry_run=True, seed=123)
    second = create_rotation_splits(cfg, dry_run=True, seed=123)
    assert first["rows"] == second["rows"]

    rows = {row["page_id"]: row for row in first["rows"]}
    assert rows["page_00"]["project_split"] == rows["page_01"]["project_split"]
    assert rows["page_00"]["split_group_id"] == rows["page_01"]["split_group_id"]
    assert rows["page_00"]["exact_duplicate_group"] == rows["page_01"]["exact_duplicate_group"]
    assert rows["page_00"]["exact_duplicate_group"].startswith("exact_")
    assert {row["project_split"] for row in rows.values() if row["private_status"] == "public"} == {
        "train",
        "validation",
        "test",
    }
    assert rows["private_page_001"]["project_split"] == "private_test"
    assert all(
        row["project_split"] != "private_test"
        for row in rows.values()
        if row["private_status"] == "public"
    )
    assert secret_name not in repr(first)


def test_split_validator_rejects_document_and_private_leakage():
    public = page_row("public_a", document_id="shared_doc")
    crossed = [split_row(public, "train", "g1"), split_row(public | {"page_id": "public_b"}, "test", "g2")]
    with pytest.raises(LeakageError, match="document leakage"):
        _validate_split_rows(crossed)

    duplicate_page = [
        split_row(page_row("same_page", document_id="doc_train"), "train", "g_train"),
        split_row(page_row("same_page", document_id="doc_test"), "test", "g_test"),
    ]
    with pytest.raises(LeakageError, match="page leakage"):
        _validate_split_rows(duplicate_page)

    private = split_row(page_row("private_a", private=True), "train")
    with pytest.raises(LeakageError, match="private page entered a public split"):
        _validate_split_rows([private])


def test_rotation_worker_uses_counterclockwise_direction_and_boundary_zone(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    cfg["rotation_generation"]["smoke_angles"] = {
        "zone_1": [90],
        "zone_2": [],
        "zone_3": [],
        "zone_4": [],
    }
    source = tmp_path / "data/processed/public/page_images/asymmetric.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 50), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 18, 78, 32), fill="black")
    image.save(source)
    page = page_row("asymmetric", prepared_image_path=source.relative_to(tmp_path).as_posix())
    split = split_row(page, "train")

    rows, errors = _rotate_page(
        tmp_path,
        split,
        page,
        cfg,
        "smoke",
        tmp_path / "data/processed/rotated_images/smoke",
        "cfg_hash",
        False,
    )
    assert errors == []
    assert len(rows) == 1
    row = rows[0]
    assert row["normalized_angle"] == 90
    assert row["rotation_zone"] == 2
    assert row["rotation_direction"] == "counterclockwise"
    assert "/zone_2/" in f"/{row['rotated_image_path']}/"
    assert row["rotated_image_path"].endswith("_angle_090_zone_2.png")

    with Image.open(tmp_path / row["rotated_image_path"]) as rotated:
        gray = np.asarray(rotated.convert("L"))
    # A mark at the source right edge moves to the output top edge under CCW rotation.
    assert gray[: gray.shape[0] // 2].mean() < gray[gray.shape[0] // 2 :].mean()


def test_verifier_reports_manifest_corruption(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    artifacts = build_valid_verification_artifacts(tmp_path, cfg)
    clean = verify_rotation_data(cfg, profile="smoke")
    assert clean["all_passed"], clean

    corrupted = copy.deepcopy(artifacts["rotations"])
    corrupted[0]["rotation_zone"] = 4
    write_csv(tmp_path / "data/metadata/rotation_manifest.csv", corrupted, ROTATION_COLUMNS)
    result = verify_rotation_data(cfg, profile="smoke")
    checks = {check["name"]: check for check in result["checks"]}
    assert not result["all_passed"]
    assert checks["rotation-artifacts-valid"]["passed"] is False
    assert "3/4 valid" in checks["rotation-artifacts-valid"]["detail"]


def test_verifier_reports_cross_split_document_leakage(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    artifacts = build_valid_verification_artifacts(tmp_path, cfg)
    second_page = page_row(
        "page_b",
        document_id=artifacts["page"]["document_id"],
        prepared_image_path=artifacts["page"]["prepared_image_path"],
    )
    write_csv(
        tmp_path / "data/metadata/page_manifest.csv",
        [artifacts["page"], second_page],
        PAGE_COLUMNS,
    )
    write_csv(
        tmp_path / "data/metadata/split_manifest.csv",
        [artifacts["split"], split_row(second_page, "test", "group_b")],
        SPLIT_COLUMNS,
    )

    result = verify_rotation_data(cfg, profile="smoke")
    checks = {check["name"]: check for check in result["checks"]}
    assert not result["all_passed"]
    assert checks["split-leakage"]["passed"] is False
    assert "document leakage" in checks["split-leakage"]["detail"]


def test_private_filename_scan_covers_committable_source_and_tests(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    metadata = tmp_path / "data/metadata"
    private_name = "Synthetic Confidential Statement.pdf"
    write_csv(
        metadata / "private_file_inventory.csv",
        [{"file_id": "private_1", "original_filename": private_name}],
        ["file_id", "original_filename"],
    )
    leak = tmp_path / "tests/test_accidental_private_literal.py"
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_text(f"PRIVATE_LITERAL = {private_name!r}\n", encoding="utf-8")

    result = _scan_private_name_leaks(cfg)
    assert result["passed"] is False
    assert "tests" in result["detail"]
    assert private_name not in result["detail"]

    leak.unlink()
    assert _scan_private_name_leaks(cfg)["passed"] is True
