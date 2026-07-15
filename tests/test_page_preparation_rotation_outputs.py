from __future__ import annotations

import copy
import csv
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.orientation_features import extract_rotation_features, load_feature_split
from src.page_preparation import PAGE_COLUMNS, PREPARATION_ERROR_COLUMNS, prepare_page_images
from src.rotation_common import RotationPipelineError, read_csv_rows, sha256_file
from src.rotation_dataset import ROTATION_COLUMNS, SPLIT_COLUMNS, generate_rotation_data
from tests.rotation_test_helpers import make_rotation_config, page_row, split_row, write_csv


INVENTORY_COLUMNS = [
    "file_id",
    "document_id",
    "dataset",
    "current_relative_path",
    "extension",
    "is_image",
    "is_pdf",
    "is_annotation",
    "is_readable",
    "is_empty",
    "sha256",
    "document_category",
]


def _create_pdf(path: Path, page_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(page_count):
        page = document.new_page(width=144, height=72)
        page.insert_text((12, 30), f"synthetic page {index + 1}")
        page.draw_rect(fitz.Rect(8, 8, 136, 64), color=(0, 0, 0), width=1)
    document.save(path)
    document.close()


def _inventory_row(
    file_id: str,
    dataset: str,
    relative_path: str,
    *,
    image: bool = False,
    pdf: bool = False,
    sha256: str = "",
) -> dict[str, object]:
    return {
        "file_id": file_id,
        "document_id": f"document_{file_id}",
        "dataset": dataset,
        "current_relative_path": relative_path,
        "extension": Path(relative_path).suffix.lower(),
        "is_image": image,
        "is_pdf": pdf,
        "is_annotation": False,
        "is_readable": True,
        "is_empty": False,
        "sha256": sha256,
        "document_category": "document",
    }


def test_page_preparation_renders_single_and_multipage_pdfs_handles_exif_and_preserves_raw(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    metadata = tmp_path / "data/metadata"
    public_image = tmp_path / "data/raw/public/sroie/train/img/oriented.jpg"
    public_image.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[274] = 6  # Display rotated 90 degrees clockwise: width/height swap.
    Image.new("RGB", (60, 30), (220, 230, 240)).save(public_image, exif=exif)

    single_pdf = tmp_path / "data/raw/private/gmail/Secret Single.pdf"
    multiple_pdf = tmp_path / "data/raw/private/gmail/Private Multipage.pdf"
    malformed_pdf = tmp_path / "data/raw/private/gmail/Broken Statement.pdf"
    _create_pdf(single_pdf, 1)
    _create_pdf(multiple_pdf, 2)
    malformed_pdf.parent.mkdir(parents=True, exist_ok=True)
    malformed_pdf.write_bytes(b"not a PDF")

    public_rows = [
        _inventory_row(
            "public_image",
            "sroie",
            "sroie/train/img/oriented.jpg",
            image=True,
            sha256=sha256_file(public_image),
        ),
        _inventory_row("gmail_single", "gmail", "gmail/receipts/gmail_0001.pdf"),
        _inventory_row("gmail_multiple", "gmail", "gmail/invoices/gmail_0002.pdf"),
        _inventory_row("gmail_broken", "gmail", "gmail/legal/gmail_0003.pdf"),
    ]
    private_rows = [
        _inventory_row(
            "gmail_single", "gmail", "gmail/Secret Single.pdf", pdf=True, sha256=sha256_file(single_pdf)
        ),
        _inventory_row(
            "gmail_multiple", "gmail", "gmail/Private Multipage.pdf", pdf=True, sha256=sha256_file(multiple_pdf)
        ),
        _inventory_row(
            "gmail_broken", "gmail", "gmail/Broken Statement.pdf", pdf=True, sha256=sha256_file(malformed_pdf)
        ),
    ]
    write_csv(metadata / "file_inventory.csv", public_rows, INVENTORY_COLUMNS)
    write_csv(metadata / "private_file_inventory.csv", private_rows, INVENTORY_COLUMNS)
    raw_before = {
        path.relative_to(tmp_path).as_posix(): sha256_file(path)
        for path in (tmp_path / "data/raw").rglob("*")
        if path.is_file()
    }

    dry_run = prepare_page_images(cfg, dry_run=True)
    assert dry_run["dry_run"] is True
    assert dry_run["summary"]["selected_pages"] == 4
    assert not (metadata / "page_manifest.csv").exists()
    assert not (tmp_path / "data/processed/private/page_images").exists()

    result = prepare_page_images(cfg)
    assert result["summary"]["private_documents"] == 2
    assert result["summary"]["private_pages"] == 3
    assert result["summary"]["preparation_errors"] == 1
    pages = read_csv_rows(metadata / "page_manifest.csv")
    public_page = next(row for row in pages if row["dataset"] == "sroie")
    private_pages = [row for row in pages if row["dataset"] == "gmail"]
    assert (int(public_page["prepared_width"]), int(public_page["prepared_height"])) == (30, 60)
    assert public_page["materialization_mode"] == "referenced_image"
    assert (tmp_path / public_page["prepared_image_path"]).resolve() == public_image.resolve()
    assert Counter(int(row["source_page_count"]) for row in private_pages) == {1: 1, 2: 2}
    assert sorted(int(row["source_page_number"]) for row in private_pages if row["source_page_count"] == "2") == [1, 2]
    for row in private_pages:
        rendered = tmp_path / row["prepared_image_path"]
        assert rendered.is_file()
        with Image.open(rendered) as image:
            assert image.mode == "RGB"
            assert image.size == (144, 72)

    public_manifest_text = (metadata / "page_manifest.csv").read_text(encoding="utf-8")
    assert "Secret Single.pdf" not in public_manifest_text
    assert "Private Multipage.pdf" not in public_manifest_text
    assert "Broken Statement.pdf" not in public_manifest_text
    with (metadata / "page_manifest.csv").open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == PAGE_COLUMNS
    with (metadata / "rotation_preparation_errors.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PREPARATION_ERROR_COLUMNS
        errors = list(reader)
    assert len(errors) == 1
    assert errors[0]["operation"] == "open_pdf"
    assert errors[0]["error_type"]

    rendered_before = {
        path.name: (path.stat().st_mtime_ns, sha256_file(path))
        for path in (tmp_path / "data/processed/private/page_images").glob("*.png")
    }
    rerun = prepare_page_images(cfg)
    rendered_after = {
        path.name: (path.stat().st_mtime_ns, sha256_file(path))
        for path in (tmp_path / "data/processed/private/page_images").glob("*.png")
    }
    assert rerun["summary"]["selected_pages"] == result["summary"]["selected_pages"]
    assert rendered_after == rendered_before
    assert len(read_csv_rows(metadata / "page_manifest.csv")) == len(pages)

    raw_after = {
        path.relative_to(tmp_path).as_posix(): sha256_file(path)
        for path in (tmp_path / "data/raw").rglob("*")
        if path.is_file()
    }
    assert raw_after == raw_before


def test_page_preparation_invalidates_stale_pdf_pages_and_materializes_public_images(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    metadata = tmp_path / "data/metadata"
    public_image = tmp_path / "data/raw/public/sroie/train/img/source.jpg"
    public_image.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (60, 30), (210, 220, 230)).save(public_image, exif=exif)
    private_pdf = tmp_path / "data/raw/private/gmail/Synthetic Private.pdf"
    _create_pdf(private_pdf, 1)

    public_rows = [
        _inventory_row(
            "public_source",
            "sroie",
            "sroie/train/img/source.jpg",
            image=True,
            sha256=sha256_file(public_image),
        ),
        _inventory_row("private_source", "gmail", "gmail/legal/gmail_0001.pdf"),
    ]
    private_rows = [
        _inventory_row(
            "private_source",
            "gmail",
            "gmail/Synthetic Private.pdf",
            pdf=True,
            sha256=sha256_file(private_pdf),
        )
    ]
    write_csv(metadata / "file_inventory.csv", public_rows, INVENTORY_COLUMNS)
    write_csv(metadata / "private_file_inventory.csv", private_rows, INVENTORY_COLUMNS)

    first = prepare_page_images(cfg)
    first_rows = read_csv_rows(metadata / "page_manifest.csv")
    first_private = next(row for row in first_rows if row["dataset"] == "gmail")
    private_output = tmp_path / first_private["prepared_image_path"]
    first_private_hash = sha256_file(private_output)
    assert (int(first_private["prepared_width"]), int(first_private["prepared_height"])) == (144, 72)

    higher_dpi = copy.deepcopy(cfg)
    higher_dpi["page_preparation"]["pdf_dpi"] = 144
    second = prepare_page_images(higher_dpi)
    second_rows = read_csv_rows(metadata / "page_manifest.csv")
    second_private = next(row for row in second_rows if row["dataset"] == "gmail")
    assert second["summary"]["selected_pages"] == first["summary"]["selected_pages"]
    assert second_private["preparation_hash"] != first_private["preparation_hash"]
    assert (int(second_private["prepared_width"]), int(second_private["prepared_height"])) == (288, 144)
    assert sha256_file(private_output) != first_private_hash

    materialized = copy.deepcopy(higher_dpi)
    materialized["page_preparation"]["materialize_existing_images"] = True
    prepare_page_images(materialized)
    materialized_rows = read_csv_rows(metadata / "page_manifest.csv")
    public_page = next(row for row in materialized_rows if row["dataset"] == "sroie")
    public_output = tmp_path / public_page["prepared_image_path"]
    assert public_page["materialization_mode"] == "converted_image"
    assert public_output.is_relative_to(tmp_path / "data/processed/public/page_images")
    with Image.open(public_output) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (30, 60)
        assert image.info["preparation_hash"] == public_page["preparation_hash"]

    stable_before = (public_output.stat().st_mtime_ns, sha256_file(public_output))
    prepare_page_images(materialized)
    stable_after = (public_output.stat().st_mtime_ns, sha256_file(public_output))
    assert stable_after == stable_before

    Image.new("RGB", (60, 30), (20, 40, 220)).save(public_image, exif=exif)
    with pytest.raises(RotationPipelineError, match="live source hash differs from audited inventory"):
        prepare_page_images(materialized)

    public_rows[0]["sha256"] = sha256_file(public_image)
    write_csv(metadata / "file_inventory.csv", public_rows, INVENTORY_COLUMNS)
    prepare_page_images(materialized)
    assert sha256_file(public_output) != stable_before[1]


def _asymmetric_color_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, 15, 15), fill=(255, 0, 0))
    draw.rectangle((80, 1, 94, 15), fill=(0, 180, 0))
    draw.rectangle((1, 48, 15, 62), fill=(0, 0, 255))
    draw.rectangle((80, 48, 94, 62), fill=(0, 0, 0))
    draw.line((20, 31, 75, 31), fill=(0, 0, 0), width=3)
    image.save(path)


def _write_rotation_input_manifests(tmp_path: Path, *, include_missing: bool = False):
    metadata = tmp_path / "data/metadata"
    source = tmp_path / "data/processed/public/page_images/color_page.png"
    _asymmetric_color_page(source)
    page = page_row("color_page", prepared_image_path=source.relative_to(tmp_path).as_posix())
    pages = [page]
    splits = [split_row(page, "train")]
    if include_missing:
        missing = page_row(
            "missing_page",
            dataset="funsd",
            prepared_image_path="data/processed/public/page_images/does_not_exist.png",
        )
        pages.append(missing)
        splits.append(split_row(missing, "train"))
    write_csv(metadata / "page_manifest.csv", pages, PAGE_COLUMNS)
    write_csv(metadata / "split_manifest.csv", splits, SPLIT_COLUMNS)
    return source


def test_rotation_materialization_expands_canvas_preserves_content_and_propagates_feature_metadata(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    source = _write_rotation_input_manifests(tmp_path)
    source_hash = sha256_file(source)

    dry_run = generate_rotation_data(cfg, profile="smoke", dry_run=True)
    assert dry_run["expected_rotations"] == 4
    assert not (tmp_path / "data/metadata/rotation_manifest.csv").exists()
    assert not (tmp_path / "data/processed/rotated_images").exists()

    result = generate_rotation_data(cfg, profile="smoke")
    assert result["summary"]["successful_rows"] == 4
    assert result["summary"]["failed_rows"] == 0
    assert result["summary"]["counts_by_zone"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    manifest_path = tmp_path / "data/metadata/rotation_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ROTATION_COLUMNS
        rotations = list(reader)
    assert len(rotations) == 4
    for row in rotations:
        output = tmp_path / row["rotated_image_path"]
        assert output.is_file()
        assert output.resolve().is_relative_to((tmp_path / "data/processed/rotated_images").resolve())
        with Image.open(output) as image:
            assert image.mode == "RGB"
            assert image.width > 96 and image.height > 64
            pixels = np.asarray(image)
        assert np.all(pixels[0, 0] >= 245)
        assert np.count_nonzero((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80) & (pixels[:, :, 2] < 80)) > 20
        assert np.count_nonzero((pixels[:, :, 1] > 100) & (pixels[:, :, 0] < 100) & (pixels[:, :, 2] < 100)) > 20
        assert np.count_nonzero((pixels[:, :, 2] > 180) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 80)) > 20
        assert np.count_nonzero(np.all(pixels < 60, axis=2)) > 20
    image_artifacts_before = {
        row["rotated_image_path"]: (tmp_path / row["rotated_image_path"]).stat().st_mtime_ns
        for row in rotations
    }
    rerun = generate_rotation_data(cfg, profile="smoke")
    assert rerun["summary"]["successful_rows"] == 4
    assert len(read_csv_rows(manifest_path)) == 4
    assert {
        row["rotated_image_path"]: (tmp_path / row["rotated_image_path"]).stat().st_mtime_ns
        for row in rotations
    } == image_artifacts_before
    assert sha256_file(source) == source_hash

    extracted = extract_rotation_features(cfg, profile="smoke")
    assert extracted["skipped"] is False
    values = load_feature_split(cfg, "train")
    assert list(values["rotation_ids"]) == [row["rotation_id"] for row in rotations]
    assert set(values["datasets"]) == {"sroie"}
    np.testing.assert_array_equal(values["true_zones"], np.asarray([int(row["rotation_zone"]) for row in rotations]))
    np.testing.assert_allclose(values["true_angles"], np.asarray([float(row["normalized_angle"]) for row in rotations]))
    assert extract_rotation_features(cfg, profile="smoke")["skipped"] is True

    changed = make_rotation_config(tmp_path)
    changed["rotation_features"]["hough"]["angle_bins"] = 18
    invalidated = extract_rotation_features(changed, profile="smoke")
    assert invalidated["skipped"] is False
    assert invalidated["configuration_hash"] != extracted["configuration_hash"]


def test_rotation_generation_records_each_failed_variant_without_partial_artifacts(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    _write_rotation_input_manifests(tmp_path, include_missing=True)
    result = generate_rotation_data(cfg, profile="smoke")
    rows = read_csv_rows(tmp_path / "data/metadata/rotation_manifest.csv")
    errors = read_csv_rows(tmp_path / "data/metadata/rotation_preparation_errors.csv")
    assert len(rows) == 8
    assert Counter(row["generation_status"] for row in rows) == {"success": 4, "failed": 4}
    assert len(result["errors"]) == 4
    assert len(errors) == 4
    assert {row["operation"] for row in errors} == {"load_source"}
    assert all(row["error_message"] for row in errors)
    assert all(not row["rotated_image_path"] for row in rows if row["generation_status"] == "failed")


def test_rotation_generation_regenerates_artifacts_when_configuration_changes(tmp_path: Path):
    cfg = make_rotation_config(tmp_path)
    _write_rotation_input_manifests(tmp_path)
    generate_rotation_data(cfg, profile="smoke")
    manifest_path = tmp_path / "data/metadata/rotation_manifest.csv"
    first_rows = read_csv_rows(manifest_path)
    first = next(row for row in first_rows if float(row["normalized_angle"]) == 45.0)
    output = tmp_path / first["rotated_image_path"]
    first_hash = sha256_file(output)
    first_size = (int(first["output_width"]), int(first["output_height"]))

    changed = copy.deepcopy(cfg)
    changed["rotation_generation"]["output_max_dimension"] = 32
    generate_rotation_data(changed, profile="smoke")
    second_rows = read_csv_rows(manifest_path)
    second = next(row for row in second_rows if float(row["normalized_angle"]) == 45.0)
    second_hash = sha256_file(output)
    second_size = (int(second["output_width"]), int(second["output_height"]))
    assert second["configuration_hash"] != first["configuration_hash"]
    assert second_hash != first_hash
    assert second_size != first_size
    with Image.open(output) as image:
        assert image.info["configuration_hash"] == second["configuration_hash"]

    stable_before = (output.stat().st_mtime_ns, second_hash)
    generate_rotation_data(changed, profile="smoke")
    stable_after = (output.stat().st_mtime_ns, sha256_file(output))
    assert stable_after == stable_before
