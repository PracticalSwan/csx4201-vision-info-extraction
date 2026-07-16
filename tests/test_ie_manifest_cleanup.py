from __future__ import annotations

import csv
from pathlib import Path

from src.information_extraction import manifest as manifest_module
from tests.conftest import make_config


def test_unsupported_normalized_record_is_not_materialized(
    tmp_path: Path, monkeypatch,
) -> None:
    metadata = tmp_path / "data" / "metadata"
    metadata.mkdir(parents=True)
    page_id = "coru_unsupported_001"
    columns = (
        "page_id", "document_id", "dataset", "dataset_component",
        "private_status", "original_dataset_split", "prepared_image_path",
        "usability_status", "annotation_availability", "sha256",
    )
    with (metadata / "page_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "page_id": page_id,
            "document_id": "coru_document_001",
            "dataset": "coru",
            "dataset_component": "Receipt Images & Key Information Detection",
            "private_status": "public",
            "original_dataset_split": "train",
            "prepared_image_path": "data/raw/public/coru/page.jpg",
            "usability_status": "usable",
            "annotation_availability": "yes",
            "sha256": "0" * 64,
        })

    monkeypatch.setattr(
        manifest_module,
        "normalize_public_page",
        lambda *_args, **_kwargs: {
            "alignment_status": "unsupported",
            "tokens": [],
            "source_qa": [],
            "entities": [],
            "relations": [],
            "canonical_fields": {},
            "annotation_provenance": {"source_paths": ["source.json"]},
        },
    )

    result = manifest_module.build_information_extraction_manifest(
        make_config(tmp_path), force=True,
    )

    output = (
        tmp_path / "data" / "processed" / "normalized_ie_annotations"
        / "coru" / f"{page_id}.json"
    )
    assert result["exclusion_counts"]["unsupported_annotation_semantics"] == 1
    assert not output.exists()
    public_manifest = (metadata / "information_extraction_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert "unsupported_annotation_semantics" in public_manifest
    assert f"{page_id}.json" not in public_manifest
