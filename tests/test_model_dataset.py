from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from src.information_extraction.alignment import align_ocr_to_annotations
from src.information_extraction.layoutxlm_data import load_model_examples, to_bio_labels
from src.information_extraction.model_dataset import (
    assign_leakage_safe_splits,
    build_ground_truth_example,
    build_hybrid_example,
    build_paddleocr_example,
    compute_dataset_build_id,
    predict_model_data_ocr,
    profile_manifest_path,
    select_ocr_variant_rows,
    validate_manifest_profile,
    validate_profile_requirements,
    validate_reusable_example,
)


def _row(
    page_id: str,
    *,
    dataset: str = "funsd",
    document_id: str | None = None,
    duplicate_group_id: str = "",
    sha256: str = "",
    split: str = "train",
    private: bool = False,
) -> dict[str, str]:
    return {
        "page_id": page_id,
        "document_id": document_id or f"doc_{page_id}",
        "dataset": dataset,
        "dataset_component": "forms" if dataset == "funsd" else "receipts",
        "document_type": "form" if dataset == "funsd" else "receipt",
        "language": "en",
        "image_path": f"data/raw/public/{dataset}/{page_id}.png",
        "normalized_annotation_path": f"data/processed/normalized/{page_id}.json",
        "project_split": split,
        "duplicate_group_id": duplicate_group_id,
        "sha256": sha256,
        "is_private": "true" if private else "false",
        "is_usable": "true",
    }


def _annotation() -> dict:
    return {
        "schema_version": "1.0",
        "document_id": "doc_page_1",
        "page_id": "page_1",
        "dataset": "funsd",
        "dataset_component": "forms",
        "document_type": "form",
        "language": "en",
        "image_path": "data/raw/public/funsd/page_1.png",
        "page": {"width": 200, "height": 100},
        "tokens": [
            {
                "id": "source_key",
                "text": "Total",
                "polygon": [[0, 0], [40, 0], [40, 10], [0, 10]],
                "bbox": [0, 0, 40, 10],
                "entity_label": "KEY",
            },
            {
                "id": "source_value",
                "text": "10.00",
                "polygon": [[50, 0], [90, 0], [90, 10], [50, 10]],
                "bbox": [50, 0, 90, 10],
                "entity_label": "VALUE",
            },
            {
                "id": "source_note",
                "text": "Note",
                "polygon": [[0, 30], [40, 30], [40, 40], [0, 40]],
                "bbox": [0, 30, 40, 40],
                "entity_label": "HEADER",
            },
        ],
        "entities": [
            {
                "id": "entity_key",
                "label": "KEY",
                "text": "Total",
                "token_ids": ["source_key"],
                "polygon": [[0, 0], [40, 0], [40, 10], [0, 10]],
                "bbox": [0, 0, 40, 10],
            },
            {
                "id": "entity_value",
                "label": "VALUE",
                "text": "10.00",
                "token_ids": ["source_value"],
                "polygon": [[50, 0], [90, 0], [90, 10], [50, 10]],
                "bbox": [50, 0, 90, 10],
            },
            {
                "id": "entity_note",
                "label": "HEADER",
                "text": "Note",
                "token_ids": ["source_note"],
                "polygon": [[0, 30], [40, 30], [40, 40], [0, 40]],
                "bbox": [0, 30, 40, 40],
            },
        ],
        "relations": [
            {
                "id": "relation_total",
                "type": "KEY_VALUE",
                "source_id": "entity_key",
                "target_id": "entity_value",
            },
            {
                "id": "relation_note",
                "type": "HEADER_SECTION",
                "source_id": "entity_note",
                "target_id": "entity_value",
            },
        ],
        "canonical_fields": {
            "total_amount": {
                "value": "10.00",
                "raw_text": "10.00",
                "token_ids": ["source_value"],
                "source_key": "total",
            }
        },
        "source_qa": [],
        "project_split": "train",
        "is_private": False,
    }


def _ocr_words() -> list[dict]:
    return [
        {
            "id": "ocr_key",
            "text": "Total",
            "polygon": [[0, 0], [40, 0], [40, 10], [0, 10]],
            "bbox": [0, 0, 40, 10],
            "confidence": 0.95,
        },
        {
            "id": "ocr_value",
            "text": "10.00",
            "polygon": [[50, 0], [90, 0], [90, 10], [50, 10]],
            "bbox": [50, 0, 90, 10],
            "confidence": 0.90,
        },
    ]


def test_leakage_safe_splits_union_templates_hashes_and_hold_out_coru() -> None:
    rows = [
        _row("fat_1", dataset="fatura", duplicate_group_id="template_7", sha256="a"),
        _row("fat_2", dataset="fatura", duplicate_group_id="template_7", sha256="b"),
        _row("sroie_1", dataset="sroie", duplicate_group_id="left", sha256="same"),
        _row("sroie_2", dataset="sroie", duplicate_group_id="right", sha256="same"),
        _row("funsd_1", dataset="funsd"),
        _row("coru_1", dataset="coru"),
    ]

    first = assign_leakage_safe_splits(rows, seed=42, unseen_datasets={"coru"})
    second = assign_leakage_safe_splits(list(reversed(rows)), seed=42, unseen_datasets={"coru"})
    by_page = {row["page_id"]: row for row in first}
    repeated = {row["page_id"]: row for row in second}

    assert by_page["fat_1"]["project_split"] == by_page["fat_2"]["project_split"]
    assert by_page["fat_1"]["split_group_id"] == by_page["fat_2"]["split_group_id"]
    assert by_page["sroie_1"]["project_split"] == by_page["sroie_2"]["project_split"]
    assert by_page["sroie_1"]["split_group_id"] == by_page["sroie_2"]["split_group_id"]
    assert by_page["coru_1"]["project_split"] == "unseen_domain_test"
    assert {
        row["project_split"] for row in first if row["dataset"] != "coru"
    } <= {"train", "dev_select", "dev_calibration", "test_in_domain"}
    assert {
        page_id: (row["project_split"], row["split_group_id"])
        for page_id, row in by_page.items()
    } == {
        page_id: (row["project_split"], row["split_group_id"])
        for page_id, row in repeated.items()
    }


def test_profile_manifests_are_distinct_and_final_uses_required_name(tmp_path: Path) -> None:
    assert profile_manifest_path(tmp_path, "smoke") == tmp_path / "model_dataset_smoke_manifest.csv"
    assert profile_manifest_path(tmp_path, "development") == tmp_path / "model_dataset_development_manifest.csv"
    assert profile_manifest_path(tmp_path, "final") == tmp_path / "final_model_dataset_manifest.csv"


def test_split_assignment_balances_large_template_components() -> None:
    rows = [
        _row(
            f"fat_{template}_{page}",
            dataset="fatura",
            duplicate_group_id=f"template_{template}",
            sha256=f"sha_{template}_{page}",
        )
        for template in range(50)
        for page in range(10)
    ]
    rows.extend(_row(f"sroie_{index}", dataset="sroie") for index in range(100))
    rows.extend(_row(f"funsd_{index}", dataset="funsd") for index in range(20))

    assigned = assign_leakage_safe_splits(rows, seed=42, unseen_datasets={"coru"})
    counts = Counter(row["project_split"] for row in assigned)

    assert abs(counts["train"] - 434) <= 10
    assert abs(counts["dev_select"] - 62) <= 10
    assert abs(counts["dev_calibration"] - 31) <= 10
    assert abs(counts["test_in_domain"] - 93) <= 10


def test_ground_truth_example_preserves_real_targets_and_quality_metadata() -> None:
    row = _row("page_1", dataset="funsd")
    example = build_ground_truth_example(
        row,
        _annotation(),
        profile="final",
        split_group_id="group_1",
    )

    token_ids = {token["id"] for token in example["tokens"]}
    assert example["token_source"] == "ground_truth"
    assert example["example_id"] == "page_1__ground_truth"
    assert example["profile"] == "final"
    assert example["split_group_id"] == "group_1"
    assert example["token_alignment_score"] == 1.0
    assert example["entity_retention_rate"] == 1.0
    assert example["relation_retention_rate"] == 1.0
    assert example["data_quality_score"] == 1.0
    assert len(example["entity_ids"]) == len(example["tokens"])
    assert set(example["canonical_fields"]["total_amount"]["token_ids"]) <= token_ids
    assert example["relations"] == _annotation()["relations"]


def test_partial_paddleocr_alignment_keeps_valid_targets_and_remaps_evidence() -> None:
    annotation = _annotation()
    words = _ocr_words()
    alignment = align_ocr_to_annotations(words, annotation["tokens"])
    ocr = {
        "words": words,
        "detector_model": "PP-OCRv6_medium_det",
        "recognizer_model": "PP-OCRv6_medium_rec",
        "provenance_hash": "result-hash",
        "orientation": 343.0,
        "fine_deskew": {"correction_degrees": -17.0, "reliability": 0.98},
    }

    example = build_paddleocr_example(
        _row("page_1", dataset="funsd"),
        annotation,
        ocr,
        alignment,
        route="general",
        profile="final",
        split_group_id="group_1",
        model_hashes={
            "PP-OCRv6_medium_det": "detector-hash",
            "PP-OCRv6_medium_rec": "recognizer-hash",
        },
    )

    token_ids = {token["id"] for token in example["tokens"]}
    assert example["token_source"] == "paddleocr"
    assert example["inference_realistic"] is True
    assert example["alignment_coverage"] == pytest.approx(2 / 3)
    assert example["entity_retention_rate"] == pytest.approx(2 / 3)
    assert example["relation_retention_rate"] == pytest.approx(1 / 2)
    assert example["relations"] == [_annotation()["relations"][0]]
    assert set(example["canonical_fields"]["total_amount"]["token_ids"]) == {"ocr_value"}
    assert set(example["canonical_fields"]["total_amount"]["token_ids"]) <= token_ids
    assert example["ocr_confidence"] == pytest.approx(0.925)
    assert 0 < example["data_quality_score"] < 1
    assert example["has_valid_learning_target"] is True
    assert example["token_loss_mask"] == [True, True]
    assert example["source_targets"]["entities"] == annotation["entities"]
    assert example["ocr_provenance"]["selected_orientation"] == 343.0
    assert example["ocr_provenance"]["fine_deskew"]["correction_degrees"] == -17.0
    assert example["ocr_provenance"]["orientation_policy"] == "cardinal_plus_polygon_fine_deskew"


def test_hybrid_alignment_retains_explicit_ground_truth_fallbacks() -> None:
    annotation = _annotation()
    words = _ocr_words()
    alignment = align_ocr_to_annotations(words, annotation["tokens"])
    example = build_hybrid_example(
        _row("page_1", dataset="funsd"),
        annotation,
        {
            "words": words,
            "detector_model": "PP-OCRv6_medium_det",
            "recognizer_model": "PP-OCRv6_medium_rec",
            "provenance_hash": "result-hash",
        },
        alignment,
        route="general",
        profile="final",
        split_group_id="group_1",
        model_hashes={
            "PP-OCRv6_medium_det": "detector-hash",
            "PP-OCRv6_medium_rec": "recognizer-hash",
        },
    )

    assert example["token_source"] == "hybrid"
    assert example["inference_realistic"] is False
    assert example["training_only"] is True
    assert example["ground_truth_fallback_fraction"] == pytest.approx(1 / 3)
    assert {token["origin"] for token in example["tokens"]} == {
        "paddleocr",
        "ground_truth_fallback",
    }
    assert example["entity_retention_rate"] == 1.0
    assert example["relation_retention_rate"] == 1.0
    assert example["relations"] == annotation["relations"]
    assert example["canonical_fields"]["total_amount"]["evidence_valid"] is True
    assert example["source_targets"]["canonical_fields"] == annotation["canonical_fields"]
    assert all(example["token_loss_mask"])


def test_bio_boundaries_use_entity_identity_for_adjacent_same_class_tokens() -> None:
    assert to_bio_labels(["KEY", "KEY"], entity_ids=["key_a", "key_b"]) == [
        "B-KEY",
        "B-KEY",
    ]
    assert to_bio_labels(["KEY", "KEY"], entity_ids=["key_a", "key_a"]) == [
        "B-KEY",
        "I-KEY",
    ]


def test_training_manifest_profile_mismatch_is_refused() -> None:
    rows = [
        {"profile": "smoke", "build_id": "smoke-build", "is_usable": "true"},
        {"profile": "smoke", "build_id": "smoke-build", "is_usable": "false"},
    ]
    with pytest.raises(ValueError, match="profile"):
        validate_manifest_profile(rows, expected_profile="final")
    with pytest.raises(ValueError, match="build"):
        validate_manifest_profile(rows, expected_profile="smoke", expected_build_id="other")
    validate_manifest_profile(rows, expected_profile="smoke", expected_build_id="smoke-build")


def test_final_profile_requirements_refuse_caps_and_small_or_narrow_corpora() -> None:
    rows = [
        _row(f"fatura_{index}", dataset="fatura") for index in range(1800)
    ] + [
        _row(f"sroie_{index}", dataset="sroie") for index in range(300)
    ] + [
        _row(f"funsd_{index}", dataset="funsd") for index in range(100)
    ]
    validate_profile_requirements(rows, profile="final", requested_limit=0)
    with pytest.raises(ValueError, match="limit"):
        validate_profile_requirements(rows, profile="final", requested_limit=2000)
    with pytest.raises(ValueError, match="2,000"):
        validate_profile_requirements(rows[:1999], profile="final", requested_limit=0)
    with pytest.raises(ValueError, match="three"):
        validate_profile_requirements(rows[:2100], profile="final", requested_limit=0)
    with pytest.raises(ValueError, match="500"):
        validate_profile_requirements(rows[:499], profile="development", requested_limit=0)


def test_build_id_is_order_independent_and_bound_to_stream_policy() -> None:
    rows = [_row("page_1"), _row("page_2", dataset="sroie")]
    first = compute_dataset_build_id(
        rows,
        profile="development",
        streams=("ground_truth", "paddleocr"),
        split_manifest_sha256="split-hash",
    )
    repeated = compute_dataset_build_id(
        list(reversed(rows)),
        profile="development",
        streams=("paddleocr", "ground_truth"),
        split_manifest_sha256="split-hash",
    )
    changed = compute_dataset_build_id(
        rows,
        profile="development",
        streams=("ground_truth",),
        split_manifest_sha256="split-hash",
    )
    changed_preprocessing = compute_dataset_build_id(
        rows,
        profile="development",
        streams=("ground_truth", "paddleocr"),
        split_manifest_sha256="split-hash",
        build_provenance={"preprocessing_version": "cardinal-plus-fine-v2"},
    )
    assert first == repeated
    assert first != changed
    assert first != changed_preprocessing


def test_reused_example_must_match_freshly_derived_content(tmp_path: Path) -> None:
    expected = {
        "build_id": "final-build",
        "profile": "final",
        "token_source": "ground_truth",
        "tokens": [{"text": "current"}],
    }
    validate_reusable_example(expected, expected, output_path=tmp_path / "example.json")

    stale = {**expected, "tokens": [{"text": "stale"}]}
    with pytest.raises(ValueError, match="content-mismatched"):
        validate_reusable_example(
            stale, expected, output_path=tmp_path / "example.json"
        )


def test_ocr_variant_selection_is_bounded_balanced_and_deterministic() -> None:
    rows = [
        _row(
            f"{dataset}_{split}_{index}",
            dataset=dataset,
            split=split,
        )
        for dataset in ("fatura", "funsd", "sroie")
        for split in ("train", "dev_select", "dev_calibration", "test_in_domain")
        for index in range(5)
    ]

    selected = select_ocr_variant_rows(rows, 24)
    repeated = select_ocr_variant_rows(list(reversed(rows)), 24)

    assert len(selected) == 24
    assert {row["dataset"] for row in selected} == {"fatura", "funsd", "sroie"}
    assert {row["project_split"] for row in selected} == {
        "train",
        "dev_select",
        "dev_calibration",
        "test_in_domain",
    }
    assert [row["page_id"] for row in selected] == [
        row["page_id"] for row in repeated
    ]
    assert len(select_ocr_variant_rows(rows, 0)) == len(rows)


def test_model_data_ocr_uses_verified_orientation_pipeline(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str, bool]] = []

        def extract_path(
            self,
            image_path: Path,
            *,
            language_mode: str,
            private: bool,
        ) -> dict:
            self.calls.append((image_path, language_mode, private))
            return {
                "orientation": 343.0,
                "fine_deskew": {"correction_degrees": -17.0},
                "words": [],
            }

    image_path = tmp_path / "page.png"
    pipeline = FakePipeline()
    result = predict_model_data_ocr(pipeline, image_path, route="thai")

    assert result["orientation"] == 343.0
    assert result["fine_deskew"]["correction_degrees"] == -17.0
    assert pipeline.calls == [(image_path, "thai", False)]
    with pytest.raises(ValueError, match="route"):
        predict_model_data_ocr(pipeline, image_path, route="unknown")


def test_loader_refuses_example_that_is_not_bound_to_manifest_build(tmp_path: Path) -> None:
    example_path = tmp_path / "example.json"
    example_path.write_text(json.dumps({
        "example_id": "page_1__ground_truth",
        "build_id": "build-1",
        "profile": "smoke",
        "project_split": "train",
        "token_source": "ground_truth",
        "is_private": False,
    }), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    row = {
        "example_id": "page_1__ground_truth",
        "build_id": "build-1",
        "profile": "smoke",
        "project_split": "train",
        "token_source": "ground_truth",
        "is_private": "false",
        "is_usable": "true",
        "model_example_path": str(example_path),
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    loaded = load_model_examples(
        manifest_path,
        "train",
        expected_profile="smoke",
        expected_build_id="build-1",
        token_sources={"ground_truth"},
    )
    assert [example["example_id"] for example in loaded] == ["page_1__ground_truth"]

    stale = json.loads(example_path.read_text(encoding="utf-8"))
    stale["build_id"] = "other-build"
    example_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="build"):
        load_model_examples(
            manifest_path,
            "train",
            expected_profile="smoke",
            expected_build_id="build-1",
        )


def test_private_ground_truth_example_is_refused() -> None:
    with pytest.raises(ValueError, match="private"):
        build_ground_truth_example(
            _row("page_1", private=True),
            _annotation(),
            profile="final",
            split_group_id="group_1",
        )
