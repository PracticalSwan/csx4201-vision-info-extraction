from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_information_extraction import (
    _integration_semantic_errors,
    _private_name_scan,
    _secret_scan,
    _split_leakage_summary,
    _valid_locked_unseen_evaluation,
)


def test_integration_provenance_covers_the_learned_worker_call_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sections = []
    for relative, marker in (
        ("scripts/run_integration_smoke.py", "required_sources = {"),
        ("scripts/verify_information_extraction.py", "expected_sources = {"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        sections.append(text.split(marker, 1)[1].split("\n    }", 1)[0])

    required = {
        "entity_worker_client": "entity_worker_client.py",
        "layout_entity_worker": "layout_entity_worker.py",
        "multitask_inference": "multitask_inference.py",
        "layoutxlm_model": "layoutxlm_model.py",
    }
    for section in sections:
        for key, filename in required.items():
            assert f'"{key}"' in section
            assert f'"{filename}"' in section


def test_integration_verifier_accepts_learned_document_type_for_generic_fixture() -> None:
    payload = {
        "source_type": "image",
        "document_type": {"label": "invoice", "confidence": 0.75},
        "rotation_display": {"purpose": "display_only"},
        "pages": [{
            "ocr": {"words": [{"text": "TOTAL"}]},
            "entities": [{"id": "entity-1"}],
            "key_value_pairs": [{"id": "relation-1"}],
        }],
    }

    assert _integration_semantic_errors("unknown_upright_image", payload) == []


def test_split_leakage_summary_catches_document_and_hash_crossovers() -> None:
    rows = [
        {
            "dataset": "funsd", "document_id": "doc-1", "page_id": "p1",
            "split_group_id": "group-1", "duplicate_group_id": "dup-1",
            "sha256": "abc", "project_split": "train",
        },
        {
            "dataset": "funsd", "document_id": "doc-1", "page_id": "p2",
            "split_group_id": "group-2", "duplicate_group_id": "dup-2",
            "sha256": "abc", "project_split": "test_in_domain",
        },
    ]

    summary = _split_leakage_summary(rows)

    assert summary["violation_count"] == 2
    assert any(value.startswith("document:") for value in summary["violation_sample"])
    assert any(value.startswith("sha256:") for value in summary["violation_sample"])


def test_streaming_publication_scans_find_private_name_and_secret(tmp_path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "private_file_inventory.csv").write_text(
        "relative_path\nprivate/source-sensitive.pdf\n", encoding="utf-8"
    )
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(
        "source-sensitive.pdf\n"
        + "api_" + "key='" + "abcdefghijklmnop" + "1234'\n",
        encoding="utf-8",
    )

    assert _private_name_scan([candidate], metadata) == 1
    assert _secret_scan([candidate])


def test_private_name_scan_supports_live_inventory_columns(tmp_path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "private_file_inventory.csv").write_text(
        "file_id,original_filename,current_relative_path\n"
        "private_1,private-source.pdf,data/raw/private/gmail/private-source.pdf\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("private-source.pdf\n", encoding="utf-8")

    assert _private_name_scan([candidate], metadata) == 1


def test_private_name_scan_fails_closed_without_inventory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="inventory"):
        _private_name_scan([], tmp_path)

    (tmp_path / "private_file_inventory.csv").write_text(
        "file_id,original_filename\nprivate_1,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no usable filename"):
        _private_name_scan([], tmp_path)


def test_unseen_evaluation_requires_locked_100_page_zero_failure_run() -> None:
    checkpoint_hash = "a" * 64
    report = {
        "dataset": "coru",
        "split": "unseen_domain_test",
        "public_only": True,
        "private_page_count": 0,
        "sample_pages": 100,
        "successful_pages": 100,
        "failed_pages": 0,
        "checkpoint_model_sha256": checkpoint_hash,
    }

    assert _valid_locked_unseen_evaluation(
        report, checkpoint_model_sha256=checkpoint_hash
    )
    report["successful_pages"] = 99
    report["failed_pages"] = 1
    assert not _valid_locked_unseen_evaluation(
        report, checkpoint_model_sha256=checkpoint_hash
    )
