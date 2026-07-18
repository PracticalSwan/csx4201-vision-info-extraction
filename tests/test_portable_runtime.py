from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from src.ocr.model_registry import REQUIRED_MODEL_NAMES, ModelRegistry
from src.portable.api import build_command
from src.portable.results import field_rows
from src.portable.review import (
    ReviewPayloadError,
    list_reviewable_results,
    prepare_review_payload,
)
from src.portable.runtime import RuntimeSettings, _resolve


def _write_portable_config(root: Path) -> None:
    payload = {
        "paths": {
            "project_root": ".",
            "external_assets": "assets",
            "ocr_environment": ".runtime/ocr",
            "layout_environment": ".runtime/layout",
            "layout_python": ".runtime/layout/Scripts/python.exe",
        },
        "ocr": {"device": "cpu"},
        "layout_model": {
            "inference_checkpoint": "assets/checkpoints/layoutxlm_multitask/final"
        },
    }
    (root / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _touch_runtime(root: Path) -> None:
    for path in (
        root / ".runtime/ocr/Scripts/python.exe",
        root / ".runtime/layout/Scripts/python.exe",
        root / "reports/ocr/model_setup.json",
        root / "assets/checkpoints/layoutxlm_multitask/final/model.safetensors",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "config": "config.yaml",
                "ocr_python": ".runtime/ocr/Scripts/python.exe",
                "layout_python": ".runtime/layout/Scripts/python.exe",
                "model_setup": "reports/ocr/model_setup.json",
                "layout_checkpoint": "assets/checkpoints/layoutxlm_multitask/final",
                "asset_root": "assets",
                "output_root": "outputs",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )


def test_runtime_settings_resolve_relative_bundle_and_ignore_stale_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_portable_config(tmp_path)
    _touch_runtime(tmp_path)
    (tmp_path / "runtime.local.json").write_text(
        json.dumps(
            {
                "ocr_python": "Z:/missing/ocr/python.exe",
                "layout_python": "Z:/missing/layout/python.exe",
                "device": "gpu:0",
            }
        ),
        encoding="utf-8-sig",
    )
    for name in (
        "OCR_MODEL_HOME",
        "OCR_MODEL_CONFIG",
        "OCR_MODEL_OCR_PYTHON",
        "OCR_MODEL_LAYOUT_PYTHON",
        "OCR_MODEL_DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.load(tmp_path)

    assert settings.ready
    assert settings.home == tmp_path.resolve()
    assert settings.ocr_python == (
        tmp_path / ".runtime/ocr/Scripts/python.exe"
    ).resolve()
    assert settings.device == "cpu"
    assert settings.environment()["OCR_MODEL_HOME"] == str(tmp_path.resolve())
    assert "OPENAI_API_KEY" not in {
        key for key in settings.environment() if key.startswith("OCR_MODEL_")
    }


def test_runtime_path_normalization_does_not_dereference_venv_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / ".runtime/ocr/bin/python"
    expected = Path(os.path.abspath(candidate))

    def fail_if_resolved(_path: Path) -> Path:
        raise AssertionError("virtual-environment interpreter symlink was dereferenced")

    monkeypatch.setattr(Path, "resolve", fail_if_resolved)

    assert _resolve(tmp_path, ".runtime/ocr/bin/python") == expected


def test_build_command_uses_existing_worker_and_no_openai_argument(tmp_path: Path) -> None:
    _write_portable_config(tmp_path)
    _touch_runtime(tmp_path)
    settings = RuntimeSettings.load(tmp_path)
    source = tmp_path / "sample.png"
    source.write_bytes(b"png")
    command = build_command(
        settings, source, tmp_path / "outputs/run", device="cpu"
    )
    joined = " ".join(command)
    assert str(tmp_path / "scripts" / "extract_document.py") in joined
    assert "--model-checkpoint" in command
    assert "--save-visualization" in command
    assert "api" not in joined.casefold()
    assert "key" not in joined.casefold()


def test_registry_resolves_portable_model_paths_relative_to_manifest(
    tmp_path: Path,
) -> None:
    setup_root = tmp_path / "reports" / "ocr"
    setup_root.mkdir(parents=True)
    models = {}
    for name in REQUIRED_MODEL_NAMES:
        directory = tmp_path / "assets" / "ocr_models" / name
        directory.mkdir(parents=True)
        artifact = directory / "inference.json"
        artifact.write_text(name, encoding="utf-8")
        models[name] = {
            "requested_name": name,
            "resolved_name": name,
            "resolved_path": f"../../assets/ocr_models/{name}",
            "role": "detector" if name.endswith("det") else "recognizer",
            "language": "thai" if name.startswith("th_") else "general",
            "files": [
                {
                    "path": artifact.name,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
    setup = setup_root / "model_setup.json"
    setup.write_text(json.dumps({"models": models}), encoding="utf-8")

    registry = ModelRegistry.from_setup(setup)

    assert registry.require(REQUIRED_MODEL_NAMES[0]).path == (
        tmp_path / "assets" / "ocr_models" / REQUIRED_MODEL_NAMES[0]
    ).resolve()


def _result(*, private: bool = False) -> dict:
    return {
        "document_id": "secret-source-name",
        "document_type": {"label": "invoice", "confidence": 0.9},
        "language": {"selected_route": "general"},
        "fields": {
            "total_amount": {
                "value": "123.45",
                "confidence": 0.72,
                "method": "layout",
                "page_number": 1,
                "validation_status": "unverified",
            },
            "empty_field": {"value": None, "confidence": None},
            "email": {
                "value": "hidden@example.test",
                "confidence": 0.8,
            },
        },
        "pages": [{"page_number": 1, "full_text": "TOTAL 123.45 hidden text"}],
        "processing": {"private_output": private},
    }


def test_review_catalog_and_payload_are_opaque_consent_gated_and_bounded(
    tmp_path: Path,
) -> None:
    public_path = tmp_path / "public-run" / "document_result.json"
    public_path.parent.mkdir()
    public_path.write_text(json.dumps(_result()), encoding="utf-8")
    private_path = tmp_path / "private" / "document_result.json"
    private_path.parent.mkdir()
    private_path.write_text(json.dumps(_result(private=True)), encoding="utf-8")

    catalog = list_reviewable_results(tmp_path)

    assert len(catalog["results"]) == 1
    metadata = catalog["results"][0]
    assert metadata["document_id"].startswith("doc_")
    assert metadata["field_names"] == ["email", "total_amount"]
    serialized_catalog = json.dumps(catalog)
    assert "secret-source-name" not in serialized_catalog
    assert "hidden@example.test" not in serialized_catalog
    assert str(tmp_path) not in serialized_catalog

    confirmation = prepare_review_payload(
        tmp_path,
        metadata["document_id"],
        confirmed_cloud_review=False,
        selected_fields=["total_amount"],
    )
    assert confirmation["status"] == "confirmation_required"
    assert "123.45" not in json.dumps(confirmation)

    payload = prepare_review_payload(
        tmp_path,
        metadata["document_id"],
        confirmed_cloud_review=True,
        selected_fields=["total_amount"],
        include_ocr_text=True,
        max_text_chars=10,
    )
    assert payload["fields"] == {
        "total_amount": {
            "value": "123.45",
            "confidence": 0.72,
            "method": "layout",
            "validation_status": "unverified",
            "page_number": 1,
        }
    }
    assert len(payload["ocr_text"]) == 10
    assert "hidden@example.test" not in json.dumps(payload)
    assert payload["review_contract"]["api_key_used"] is False

    with pytest.raises(ReviewPayloadError):
        prepare_review_payload(
            tmp_path,
            metadata["document_id"],
            confirmed_cloud_review=True,
            selected_fields=None,
        )


def test_field_rows_hide_unextracted_schema_placeholders() -> None:
    rows = field_rows(_result())
    assert [row[0] for row in rows] == ["email", "total_amount"]
