#!/usr/bin/env python3
"""Verify IE artifacts, privacy boundaries, storage, and reproducibility gates."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.schema import (  # noqa: E402
    load_output_schema,
    validate_document_result,
)
from src.ocr.environment import storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry, REQUIRED_MODEL_NAMES  # noqa: E402
from src.rotation_common import atomic_write_json, read_csv_rows, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--complete", action="store_true", help="require training/evaluation/private reports")
    parser.add_argument(
        "--run-integration",
        action="store_true",
        help="execute the synthetic GPU integration runner before validating its evidence",
    )
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if args.run_integration:
        ocr_python = cfgmod.resolve_path(cfg, "ocr_environment") / "Scripts" / "python.exe"
        completed = subprocess.run(
            [
                str(ocr_python),
                "scripts/run_integration_smoke.py",
                "--config",
                str(Path(args.config).resolve()),
                "--model-setup",
                str(Path(args.model_setup).resolve()),
                "--device",
                str(cfg.get("ocr", {}).get("device", "gpu:0")),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        check(
            "integration_runner_execution",
            completed.returncode == 0,
            {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-500:] if completed.returncode else "",
                "stderr_tail": completed.stderr[-500:] if completed.returncode else "",
            },
        )

    gate = storage_gate(cfgmod.resolve_path(cfg, "external_assets"))
    check("storage_reserve", gate["passed"], gate)
    try:
        schema = load_output_schema(cfg.get("information_extraction", {}).get("output_schema", "schemas/inference_output.schema.json"))
        check("output_json_schema", bool(schema.get("$schema")), schema.get("title"))
    except Exception as exc:
        check("output_json_schema", False, f"{type(exc).__name__}: {exc}")
    try:
        registry = ModelRegistry.from_setup(args.model_setup)
        names = {registry.require(name).name for name in REQUIRED_MODEL_NAMES}
        check("exact_ocr_model_registry", names == set(REQUIRED_MODEL_NAMES), sorted(names))
        check(
            "ocr_models_on_external_drive",
            all(str(registry.require(name).path).lower().startswith("d:\\") for name in REQUIRED_MODEL_NAMES),
        )
    except Exception as exc:
        check("exact_ocr_model_registry", False, f"{type(exc).__name__}: {exc}")

    metadata = cfgmod.resolve_path(cfg, "metadata")
    annotation_summary = _json(metadata / "information_extraction_manifest_summary.json")
    check("annotation_manifest_exists", annotation_summary is not None)
    if annotation_summary:
        check("annotation_gmail_fit_zero", annotation_summary.get("gmail_fit_rows") == 0)
        check("normalized_public_pages", int(annotation_summary.get("normalized_pages", 0)) > 0, annotation_summary.get("normalized_pages"))
    public_rows = read_csv_rows(metadata / "information_extraction_manifest.csv")
    private_public_rows = [row for row in public_rows if row.get("is_private") == "true"]
    check(
        "public_manifest_redacts_private_sources",
        bool(private_public_rows) and all(
            not row.get("image_path") and not row.get("annotation_path") and not row.get("normalized_annotation_path")
            and not row.get("duplicate_group_id") and not row.get("sha256")
            for row in private_public_rows
        ),
        {"private_rows": len(private_public_rows)},
    )
    check(
        "no_private_public_training_rows",
        all(
            row.get("is_private") != "true" or row.get("project_split") == "private_test"
            for row in public_rows
        ),
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "data/metadata/private_information_extraction_manifest.csv"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    check("private_operational_manifest_ignored", ignored.returncode == 0)

    model_summary = _json(cfgmod.resolve_path(cfg, "reports") / "information_extraction" / "model_dataset_summary.json")
    check("model_dataset_summary", model_summary is not None, None if model_summary is None else model_summary.get("usable_example_count"))
    if model_summary:
        check("model_dataset_gmail_fit_zero", model_summary.get("gmail_fit_rows") == 0)
        check("model_dataset_has_examples", int(model_summary.get("usable_example_count", 0)) > 0)

    training = _json(cfgmod.resolve_path(cfg, "reports") / "information_extraction" / "layout_model_training.json")
    if training or args.complete:
        check("layout_training_report", training is not None)
    if training:
        check("layout_checkpoint_reload", training.get("checkpoint_reload_passed") is True)
        check("relation_head_reload", training.get("relation_head", {}).get("reload_passed") is True)
        check("layout_training_gmail_fit_zero", training.get("gmail_fit_rows") == 0)
        check("layout_checkpoint_on_d", str(training.get("checkpoint", "")).lower().startswith("d:\\"))

    ocr_verification = _json(cfgmod.resolve_path(cfg, "reports") / "ocr" / "model_verification.json")
    check("ocr_model_verification", bool(ocr_verification and ocr_verification.get("passed") is True))

    evaluation = _json(cfgmod.resolve_path(cfg, "reports") / "model_evaluation" / "smoke_evaluation.json")
    private_aggregate = _json(cfgmod.resolve_path(cfg, "reports") / "model_evaluation" / "private_gmail_aggregate.json")
    integration = _json(cfgmod.resolve_path(cfg, "reports") / "information_extraction" / "integration_smoke.json")
    if args.complete:
        check("public_smoke_evaluation", bool(evaluation and evaluation.get("successful_runs", 0) > 0))
        check("private_aggregate_evaluation", bool(private_aggregate and private_aggregate.get("successful_pages", 0) > 0))
        evidence_passed, evidence_detail = _validate_integration_evidence(
            integration,
            cfg=cfg,
            config_path=Path(args.config).resolve(),
            model_setup_path=Path(args.model_setup).resolve(),
        )
        check("integration_evidence_cryptographically_bound", evidence_passed, evidence_detail)
        cases = {item.get("case"): item for item in (integration or {}).get("cases", [])}
        required_cases = {
            "unknown_upright_image", "unknown_45_degree_image",
            "mixed_language_multipage_pdf", "thai_auto_with_metadata_hint",
        }
        check(
            "integration_image_pdf_rotation_thai_unknown",
            bool(
                integration and integration.get("status") == "passed"
                and required_cases <= set(cases)
                and all(
                    cases[name].get("assertions")
                    and all(cases[name]["assertions"].values())
                    for name in required_cases
                )
                and cases["mixed_language_multipage_pdf"].get("pages") == 2
                and cases["thai_auto_with_metadata_hint"].get("assertions", {}).get("thai_unicode_present") is True
            ),
        )
        check(
            "natural_unseen_dataset_evaluation",
            bool(
                evaluation
                and evaluation.get("unseen_document_protocol", {}).get("status")
                == "executed_natural_dataset_holdout"
                and evaluation.get("unseen_document_protocol", {}).get("fit_rows_for_evaluated_datasets") == 0
            ),
        )
    if private_aggregate:
        check(
            "private_report_aggregate_only",
            private_aggregate.get("contains_filenames") is False
            and private_aggregate.get("contains_ocr_text") is False
            and private_aggregate.get("contains_images") is False
            and private_aggregate.get("contains_per_document_predictions") is False,
        )
        check("private_report_gmail_fit_zero", private_aggregate.get("gmail_fit_rows") == 0)

    candidate_files = _git_candidate_files()
    oversized = [str(path.relative_to(PROJECT_ROOT)) for path in candidate_files if path.is_file() and path.stat().st_size > 50 * 1024**2]
    check("no_unignored_file_over_50_mib", not oversized, oversized)
    secret_hits = _secret_scan(candidate_files)
    check("no_obvious_secrets_in_git_candidates", not secret_hits, {"hit_count": len(secret_hits)})
    private_name_hit_count = _private_name_scan(candidate_files, metadata)
    check("no_private_filenames_in_git_candidates", private_name_hit_count == 0, {"hit_count": private_name_hit_count})

    failed = [item for item in checks if not item["passed"]]
    report = {
        "schema_version": "1.0",
        "status": "passed" if not failed else "failed",
        "complete_mode": args.complete,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "checks": checks,
    }
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "verification" / "information_extraction_verification.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validate_integration_evidence(
    report: dict[str, Any] | None,
    *,
    cfg: dict[str, Any],
    config_path: Path,
    model_setup_path: Path,
) -> tuple[bool, dict[str, Any]]:
    """Re-hash runner inputs/outputs and independently inspect synthetic results."""
    errors: list[str] = []
    if not report:
        return False, {"errors": ["integration report is missing or invalid JSON"]}
    if report.get("generated_by") != "scripts/run_integration_smoke.py":
        errors.append("unexpected or missing generated_by")
    if report.get("private_inputs_used") is not False:
        errors.append("private_inputs_used must be false")
    if report.get("status") != "passed":
        errors.append("integration report status is not passed")

    schema_path = (PROJECT_ROOT / str(cfg["information_extraction"]["output_schema"])).resolve()
    expected_sources = {
        "runner": PROJECT_ROOT / "scripts" / "run_integration_smoke.py",
        "integration_verifier": PROJECT_ROOT / "scripts" / "verify_information_extraction.py",
        "config": config_path,
        "output_schema": schema_path,
        "model_setup": model_setup_path,
        "layout_training_report": PROJECT_ROOT
        / "reports"
        / "information_extraction"
        / "layout_model_training.json",
        "document_pipeline": PROJECT_ROOT / "src" / "inference" / "document_pipeline.py",
        "document_io": PROJECT_ROOT / "src" / "inference" / "document_io.py",
        "ocr_pipeline": PROJECT_ROOT / "src" / "ocr" / "pipeline.py",
        "ocr_adapter": PROJECT_ROOT / "src" / "ocr" / "paddleocr_adapter.py",
        "language_router": PROJECT_ROOT / "src" / "ocr" / "language_router.py",
        "ocr_scoring": PROJECT_ROOT / "src" / "ocr" / "scoring.py",
    }
    source_hashes = report.get("source_hashes") or {}
    for name, path in expected_sources.items():
        record = source_hashes.get(name) or {}
        if not path.is_file():
            errors.append(f"source is missing: {name}")
            continue
        expected_relative = str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
        if record.get("path") != expected_relative:
            errors.append(f"source path mismatch: {name}")
        if record.get("sha256") != sha256_file(path):
            errors.append(f"source hash mismatch: {name}")

    external_root = cfgmod.resolve_path(cfg, "external_assets").resolve()
    checked_artifacts = 0

    def validate_artifact(record: Any, label: str) -> Path | None:
        nonlocal checked_artifacts
        if not isinstance(record, dict) or not record.get("path"):
            errors.append(f"artifact record missing: {label}")
            return None
        path = Path(str(record["path"])).resolve()
        if path != external_root and external_root not in path.parents:
            errors.append(f"artifact escaped external root: {label}")
            return None
        if not path.is_file():
            errors.append(f"artifact file missing: {label}")
            return None
        if int(record.get("size_bytes", -1)) != path.stat().st_size:
            errors.append(f"artifact size mismatch: {label}")
        if record.get("sha256") != sha256_file(path):
            errors.append(f"artifact hash mismatch: {label}")
        checked_artifacts += 1
        return path

    training = _json(expected_sources["layout_training_report"])
    checkpoint_root = Path(str((training or {}).get("checkpoint", ""))).resolve()
    checkpoint_records = list(report.get("checkpoint_artifacts") or [])
    expected_checkpoint_names = {"model.safetensors", "relation_head.pt", "training_state.json"}
    if {Path(str(item.get("path", ""))).name for item in checkpoint_records} != expected_checkpoint_names:
        errors.append("checkpoint artifact set mismatch")
    for record in checkpoint_records:
        path = validate_artifact(record, f"checkpoint:{Path(str(record.get('path', ''))).name}")
        if path is not None and path.parent != checkpoint_root:
            errors.append(f"checkpoint path mismatch: {path.name}")

    required_cases = {
        "unknown_upright_image",
        "unknown_45_degree_image",
        "mixed_language_multipage_pdf",
        "thai_auto_with_metadata_hint",
    }
    fixtures = report.get("fixture_artifacts") or {}
    outputs = report.get("output_artifacts") or {}
    if set(fixtures) != required_cases:
        errors.append("fixture artifact set mismatch")
    if set(outputs) != required_cases:
        errors.append("output artifact set mismatch")
    for case in sorted(required_cases):
        validate_artifact(fixtures.get(case), f"fixture:{case}")
        output_path = validate_artifact(outputs.get(case), f"output:{case}")
        if output_path is None:
            continue
        payload = _json(output_path)
        if payload is None:
            errors.append(f"output is invalid JSON: {case}")
            continue
        try:
            validate_document_result(payload, schema_path)
        except Exception as exc:
            errors.append(f"output schema invalid for {case}: {type(exc).__name__}")
            continue
        errors.extend(_integration_semantic_errors(case, payload))

    case_summaries = {
        item.get("case"): item for item in report.get("cases") or [] if isinstance(item, dict)
    }
    if set(case_summaries) != required_cases:
        errors.append("case summary set mismatch")
    for case in required_cases & set(case_summaries):
        assertions = case_summaries[case].get("assertions")
        if not isinstance(assertions, dict) or not assertions or not all(assertions.values()):
            errors.append(f"recorded assertion failed: {case}")
    return not errors, {"checked_artifacts": checked_artifacts, "errors": errors}


def _integration_semantic_errors(case: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pages = list(payload.get("pages") or [])
    routes = [page.get("ocr", {}).get("language_route") for page in pages]
    recognizers = [page.get("ocr", {}).get("recognizer_model") for page in pages]
    if payload.get("rotation_display", {}).get("purpose") != "display_only":
        errors.append(f"K-Means purpose is not display_only: {case}")
    if case in {"unknown_upright_image", "unknown_45_degree_image"}:
        if payload.get("source_type") != "image" or len(pages) != 1:
            errors.append(f"invalid image/page shape: {case}")
        if payload.get("document_type", {}).get("label") != "unknown":
            errors.append(f"document type is not unknown: {case}")
        if not pages or not pages[0].get("ocr", {}).get("words"):
            errors.append(f"empty OCR: {case}")
        if not pages or not pages[0].get("entities") or not pages[0].get("key_value_pairs"):
            errors.append(f"empty generic IE output: {case}")
    elif case == "mixed_language_multipage_pdf":
        if payload.get("source_type") != "pdf" or len(pages) != 2:
            errors.append("mixed-language fixture is not a two-page PDF result")
        if routes != ["general", "thai"]:
            errors.append("mixed-language routes are not general then Thai")
        if recognizers != ["PP-OCRv6_medium_rec", "th_PP-OCRv5_mobile_rec"]:
            errors.append("mixed-language recognizers are not the exact requested models")
        if len(pages) != 2 or not all(page.get("ocr", {}).get("words") for page in pages):
            errors.append("mixed-language OCR output is empty")
    elif case == "thai_auto_with_metadata_hint":
        text = "\n".join(str(page.get("full_text", "")) for page in pages)
        if routes != ["thai"] or recognizers != ["th_PP-OCRv5_mobile_rec"]:
            errors.append("Thai hint did not select the exact Thai recognizer")
        if not any("\u0e00" <= character <= "\u0e7f" for character in text):
            errors.append("Thai output contains no Thai Unicode")
    return errors


def _git_candidate_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT, capture_output=True, check=True,
    )
    return [PROJECT_ROOT / value.decode("utf-8") for value in completed.stdout.split(b"\0") if value]


def _secret_scan(paths: list[Path]) -> list[str]:
    pattern = re.compile(
        r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-/.+=]{16,}"
    )
    hits = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 10 * 1024**2:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    return hits


def _private_name_scan(paths: list[Path], metadata: Path) -> int:
    inventory = metadata / "private_file_inventory.csv"
    if not inventory.is_file():
        return 0
    private_names = {
        Path(row.get("relative_path") or row.get("path") or row.get("filename") or "").name.casefold()
        for row in read_csv_rows(inventory)
    }
    private_names.discard("")
    hits = 0
    for path in paths:
        if not path.is_file() or path.stat().st_size > 20 * 1024**2:
            continue
        try:
            folded = path.read_text(encoding="utf-8").casefold()
        except (UnicodeDecodeError, OSError):
            continue
        if any(name in folded for name in private_names):
            hits += 1
    return hits


if __name__ == "__main__":
    raise SystemExit(main())
