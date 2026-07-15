#!/usr/bin/env python3
"""Run reproducible synthetic image, rotation, Thai, and multi-page integration smoke tests."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from src import config as cfgmod  # noqa: E402
from src.information_extraction.schema import validate_document_result  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment  # noqa: E402
from src.rotation_common import atomic_write_json, sha256_file  # noqa: E402

RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_integration_smoke.py"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "information_extraction" / "integration_smoke.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"),
    )
    parser.add_argument("--layout-checkpoint")
    parser.add_argument("--artifact-root")
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    model_setup_path = Path(args.model_setup).resolve()
    cfg = cfgmod.load_config(config_path)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    artifact_root = (
        Path(args.artifact_root).resolve()
        if args.artifact_root
        else (cfgmod.resolve_path(cfg, "generated_documents") / "integration_smoke").resolve()
    )
    external_root = cfgmod.resolve_path(cfg, "external_assets").resolve()
    if artifact_root != external_root and external_root not in artifact_root.parents:
        parser.error(f"--artifact-root must remain under external assets: {external_root}")

    fixture_root = artifact_root / "fixtures"
    output_root = artifact_root / "outputs"
    fixture_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    fixtures = _write_fixtures(fixture_root)

    checkpoint = (
        Path(args.layout_checkpoint).resolve()
        if args.layout_checkpoint
        else (cfgmod.resolve_path(cfg, "ie_checkpoints") / "layoutxlm" / "smoke").resolve()
    )
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=model_setup_path,
        layout_checkpoint=checkpoint,
        enable_kmeans_display=True,
    )
    case_specs = (
        ("unknown_upright_image", fixtures["unknown_upright_image"], "auto", None),
        ("unknown_45_degree_image", fixtures["unknown_45_degree_image"], "auto", None),
        ("mixed_language_multipage_pdf", fixtures["mixed_language_multipage_pdf"], "auto", None),
        ("thai_auto_with_metadata_hint", fixtures["thai_auto_with_metadata_hint"], "auto", "th"),
    )
    cases: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    try:
        for case, path, language, language_hint in case_specs:
            result = pipeline.extract_path(
                path,
                language=language,
                language_hint=language_hint,
                pdf_dpi=150,
            )
            validate_document_result(result)
            assertions = _case_assertions(case, result)
            output_path = output_root / f"{case}.json"
            atomic_write_json(output_path, result)
            outputs[case] = _artifact_record(output_path)
            cases.append(_case_summary(case, result, assertions))
    finally:
        pipeline.close()

    training_report = PROJECT_ROOT / "reports" / "information_extraction" / "layout_model_training.json"
    schema_path = (PROJECT_ROOT / str(cfg["information_extraction"]["output_schema"])).resolve()
    checkpoint_files = [
        checkpoint / "model.safetensors",
        checkpoint / "relation_head.pt",
        checkpoint / "training_state.json",
    ]
    required_sources = {
        "runner": RUNNER_PATH,
        "integration_verifier": PROJECT_ROOT / "scripts" / "verify_information_extraction.py",
        "config": config_path,
        "output_schema": schema_path,
        "model_setup": model_setup_path,
        "layout_training_report": training_report,
        "document_pipeline": PROJECT_ROOT / "src" / "inference" / "document_pipeline.py",
        "document_io": PROJECT_ROOT / "src" / "inference" / "document_io.py",
        "ocr_pipeline": PROJECT_ROOT / "src" / "ocr" / "pipeline.py",
        "ocr_adapter": PROJECT_ROOT / "src" / "ocr" / "paddleocr_adapter.py",
        "language_router": PROJECT_ROOT / "src" / "ocr" / "language_router.py",
        "ocr_scoring": PROJECT_ROOT / "src" / "ocr" / "scoring.py",
    }
    missing = [str(path) for path in (*required_sources.values(), *checkpoint_files) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"integration provenance inputs are missing: {missing}")

    passed = all(all(case["assertions"].values()) for case in cases)
    report = {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "generated_by": "scripts/run_integration_smoke.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [
            str(Path(sys.executable).resolve()),
            "scripts/run_integration_smoke.py",
            "--device",
            args.device,
        ],
        "source_revision": _git_revision(),
        "device": args.device,
        "private_inputs_used": False,
        "source_hashes": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for name, path in required_sources.items()
        },
        "checkpoint_artifacts": [_artifact_record(path) for path in checkpoint_files],
        "fixture_artifacts": {
            name: _artifact_record(path) for name, path in fixtures.items()
        },
        "output_artifacts": outputs,
        "cases": cases,
        "limitations": [
            "Synthetic fixtures prove integration paths, not real-world accuracy.",
            "The 45-degree case requires useful output; exact correction-angle quality is evaluated separately.",
        ],
    }
    atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


def _write_fixtures(root: Path) -> dict[str, Path]:
    upright = _text_image(
        (
            "NORTHSTAR LAB",
            "REFERENCE NO: AB-123",
            "CUSTOMER: TEST USER",
            "AMOUNT DUE: 123.45 USD",
        ),
        font_size=58,
    )
    thai = _text_image(
        ("ใบเสร็จรับเงิน", "ยอดรวม: 123.45 บาท"),
        font_size=64,
    )
    rotated = upright.rotate(45, expand=True, fillcolor="white")
    upright_path = root / "unknown_upright.png"
    rotated_path = root / "unknown_45_degree.png"
    thai_path = root / "thai_receipt.png"
    pdf_path = root / "mixed_language_two_page.pdf"
    upright.save(upright_path, format="PNG", optimize=False)
    rotated.save(rotated_path, format="PNG", optimize=False)
    thai.save(thai_path, format="PNG", optimize=False)
    upright.save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=[thai],
        resolution=150.0,
    )
    return {
        "unknown_upright_image": upright_path,
        "unknown_45_degree_image": rotated_path,
        "mixed_language_multipage_pdf": pdf_path,
        "thai_auto_with_metadata_hint": thai_path,
    }


def _text_image(lines: tuple[str, ...], *, font_size: int) -> Image.Image:
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    y = 100
    for line in lines:
        draw.text((100, y), line, fill="black", font=font)
        y += font_size + 70
    return image


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _case_assertions(case: str, result: dict[str, Any]) -> dict[str, bool]:
    pages = list(result.get("pages") or [])
    word_counts = [len(page.get("ocr", {}).get("words") or []) for page in pages]
    entity_counts = [len(page.get("entities") or []) for page in pages]
    pair_counts = [len(page.get("key_value_pairs") or []) for page in pages]
    routes = [str(page.get("ocr", {}).get("language_route")) for page in pages]
    recognizers = [str(page.get("ocr", {}).get("recognizer_model")) for page in pages]
    common = {
        "schema_valid": True,
        "display_only_kmeans": result.get("rotation_display", {}).get("purpose") == "display_only",
    }
    if case in {"unknown_upright_image", "unknown_45_degree_image"}:
        return {
            **common,
            "image_input": result.get("source_type") == "image",
            "single_page": len(pages) == 1,
            "unknown_document_type": result.get("document_type", {}).get("label") == "unknown",
            "nonempty_ocr": bool(word_counts and word_counts[0] > 0),
            "nonempty_entities": bool(entity_counts and entity_counts[0] > 0),
            "nonempty_key_values": bool(pair_counts and pair_counts[0] > 0),
        }
    if case == "mixed_language_multipage_pdf":
        return {
            **common,
            "pdf_input": result.get("source_type") == "pdf",
            "two_pages": len(pages) == 2,
            "general_then_thai_routes": routes == ["general", "thai"],
            "exact_recognizers": recognizers
            == ["PP-OCRv6_medium_rec", "th_PP-OCRv5_mobile_rec"],
            "both_pages_nonempty": len(word_counts) == 2 and all(count > 0 for count in word_counts),
        }
    if case == "thai_auto_with_metadata_hint":
        text = "\n".join(str(page.get("full_text", "")) for page in pages)
        return {
            **common,
            "image_input": result.get("source_type") == "image",
            "single_page": len(pages) == 1,
            "thai_route": routes == ["thai"],
            "exact_thai_recognizer": recognizers == ["th_PP-OCRv5_mobile_rec"],
            "nonempty_ocr": bool(word_counts and word_counts[0] > 0),
            "thai_unicode_present": any("\u0e00" <= char <= "\u0e7f" for char in text),
        }
    raise ValueError(f"unknown integration case: {case}")


def _case_summary(
    case: str, result: dict[str, Any], assertions: dict[str, bool]
) -> dict[str, Any]:
    pages = list(result["pages"])
    return {
        "case": case,
        "source_type": result["source_type"],
        "pages": len(pages),
        "document_type": result["document_type"]["label"],
        "routes": [page["ocr"]["language_route"] for page in pages],
        "recognizers": [page["ocr"]["recognizer_model"] for page in pages],
        "selected_orientations": [page["selected_ocr_orientation"] for page in pages],
        "ocr_words": [len(page["ocr"]["words"]) for page in pages],
        "entities": [len(page["entities"]) for page in pages],
        "key_value_pairs": [len(page["key_value_pairs"]) for page in pages],
        "assertions": assertions,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
