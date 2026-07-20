#!/usr/bin/env python3
"""Build the private-data-free, relocatable OCR_Model distribution."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path("D:/OCR_Model")
DEFAULT_ASSET_ROOT = Path("D:/CSX4201/vision-info-extraction-assets")
EXPECTED_LAYOUT_SHA256 = (
    "34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189"
)
OCR_MODEL_NAMES = (
    "PP-OCRv6_medium_det",
    "PP-OCRv6_medium_rec",
    "th_PP-OCRv5_mobile_rec",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def prepare_target(target: Path, *, force: bool) -> None:
    resolved = target.resolve()
    if resolved.name != "OCR_Model":
        raise ValueError("portable target directory must be named exactly OCR_Model")
    if resolved.exists():
        if not force:
            raise FileExistsError(
                f"target exists: {resolved}; pass --force to rebuild it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def copy_application(target: Path) -> None:
    for directory in ("src", "schemas", ".agents"):
        copy_tree(PROJECT_ROOT / directory, target / directory)
    copy_tree(PROJECT_ROOT / "models" / "kmeans_rotation", target / "models" / "kmeans_rotation")
    scripts_target = target / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=True)
    for name in (
        "extract_document.py",
        "layout_entity_worker.py",
        "setup_portable_windows.ps1",
    ):
        copy_file(PROJECT_ROOT / "scripts" / name, scripts_target / name)
    for name in (
        "LICENSE",
        "CONTRIBUTING.md",
        "app.py",
        "doctor.py",
        "mcp_server.py",
        "run_ocr.py",
        "requirements-app.txt",
        "requirements-ie.txt",
        "requirements-layout.txt",
        "requirements-ocr.txt",
        "setup_windows.ps1",
        "setup_windows.bat",
        "launch_windows.bat",
        "run_cli.bat",
        "install_codex_integration.ps1",
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "launch_macos.command",
        "install_codex_integration_macos.command",
    ):
        copy_file(PROJECT_ROOT / name, target / name)
    (target / "extract_document.py").write_text(
        "#!/usr/bin/env python3\n"
        '"""Run the complete local OCR + information-extraction model."""\n'
        "from src.portable.cli import main\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    for name in (
        "PORTABLE_USAGE.md",
        "CODEX_INTEGRATION.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        copy_file(PROJECT_ROOT / "docs" / name, target / "docs" / name)
    copy_tree(PROJECT_ROOT / "docs" / "devpost", target / "docs" / "devpost")
    copy_file(PROJECT_ROOT / "docs" / "PORTABLE_USAGE.md", target / "README.md")


def portable_config(target: Path) -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    paths.update(
        {
            "project_root": ".",
            "sroie": "data/not-included/public/sroie",
            "funsd": "data/not-included/public/funsd",
            "fatura": "data/not-included/public/fatura",
            "coru": "data/not-included/public/coru",
            "gmail_receipts": "data/not-included/private/gmail/receipts",
            "gmail_invoices": "data/not-included/private/gmail/invoices",
            "gmail_legal_financial": "data/not-included/private/gmail/legal_financial_docs",
            "gmail_unclassified": "data/not-included/private/gmail/unclassified",
            "metadata": "data/not-included/metadata",
            "processed": "data/not-included/processed",
            "page_images": "data/not-included/page_images",
            "private_page_images": "data/not-included/private/page_images",
            "rotated_images": "data/not-included/rotated_images",
            "features": "data/not-included/features",
            "splits": "data/not-included/splits",
            "rotation_models": "models/kmeans_rotation",
            "reports": "reports",
            "external_assets": "assets",
            "ocr_environment": ".runtime/ocr",
            "layout_environment": ".runtime/layout",
            "layout_python": ".runtime/layout/Scripts/python.exe",
            "paddle_cache": "assets/cache/paddlex",
            "huggingface_cache": "assets/cache/huggingface",
            "ocr_models": "assets/ocr_models",
            "layout_models": "assets/cache/layoutxlm",
            "ie_checkpoints": "assets/checkpoints",
            "ocr_cache": "assets/cache/ocr",
            "model_datasets": "data/not-included/model_datasets",
            "generated_documents": "outputs",
            "private_outputs": "outputs/private",
        }
    )
    cfg["ocr"]["device"] = "cpu"
    cfg["ocr"]["paddle_cache_home"] = "assets/cache/paddlex"
    cfg["ocr"]["detector"]["path"] = "assets/ocr_models/PP-OCRv6_medium_det"
    cfg["ocr"]["general_recognizer"]["path"] = "assets/ocr_models/PP-OCRv6_medium_rec"
    cfg["ocr"]["thai_recognizer"]["path"] = "assets/ocr_models/th_PP-OCRv5_mobile_rec"
    cfg["layout_model"]["inference_checkpoint"] = (
        "assets/checkpoints/layoutxlm_multitask/final"
    )
    (target / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(
        target / "runtime.json",
        {
            "schema_version": "1.0",
            "config": "config.yaml",
            "ocr_python": ".runtime/ocr/Scripts/python.exe",
            "layout_python": ".runtime/layout/Scripts/python.exe",
            "model_setup": "reports/ocr/model_setup.json",
            "layout_checkpoint": "assets/checkpoints/layoutxlm_multitask/final",
            "asset_root": "assets",
            "output_root": "outputs",
            "device": "cpu",
            "uses_openai_api": False,
        },
    )


def copy_models(target: Path, asset_root: Path) -> list[dict[str, Any]]:
    checkpoint_source = asset_root / "checkpoints" / "layoutxlm_multitask" / "final"
    model_source = checkpoint_source / "model.safetensors"
    actual_hash = sha256_file(model_source)
    if actual_hash != EXPECTED_LAYOUT_SHA256:
        raise ValueError(
            f"final checkpoint hash mismatch: {actual_hash} != {EXPECTED_LAYOUT_SHA256}"
        )
    checkpoint_target = target / "assets" / "checkpoints" / "layoutxlm_multitask" / "final"
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    for source in sorted(checkpoint_source.iterdir()):
        if source.is_file():
            if source.name == "training_state.json":
                state = json.loads(source.read_text(encoding="utf-8"))
                state["manifest_path"] = "not included; public training manifest"
                write_json(checkpoint_target / source.name, state)
            else:
                copy_file(source, checkpoint_target / source.name)

    setup_source = json.loads(
        (PROJECT_ROOT / "reports" / "ocr" / "model_setup.json").read_text(
            encoding="utf-8"
        )
    )
    setup_source.update(
        {
            "device": "cpu",
            "asset_root": "../../assets",
            "cache_root": "../../assets/ocr_models",
            "offline": True,
            "portable": True,
        }
    )
    for name in OCR_MODEL_NAMES:
        source = (
            asset_root
            / "cache"
            / "paddlex"
            / "official_models"
            / name
        )
        destination = target / "assets" / "ocr_models" / name
        copy_tree(source, destination)
        setup_source["models"][name]["resolved_path"] = (
            f"../../assets/ocr_models/{name}"
        )
        setup_source["models"][name]["device"] = "cpu"
    write_json(target / "reports" / "ocr" / "model_setup.json", setup_source)

    calibration = json.loads(
        (PROJECT_ROOT / "models" / "multitask_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    calibration["checkpoint"] = "assets/checkpoints/layoutxlm_multitask/final"
    calibration["manifest_path"] = "not included; public training manifest"
    write_json(target / "models" / "multitask_calibration.json", calibration)

    records = []
    for path in sorted(
        [
            *checkpoint_target.rglob("*"),
            *(target / "assets" / "ocr_models").rglob("*"),
            *(target / "models" / "kmeans_rotation").rglob("*"),
            target / "models" / "multitask_calibration.json",
        ]
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if relative.startswith("assets/ocr_models/"):
            role = "ocr_model"
            license_name = "Apache-2.0"
        elif relative.endswith("model.safetensors"):
            role = "layout_checkpoint"
            license_name = "CC-BY-NC-SA-4.0"
        elif relative.startswith("assets/checkpoints/"):
            role = "layout_checkpoint_metadata"
            license_name = "CC-BY-NC-SA-4.0"
        elif relative.startswith("models/kmeans_rotation/"):
            role = "display_only_rotation_artifact"
            license_name = "project"
        else:
            role = "calibration"
            license_name = "project"
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
                "license": license_name,
            }
        )
    write_json(
        target / "MODEL_MANIFEST.json",
        {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "final_layout_model_sha256": EXPECTED_LAYOUT_SHA256,
            "files": records,
        },
    )
    return records


def copy_samples(target: Path, asset_root: Path) -> None:
    fixture_root = asset_root / "generated" / "integration_smoke" / "fixtures"
    copy_file(
        fixture_root / "unknown_upright.png",
        target / "samples" / "unknown_upright.png",
    )
    (target / "samples" / "README.md").write_text(
        "# Synthetic sample\n\n"
        "`unknown_upright.png` is a generated integration fixture containing "
        "no real person, account, or private document data. Use it for the "
        "first CLI/GUI run and Devpost screenshots.\n",
        encoding="utf-8",
    )


def privacy_audit(target: Path) -> dict[str, Any]:
    files = [path for path in target.rglob("*") if path.is_file()]
    prohibited = []
    for path in files:
        relative = path.relative_to(target)
        lowered = [part.casefold() for part in relative.parts]
        if relative.name.casefold() in {
            ".env",
            "runtime.local.json",
            "private_file_inventory.csv",
        }:
            prohibited.append(relative.as_posix())
        if lowered and lowered[0] == "data":
            prohibited.append(relative.as_posix())
        if any(part in {"private_outputs", "private-evaluation"} for part in lowered):
            prohibited.append(relative.as_posix())
    audit = {
        "status": "pass" if not prohibited else "fail",
        "checked_file_count": len(files),
        "prohibited_files": sorted(set(prohibited)),
        "raw_data_included": False,
        "private_gmail_data_included": False,
        "private_outputs_included": False,
        "credentials_included": False,
        "safe_sample_count": 1,
        "method": "allowlisted source copy plus prohibited-path scan",
    }
    write_json(target / "PRIVACY_AUDIT.json", audit)
    if prohibited:
        raise ValueError("privacy audit failed: " + ", ".join(prohibited))
    return audit


def build_info(target: Path, model_records: list[dict[str, Any]]) -> None:
    status = git_value("status", "--porcelain") or ""
    write_json(
        target / "BUILD_INFO.json",
        {
            "schema_version": "1.0",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_repository": "PracticalSwan/csx4201-vision-info-extraction",
            "source_commit": git_value("rev-parse", "HEAD"),
            "source_tree_dirty_at_build": bool(status),
            "model_file_count": len(model_records),
            "model_size_bytes": sum(item["size_bytes"] for item in model_records),
            "supported_hosts": {
                "windows": "native Python 3.10 setup; CPU or compatible NVIDIA GPU",
                "macos": "Docker Desktop CPU runtime using linux/amd64 emulation",
            },
            "physical_mac_tested": False,
            "uses_openai_api": False,
        },
    )


def make_zip(target: Path) -> tuple[Path, str]:
    archive = target.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as output:
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            arcname = (Path(target.name) / path.relative_to(target)).as_posix()
            # ZipFile.write streams large model files instead of reading the
            # 1+ GiB checkpoint into memory. macOS instructions invoke the
            # launchers through `bash`, so executable-bit preservation is not
            # required.
            output.write(path, arcname)
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(
        f"{digest}  {archive.name}\n",
        encoding="ascii",
    )
    return archive, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    prepare_target(target, force=args.force)
    copy_application(target)
    portable_config(target)
    model_records = copy_models(target, asset_root)
    copy_samples(target, asset_root)
    privacy = privacy_audit(target)
    build_info(target, model_records)
    archive = None
    archive_hash = None
    if args.create_zip:
        archive, archive_hash = make_zip(target)
    print(
        json.dumps(
            {
                "status": "complete",
                "target": str(target),
                "file_count": sum(1 for path in target.rglob("*") if path.is_file()),
                "model_size_bytes": sum(item["size_bytes"] for item in model_records),
                "privacy_audit": privacy["status"],
                "archive": str(archive) if archive else None,
                "archive_sha256": archive_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
