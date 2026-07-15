from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

import run_rotation_experiment
from src.rotation_common import read_csv_rows, sha256_file
from tests.rotation_test_helpers import make_rotation_config, write_csv


def _write_smoke_inventory(root: Path) -> dict[str, str]:
    columns = [
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
        "original_filename",
    ]
    public_rows = []
    for index in range(12):
        relative = f"sroie/train/img/smoke_document_{index:02d}.png"
        source = root / "data/raw/public" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (180, 110), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 170, 100), outline="black", width=3)
        draw.rectangle((18, 18, 86 + index, 30), fill="black")
        draw.line((18, 48, 155 - index, 48), fill="black", width=4)
        draw.line((18, 68 + index % 5, 130, 68 + index % 5), fill="black", width=3)
        draw.rectangle((120 + index, 76, 157, 94), outline="black", width=3)
        image.save(source)
        public_rows.append({
            "file_id": f"smoke_public_image_{index:02d}",
            "document_id": f"smoke_public_document_{index:02d}",
            "dataset": "sroie",
            "current_relative_path": relative,
            "extension": ".png",
            "is_image": True,
            "is_pdf": False,
            "is_annotation": False,
            "is_readable": True,
            "is_empty": False,
            "sha256": sha256_file(source),
            "document_category": "receipt",
            "original_filename": source.name,
        })

    private_pdf = root / "data/raw/private/gmail/Private Smoke.pdf"
    private_pdf.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=180, height=110)
    page.insert_text((18, 30), "private synthetic smoke page")
    page.draw_rect(fitz.Rect(8, 8, 170, 100), color=(0, 0, 0), width=2)
    document.save(private_pdf)
    document.close()
    private_sha = sha256_file(private_pdf)
    public_rows.append({
        "file_id": "smoke_private_pdf",
        "document_id": "smoke_private_document",
        "dataset": "gmail",
        "current_relative_path": "gmail/receipts/gmail_private_smoke.pdf",
        "extension": ".pdf",
        "is_image": False,
        "is_pdf": True,
        "is_annotation": False,
        "is_readable": True,
        "is_empty": False,
        "sha256": private_sha,
        "document_category": "private_document",
        "original_filename": "gmail_private_smoke.pdf",
    })
    private_rows = [{
        **public_rows[-1],
        "current_relative_path": "gmail/Private Smoke.pdf",
        "original_filename": "Private Smoke.pdf",
    }]
    write_csv(root / "data/metadata/file_inventory.csv", public_rows, columns)
    write_csv(root / "data/metadata/private_file_inventory.csv", private_rows, columns)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in (root / "data/raw").rglob("*")
        if path.is_file()
    }


def test_real_full_smoke_orchestrator_completes_without_changing_raw(tmp_path: Path, monkeypatch, capsys):
    cfg = make_rotation_config(tmp_path)
    cfg["pca"]["enabled"] = False
    raw_hashes = _write_smoke_inventory(tmp_path)
    monkeypatch.setattr(run_rotation_experiment, "load", lambda args: cfg)
    monkeypatch.setattr(sys, "argv", ["run_rotation_experiment.py", "--profile", "smoke"])

    assert run_rotation_experiment.main() == 0
    output = capsys.readouterr().out
    assert "Record or load raw baseline" in output
    assert "Run final complete-pipeline verification" in output
    assert "Rotation experiment complete" in output
    assert {
        path.relative_to(tmp_path).as_posix(): sha256_file(path)
        for path in (tmp_path / "data/raw").rglob("*")
        if path.is_file()
    } == raw_hashes

    metadata = tmp_path / "data/metadata"
    rotations = read_csv_rows(metadata / "rotation_manifest.csv")
    assert len(rotations) == 16
    assert {row["rotation_zone"] for row in rotations} == {"1", "2", "3", "4"}
    assert {row["project_split"] for row in rotations} == {"train", "validation", "test", "private_test"}
    assert (metadata / "feature_manifest.csv").is_file()
    assert (tmp_path / "models/rotation/kmeans.joblib").is_file()
    assert (tmp_path / "reports/kmeans_evaluation/metrics.json").is_file()
    assert (tmp_path / "reports/angle_estimation/metrics.json").is_file()
    verification = (tmp_path / "reports/verification/rotation_verification.json").read_text(encoding="utf-8")
    assert '"all_passed": true' in verification


def test_orchestrator_stops_before_features_when_leakage_verification_fails(tmp_path: Path, monkeypatch):
    cfg = make_rotation_config(tmp_path)
    calls: list[str] = []

    def stage(name, result):
        def invoke(*args, **kwargs):
            calls.append(name)
            return result
        return invoke

    monkeypatch.setattr(run_rotation_experiment, "load", lambda args: cfg)
    monkeypatch.setattr(run_rotation_experiment, "record_raw_baseline", stage("baseline", {}))
    monkeypatch.setattr(run_rotation_experiment, "prepare_page_images", stage("prepare", {}))
    monkeypatch.setattr(run_rotation_experiment, "create_rotation_splits", stage("split", {}))
    monkeypatch.setattr(run_rotation_experiment, "generate_rotation_data", stage("rotate", {}))
    monkeypatch.setattr(
        run_rotation_experiment,
        "verify_rotation_data",
        stage("verify", {
            "all_passed": False,
            "checks": [{"name": "split-leakage", "passed": False, "detail": "document leakage across splits"}],
        }),
    )

    def forbidden_feature_stage(*args, **kwargs):
        raise AssertionError("feature extraction must not run after leakage")

    monkeypatch.setattr(run_rotation_experiment, "extract_rotation_features", forbidden_feature_stage)
    monkeypatch.setattr(sys, "argv", ["run_rotation_experiment.py", "--profile", "smoke"])
    assert run_rotation_experiment.main() == 1
    assert calls == ["baseline", "prepare", "split", "rotate", "verify"]


def test_rotation_implementation_imports_no_ocr_or_neural_frameworks():
    root = Path(__file__).resolve().parents[1]
    implementation = [
        root / "src/page_preparation.py",
        root / "src/rotation_common.py",
        root / "src/rotation_dataset.py",
        root / "src/orientation_features.py",
        root / "src/rotation_model.py",
        root / "src/angle_estimation.py",
        *sorted((root / "scripts").glob("*rotation*.py")),
        root / "scripts/prepare_page_images.py",
    ]
    forbidden_roots = {
        "pytesseract",
        "easyocr",
        "paddleocr",
        "keras",
        "tensorflow",
        "torch",
        "transformers",
        "onnxruntime",
    }
    discovered: list[tuple[str, str]] = []
    for path in implementation:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".", 1)[0]]
            else:
                continue
            discovered.extend((path.name, name) for name in roots if name in forbidden_roots)
    assert discovered == []


def test_documented_python_script_commands_resolve_to_real_entrypoints():
    root = Path(__file__).resolve().parents[1]
    documents = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    command_pattern = re.compile(r"python\s+scripts[\\/]([A-Za-z0-9_.-]+\.py)")
    referenced = {
        match.group(1)
        for document in documents
        for match in command_pattern.finditer(document.read_text(encoding="utf-8"))
    }
    required = {
        "prepare_page_images.py",
        "create_rotation_splits.py",
        "generate_rotation_data.py",
        "verify_rotation_data.py",
        "extract_rotation_features.py",
        "fit_rotation_preprocessing.py",
        "train_kmeans_rotation.py",
        "evaluate_kmeans_rotation.py",
        "evaluate_angle_estimation.py",
        "run_rotation_experiment.py",
    }
    assert required <= referenced
    assert referenced
    missing = sorted(name for name in referenced if not (root / "scripts" / name).is_file())
    assert missing == []
