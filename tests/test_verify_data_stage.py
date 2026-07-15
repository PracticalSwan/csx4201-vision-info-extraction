"""Stage-aware checks for the legacy organization verifier."""
from __future__ import annotations

from pathlib import Path

import verify_data


def _touch(path: Path, content: bytes = b"test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_classical_rotation_artifacts_are_allowed(tmp_path: Path) -> None:
    _touch(tmp_path / "models" / "kmeans_rotation" / "kmeans.joblib")
    _touch(tmp_path / "models" / "kmeans_rotation" / "training_summary.json")
    _touch(tmp_path / "data" / "processed" / "rotated_images" / "full" / "sample.png")
    _touch(tmp_path / "data" / "processed" / "features" / "full" / "train.npz")
    _touch(tmp_path / "data" / "splits" / "train.csv")
    _touch(tmp_path / "reports" / "kmeans_evaluation" / "metrics.json")

    passed, detail = verify_data._check_forbidden(tmp_path)

    assert passed, detail


def test_raw_sroie_bin_is_exempt_but_generated_bin_is_rejected(tmp_path: Path) -> None:
    _touch(
        tmp_path
        / "data"
        / "raw"
        / "public"
        / "sroie"
        / "layoutlm-base-uncased"
        / "pytorch_model.bin"
    )
    passed, detail = verify_data._check_forbidden(tmp_path)
    assert passed, detail

    _touch(tmp_path / "models" / "neural" / "pytorch_model.bin")
    passed, detail = verify_data._check_forbidden(tmp_path)
    assert not passed
    assert "models/neural/pytorch_model.bin" in detail


def test_ocr_and_checkpoint_outputs_are_rejected_outside_raw(tmp_path: Path) -> None:
    _touch(tmp_path / "data" / "processed" / "ocr_outputs" / "result.json")
    _touch(tmp_path / "checkpoints" / "state.txt")

    passed, detail = verify_data._check_forbidden(tmp_path)

    assert not passed
    assert "data/processed/ocr_outputs" in detail
    assert "checkpoints" in detail

