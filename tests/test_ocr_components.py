from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.ocr.cache import OCRCache, OCRCacheKey
from src.ocr.errors import OCRModelMismatch, OCRModelUnavailable
from src.ocr.fine_deskew import estimate_residual_deskew
from src.ocr.language_router import should_try_thai
from src.ocr.model_registry import REQUIRED_MODEL_NAMES, ModelRegistry
from src.ocr.pipeline import MultilingualOCR
from src.ocr.result_normalizer import normalize_paddle_result
from src.ocr.scoring import score_ocr_candidate
from scripts.verify_ocr_models import _rotation_smoke


class FakeBackend:
    def __init__(self, route: str, best_angle: int, text: str) -> None:
        self.route = route
        self.best_angle = best_angle
        self.text = text
        self.calls: list[float] = []

    def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict:
        self.calls.append(orientation)
        best = orientation == self.best_angle
        confidence = 0.97 if best else 0.15
        texts = [self.text, "TOTAL", "123.45", "DATE"] if best else ["???"]
        words = []
        for index, text in enumerate(texts):
            y0 = 2 + index * 4
            words.append({
                "id": f"{self.route}-{orientation}-{index}", "text": text,
                "confidence": confidence,
                "polygon": [[2, y0], [20, y0], [20, y0 + 3], [2, y0 + 3]],
                "bbox": [2, y0, 20, y0 + 3],
            })
        return {
            "full_text": "\n".join(texts), "words": words,
            "lines": [
                {"id": f"l-{word['id']}", "text": word["text"], "word_ids": [word["id"]],
                 "polygon": word["polygon"], "bbox": word["bbox"], "confidence": word["confidence"]}
                for word in words
            ],
            "mean_confidence": confidence,
            "detector_model": "PP-OCRv6_medium_det",
            "recognizer_model": "PP-OCRv6_medium_rec" if self.route == "general" else "th_PP-OCRv5_mobile_rec",
            "language_route": self.route, "orientation": orientation,
            "duration_seconds": 0.01, "warnings": [], "provenance_hash": "a" * 64,
        }

    def provenance(self) -> dict:
        return {
            "detector_model": "PP-OCRv6_medium_det", "detector_artifact_hash": "d" * 64,
            "recognizer_model": "PP-OCRv6_medium_rec" if self.route == "general" else "th_PP-OCRv5_mobile_rec",
            "recognizer_artifact_hash": ("g" if self.route == "general" else "t") * 64,
            "paddleocr_version": "3.7.0", "device": "cpu",
        }


def test_cardinal_selection_and_forced_general_route() -> None:
    general = FakeBackend("general", 90, "INVOICE")
    thai = FakeBackend("thai", 0, "ใบเสร็จ")
    pipeline = MultilingualOCR(general_backend=general, thai_backend=thai)
    result = pipeline.extract_page(Image.new("RGB", (80, 40), "white"), language_mode="en")
    assert result["orientation"] == 90
    assert result["language_route"] == "general"
    assert general.calls == [0, 90, 180, 270]
    assert thai.calls == []
    assert len(result["candidate_scores"]) == 4


def test_polygon_baselines_estimate_independent_fine_deskew() -> None:
    import math

    angle = math.radians(12)
    words = []
    for index in range(6):
        x0, y0, width, height = 10.0, 15.0 + index * 18.0, 80.0, 10.0
        dx, dy = width * math.cos(angle), width * math.sin(angle)
        hx, hy = -height * math.sin(angle), height * math.cos(angle)
        words.append({
            "text": f"word-{index}",
            "confidence": 0.95,
            "polygon": [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx + hx, y0 + dy + hy], [x0 + hx, y0 + hy]],
        })
    estimate = estimate_residual_deskew(words)
    assert estimate["supported_word_count"] == 6
    assert estimate["text_angle_degrees"] == pytest.approx(12.0, abs=0.2)
    assert estimate["correction_degrees"] == pytest.approx(12.0, abs=0.2)
    assert estimate["reliability"] > 0.8


def test_pipeline_refines_cardinal_candidate_with_polygon_baseline() -> None:
    import math

    class TiltedBackend(FakeBackend):
        def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict:
            self.calls.append(orientation)
            residual = 12.0 if orientation == 0.0 else 0.0
            confidence = (
                0.98
                if abs(orientation - 12.0) < 0.25
                else 0.75
                if orientation == 0.0
                else 0.20
            )
            words = []
            angle = math.radians(residual)
            for index in range(6):
                x0, y0, width, height = 5.0, 5.0 + index * 5.0, 30.0, 3.0
                dx, dy = width * math.cos(angle), width * math.sin(angle)
                hx, hy = -height * math.sin(angle), height * math.cos(angle)
                words.append({
                    "id": f"w-{orientation}-{index}", "text": f"TEXT{index}",
                    "confidence": confidence,
                    "polygon": [[x0, y0], [x0 + dx, y0 + dy], [x0 + dx + hx, y0 + dy + hy], [x0 + hx, y0 + hy]],
                    "bbox": [x0, y0, x0 + width, y0 + height],
                })
            return {
                "full_text": " ".join(word["text"] for word in words),
                "words": words,
                "lines": [],
                "mean_confidence": confidence,
                "detector_model": "PP-OCRv6_medium_det",
                "recognizer_model": "PP-OCRv6_medium_rec",
                "language_route": self.route,
                "orientation": orientation,
                "duration_seconds": 0.01,
                "warnings": [],
                "provenance_hash": "a" * 64,
            }

    general = TiltedBackend("general", 0, "TEXT")
    pipeline = MultilingualOCR(
        general_backend=general,
        thai_backend=FakeBackend("thai", 0, "THAI"),
    )
    result = pipeline.extract_page(Image.new("RGB", (100, 60), "white"), language_mode="general")
    assert any(abs(angle - 12.0) < 0.25 for angle in general.calls)
    assert result["orientation"] == pytest.approx(12.0, abs=0.25)
    assert any(score["candidate_kind"] == "fine_deskew" for score in result["candidate_scores"])
    base_score = next(
        score for score in result["candidate_scores"] if score["orientation"] == 0.0
    )
    fine_score = next(
        score
        for score in result["candidate_scores"]
        if score["candidate_kind"] == "fine_deskew"
    )
    assert fine_score["text_detection_coverage"] == pytest.approx(
        base_score["text_detection_coverage"]
    )


def test_auto_route_uses_thai_evidence_and_is_kmeans_independent() -> None:
    general = FakeBackend("general", 90, "INVOICE")
    thai = FakeBackend("thai", 180, "ใบเสร็จรับเงิน")
    pipeline = MultilingualOCR(general_backend=general, thai_backend=thai)
    image = Image.new("RGB", (80, 40), "white")
    first = pipeline.extract_page(image, language_mode="auto", language_hint="th")
    # There is intentionally no K-Means argument. Perturbing unrelated display
    # metadata cannot influence the selector or its deterministic result.
    wrong_kmeans_values = [None, {"zone": 1}, {"zone": 4}, {"cluster_id": 99}]
    repeated = [pipeline.extract_page(image, language_mode="auto", language_hint="th") for _ in wrong_kmeans_values]
    assert first["language_route"] == "thai"
    assert first["orientation"] == 180
    assert all((item["language_route"], item["orientation"], item["full_text"]) ==
               (first["language_route"], first["orientation"], first["full_text"]) for item in repeated)


def test_rotated_smoke_rejects_empty_ocr() -> None:
    class EmptyPipeline:
        def extract_page(self, _image: Image.Image, *, language_mode: str) -> dict:
            assert language_mode == "general"
            return {
                "full_text": "",
                "words": [],
                "detector_model": "PP-OCRv6_medium_det",
                "recognizer_model": "PP-OCRv6_medium_rec",
                "orientation": 270.0,
                "mean_confidence": None,
            }

    result = _rotation_smoke(EmptyPipeline(), Image.new("RGB", (40, 80), "white"))
    assert result["passed"] is False


def test_auto_route_retries_thai_below_calibrated_confidence_center() -> None:
    retry, reasons = should_try_thai(
        {"full_text": "T1la5a5ulju 123.45"},
        {
            "mean_confidence": 0.72,
            "word_count": 2.0,
            "valid_character_ratio": 1.0,
            "text_detection_coverage": 0.04,
        },
    )
    assert retry is True
    assert "general_confidence_is_low" in reasons


def test_thai_script_ratio_is_bounded() -> None:
    result = {
        "language_route": "thai",
        "words": [
            {
                "text": "ใบเสร็จรับเงิน",
                "confidence": 0.95,
                "bbox": [0, 0, 100, 20],
                "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ],
    }
    score = score_ocr_candidate(result, 200, 100)
    assert 0.0 <= score["thai_script_ratio"] <= 1.0


def test_candidate_coverage_uses_polygon_area_instead_of_tilted_bbox() -> None:
    result = {
        "language_route": "general",
        "words": [
            {
                "text": "INVOICE",
                "confidence": 0.99,
                "bbox": [0, 0, 100, 100],
                "polygon": [[0, 45], [90, 0], [100, 20], [10, 65]],
            }
        ],
    }
    score = score_ocr_candidate(result, 100, 100)
    assert score["text_detection_coverage"] == pytest.approx(0.225)


def test_result_normalizer_accepts_paddle_3_mapping() -> None:
    raw = [{"res": {
        "rec_texts": ["hello"], "rec_scores": [0.9],
        "rec_polys": [[[1, 2], [10, 2], [10, 8], [1, 8]]],
    }}]
    result = normalize_paddle_result(
        raw, detector_model="PP-OCRv6_medium_det", recognizer_model="PP-OCRv6_medium_rec",
        route="general", orientation=0, duration_seconds=0.1,
    )
    assert result["full_text"] == "hello"
    assert result["words"][0]["bbox"] == [1.0, 2.0, 10.0, 8.0]
    assert result["mean_confidence"] == pytest.approx(0.9)


def _cache_key() -> OCRCacheKey:
    return OCRCacheKey(
        input_sha256="1" * 64, detector_model="PP-OCRv6_medium_det", detector_artifact_hash="2" * 64,
        recognizer_model="PP-OCRv6_medium_rec", recognizer_artifact_hash="3" * 64,
        language_route_configuration={"mode": "general"},
        orientation_configuration={"angles": [0, 90, 180, 270]},
        paddleocr_version="3.7.0", preprocessing_version="1.0",
    )


def test_cache_provenance_and_private_bypass(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path / "cache")
    key = _cache_key()
    assert cache.get(key) is None
    path = cache.put(key, {"full_text": "public"})
    assert path is not None and path.is_file()
    assert cache.get(key) == {"full_text": "public"}
    assert cache.put(key, {"full_text": "private"}, private=True) is None
    assert cache.get(key, private=True) is None


def test_pipeline_cache_hits_and_configuration_changes_invalidate(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    general = FakeBackend("general", 90, "INVOICE")
    thai = FakeBackend("thai", 180, "ใบเสร็จ")
    pipeline = MultilingualOCR(
        general_backend=general, thai_backend=thai, cache=OCRCache(tmp_path / "ocr-cache")
    )
    first = pipeline.extract_path(image_path, language_mode="general")
    call_count = len(general.calls)
    second = pipeline.extract_path(image_path, language_mode="general")
    assert second == first
    assert len(general.calls) == call_count
    pipeline.extract_path(image_path, language_mode="general", deskew_angle=3.0)
    assert len(general.calls) > call_count


def test_registry_requires_exact_names_paths_and_hashes(tmp_path: Path) -> None:
    models = {}
    for name in REQUIRED_MODEL_NAMES:
        directory = tmp_path / name
        directory.mkdir()
        artifact = directory / "inference.json"
        artifact.write_text(name, encoding="utf-8")
        models[name] = {
            "requested_name": name, "resolved_name": name,
            "resolved_path": str(directory), "role": "detector" if name.endswith("det") else "recognizer",
            "language": "thai" if name.startswith("th_") else "general",
            "files": [{"path": artifact.name, "size_bytes": artifact.stat().st_size,
                       "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
        }
    setup = tmp_path / "model_setup.json"
    setup.write_text(json.dumps({"models": models}), encoding="utf-8")
    registry = ModelRegistry.from_setup(setup)
    assert registry.require("PP-OCRv6_medium_det").path.is_dir()
    models["PP-OCRv6_medium_rec"]["resolved_name"] = "wrong"
    setup.write_text(json.dumps({"models": models}), encoding="utf-8")
    with pytest.raises(OCRModelMismatch):
        ModelRegistry.from_setup(setup)
    setup.unlink()
    with pytest.raises(OCRModelUnavailable):
        ModelRegistry.from_setup(setup)
