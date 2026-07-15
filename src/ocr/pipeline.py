"""Multilingual, rotation-aware OCR orchestration independent of K-Means."""
from __future__ import annotations

import time
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.ocr.cache import OCRCache, OCRCacheKey
from src.ocr.language_router import (
    normalize_language_mode,
    route_for_mode,
    select_route_result,
    should_try_thai,
)
from src.ocr.model_registry import ModelRegistry
from src.ocr.orientation_candidates import (
    build_orientation_candidates,
    restore_original_coordinates,
)
from src.ocr.paddleocr_adapter import PaddleOCRAdapter
from src.ocr.scoring import score_ocr_candidate


class OCRBackend(Protocol):
    def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict[str, Any]: ...


class MultilingualOCR:
    """Evaluate OCR views/routes and return the auditable best result."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        device: str = "cpu",
        general_backend: OCRBackend | None = None,
        thai_backend: OCRBackend | None = None,
        cardinal_angles: tuple[float, ...] = (0, 90, 180, 270),
        adapter_options: Mapping[str, Any] | None = None,
        cache: OCRCache | None = None,
        preprocessing_version: str = "1.0",
    ) -> None:
        options = dict(adapter_options or {})
        if general_backend is None:
            if registry is None:
                raise ValueError("registry is required when no general backend is injected")
            general_backend = PaddleOCRAdapter(registry, "general", device=device, **options)
        if thai_backend is None:
            if registry is None:
                raise ValueError("registry is required when no Thai backend is injected")
            thai_backend = PaddleOCRAdapter(registry, "thai", device=device, **options)
        self.backends = {"general": general_backend, "thai": thai_backend}
        self.cardinal_angles = tuple(float(value) for value in cardinal_angles)
        self.cache = cache
        self.preprocessing_version = preprocessing_version

    def extract_path(
        self,
        image_path: str | Path,
        *,
        language_mode: str = "auto",
        language_hint: str | None = None,
        metadata_language: str | None = None,
        deskew_angle: float | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        """Extract a local image with provenance-complete public caching."""
        path = Path(image_path)
        key = self._cache_key(
            path,
            language_mode=language_mode,
            language_hint=language_hint,
            metadata_language=metadata_language,
            deskew_angle=deskew_angle,
        ) if self.cache is not None else None
        if self.cache is not None and key is not None:
            cached = self.cache.get(key, private=private)
            if cached is not None:
                return cached
        with Image.open(path) as image:
            result = self.extract_page(
                image.convert("RGB"), language_mode=language_mode,
                language_hint=language_hint, metadata_language=metadata_language,
                deskew_angle=deskew_angle,
            )
        if self.cache is not None and key is not None:
            self.cache.put(key, result, private=private)
        return result

    def extract_page(
        self,
        image: Image.Image,
        *,
        language_mode: str = "auto",
        language_hint: str | None = None,
        metadata_language: str | None = None,
        deskew_angle: float | None = None,
    ) -> dict[str, Any]:
        """Run OCR without any K-Means input or dependency."""
        started = time.perf_counter()
        mode = normalize_language_mode(language_mode)
        forced_route = route_for_mode(mode)
        candidates = build_orientation_candidates(
            image, cardinal_angles=self.cardinal_angles, deskew_angle=deskew_angle
        )
        general_best = None
        thai_best = None
        route_reasons: list[str] = []
        if forced_route in {None, "general"}:
            general_best = self._best_route("general", candidates, image.size)
        if forced_route == "thai":
            thai_best = self._best_route("thai", candidates, image.size)
        elif forced_route is None:
            assert general_best is not None
            run_thai, route_reasons = should_try_thai(
                general_best[0], general_best[1],
                language_hint=language_hint, metadata_language=metadata_language,
            )
            if run_thai:
                thai_best = self._best_route("thai", candidates, image.size)
        hints = {str(language_hint or "").lower(), str(metadata_language or "").lower()}
        preferred_route = "thai" if hints & {"th", "thai", "th-th"} else None
        selected, route_decision = select_route_result(
            general_best, thai_best, preferred_route=preferred_route
        )
        selected = dict(selected)
        selected["route_decision"] = {**route_decision, "thai_evaluation_reasons": route_reasons}
        selected["candidate_scores"] = [
            candidate for pair in (general_best, thai_best) if pair is not None
            for candidate in pair[0].get("all_candidate_scores", [])
        ]
        selected["duration_seconds"] = time.perf_counter() - started
        return selected

    def _best_route(
        self, route: str, candidates: list, source_size: tuple[int, int]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        backend = self.backends[route]
        evaluated: list[tuple[dict[str, Any], dict[str, float]]] = []
        candidate_scores: list[dict[str, Any]] = []
        for candidate in candidates:
            result = backend.predict(candidate.image, orientation=candidate.angle)
            score = score_ocr_candidate(
                result, candidate.transform.output_width, candidate.transform.output_height
            )
            restored = restore_original_coordinates(result, candidate)
            entry = {
                "route": route,
                "orientation": candidate.angle,
                "candidate_kind": candidate.kind,
                **score,
            }
            candidate_scores.append(entry)
            evaluated.append((restored, score))
        best_result, best_score = max(
            evaluated, key=lambda pair: (pair[1]["total"], -float(pair[0]["orientation"]))
        )
        best_result["all_candidate_scores"] = candidate_scores
        best_result["source_width"], best_result["source_height"] = source_size
        return best_result, best_score

    def _cache_key(self, path: Path, **configuration: Any) -> OCRCacheKey | None:
        provenance = {}
        for route, backend in self.backends.items():
            method = getattr(backend, "provenance", None)
            if not callable(method):
                return None
            provenance[route] = dict(method())
        general = provenance["general"]
        thai = provenance["thai"]
        recognizer_hash = hashlib.sha256(
            (str(general["recognizer_artifact_hash"]) + str(thai["recognizer_artifact_hash"])).encode("ascii")
        ).hexdigest()
        return OCRCacheKey.from_image(
            path,
            detector_model=str(general["detector_model"]),
            detector_artifact_hash=str(general["detector_artifact_hash"]),
            recognizer_model=f"{general['recognizer_model']}+{thai['recognizer_model']}",
            recognizer_artifact_hash=recognizer_hash,
            language_route_configuration=configuration,
            orientation_configuration={"cardinal_angles": self.cardinal_angles},
            paddleocr_version=str(general["paddleocr_version"]),
            preprocessing_version=self.preprocessing_version,
        )
