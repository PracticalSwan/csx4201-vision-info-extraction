"""Local image/PDF to schema-validated information-extraction JSON."""
from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src import config as cfgmod
from src.information_extraction.generic_entities import generic_key_value_entities
from src.information_extraction.relations import infer_relations
from src.information_extraction.rules import extract_rule_fields
from src.information_extraction.schema import build_document_result, validate_document_result
from src.inference.document_io import DocumentInputError, DocumentPage, load_document_pages
from src.inference.kmeans_display import KMeansRotationDisplay, safe_kmeans_display
from src.ocr.cache import OCRCache
from src.ocr.model_registry import ModelRegistry
from src.ocr.pipeline import MultilingualOCR


class DocumentPipelineError(RuntimeError):
    """Raised when the main extraction path cannot produce a result."""


class DocumentPipeline:
    def __init__(
        self,
        *,
        ocr: Any,
        device: str,
        entity_extractor: Any | None = None,
        kmeans_predictor: Any | None = None,
        enable_kmeans_display: bool = True,
        initialization_warnings: Sequence[str] = (),
    ) -> None:
        self.ocr = ocr
        self.device = device
        self.entity_extractor = entity_extractor
        self.kmeans_predictor = kmeans_predictor
        self.enable_kmeans_display = bool(enable_kmeans_display)
        self.initialization_warnings = list(initialization_warnings)

    def close(self) -> None:
        close = getattr(self.entity_extractor, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        device: str = "cpu",
        model_setup: str | Path = "reports/ocr/model_setup.json",
        layout_checkpoint: str | Path | None = None,
        enable_kmeans_display: bool = True,
    ) -> "DocumentPipeline":
        registry = ModelRegistry.from_setup(model_setup)
        cache = OCRCache(cfgmod.resolve_path(cfg, "ocr_cache")) if cfg.get("ocr", {}).get("cache_enabled", True) else None
        options = {
            "use_doc_orientation_classify": bool(cfg.get("ocr", {}).get("enable_document_orientation_classifier", False)),
            "use_doc_unwarping": bool(cfg.get("ocr", {}).get("enable_document_unwarping", False)),
            "use_textline_orientation": bool(cfg.get("ocr", {}).get("enable_textline_orientation", False)),
        }
        ocr = MultilingualOCR(
            registry,
            device=device,
            cardinal_angles=tuple(cfg.get("ocr", {}).get("orientation_candidates", [0, 90, 180, 270])),
            adapter_options=options,
            cache=cache,
            preprocessing_version=str(cfg.get("ocr", {}).get("preprocessing_version", "1.0")),
        )
        warnings: list[str] = []
        entity_extractor = None
        checkpoint = Path(layout_checkpoint) if layout_checkpoint else (
            cfgmod.resolve_path(cfg, "ie_checkpoints") / "layoutxlm" / "smoke"
        )
        try:
            from src.information_extraction.entity_worker_client import (
                SubprocessLayoutEntityExtractor,
            )

            torch_device = "cuda" if device.startswith("gpu") else "cpu"
            entity_extractor = SubprocessLayoutEntityExtractor(
                checkpoint,
                python_executable=cfgmod.resolve_path(cfg, "layout_python"),
                device=torch_device,
                cache_dir=cfgmod.resolve_path(cfg, "layout_models"),
                max_length=int(cfg.get("layout_model", {}).get("max_length", 512)),
            )
        except Exception as exc:
            warnings.append(
                f"layout model unavailable; evidence-only generic/rule fallback active: {type(exc).__name__}: {exc}"
            )
        kmeans_predictor = None
        if enable_kmeans_display:
            try:
                kmeans_predictor = KMeansRotationDisplay(cfg)
            except Exception as exc:
                warnings.append(f"K-Means display initialization failed independently: {type(exc).__name__}: {exc}")
        return cls(
            ocr=ocr,
            device=device,
            entity_extractor=entity_extractor,
            kmeans_predictor=kmeans_predictor,
            enable_kmeans_display=enable_kmeans_display,
            initialization_warnings=warnings,
        )

    def extract_path(
        self,
        input_path: str | Path,
        *,
        language: str = "auto",
        language_hint: str | None = None,
        private_output: bool = False,
        max_pages: int | None = None,
        continue_on_page_error: bool = False,
        pdf_dpi: int = 200,
        deskew_angle: float | None = None,
    ) -> dict[str, Any]:
        try:
            document_id, source_type, pages = load_document_pages(
                input_path, max_pages=max_pages, pdf_dpi=pdf_dpi
            )
        except DocumentInputError as exc:
            raise DocumentPipelineError(str(exc)) from exc
        return self.extract_pages(
            document_id=document_id,
            source_type=source_type,
            pages=pages,
            language=language,
            language_hint=language_hint,
            private_output=private_output,
            continue_on_page_error=continue_on_page_error,
            deskew_angle=deskew_angle,
        )

    def extract_pages(
        self,
        *,
        document_id: str,
        source_type: str,
        pages: Sequence[DocumentPage],
        language: str = "auto",
        language_hint: str | None = None,
        private_output: bool = False,
        continue_on_page_error: bool = False,
        deskew_angle: float | None = None,
    ) -> dict[str, Any]:
        if not pages:
            raise DocumentPipelineError("document contains no pages")
        started = time.perf_counter()
        output_pages: list[dict[str, Any]] = []
        page_fields: list[dict[str, Any]] = []
        routes: list[str] = []
        confidences: list[float] = []
        warnings = list(self.initialization_warnings)
        for page in pages:
            try:
                result, fields = self._extract_page(
                    page,
                    language=language,
                    language_hint=language_hint,
                    deskew_angle=deskew_angle,
                )
            except Exception as exc:
                if not continue_on_page_error:
                    raise DocumentPipelineError(
                        f"page {page.page_number} extraction failed: {type(exc).__name__}: {exc}"
                    ) from exc
                result = _failed_page(page, exc)
                fields = {}
                warnings.append(f"page {page.page_number} failed and was retained as an empty result")
            output_pages.append(result)
            page_fields.append(fields)
            routes.append(str(result["ocr"]["language_route"]))
            if result["ocr"]["mean_confidence"] is not None:
                confidences.append(float(result["ocr"]["mean_confidence"]))

        fields, conflict_warnings = _merge_fields(page_fields)
        warnings.extend(conflict_warnings)
        title = fields.get("document_title")
        document_type = str(title["value"]) if title else "unknown"
        type_confidence = float(title["confidence"]) if title and title["confidence"] is not None else None
        detected_languages = _detected_languages(output_pages)
        selected_route = routes[0] if routes and len(set(routes)) == 1 else "auto"
        rotation_display = safe_kmeans_display(
            self.kmeans_predictor,
            pages[0].image,
            enabled=self.enable_kmeans_display,
        )
        payload = build_document_result(
            document_id=document_id,
            source_type=source_type,
            pages=output_pages,
            device=self.device,
            duration_seconds=time.perf_counter() - started,
            private_output=private_output,
            document_type=document_type,
            document_type_confidence=type_confidence,
            selected_language_route=selected_route,
            detected_languages=detected_languages,
            language_confidence=statistics.fmean(confidences) if confidences else None,
            fields=fields,
            rotation_display=rotation_display,
            warnings=warnings,
        )
        validate_document_result(payload)
        return payload

    def _extract_page(
        self,
        page: DocumentPage,
        *,
        language: str,
        language_hint: str | None,
        deskew_angle: float | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        image = page.image
        ocr = self.ocr.extract_page(
            image,
            language_mode=language,
            language_hint=language_hint,
            metadata_language=language_hint,
            deskew_angle=deskew_angle,
        )
        page_warnings = list(ocr.get("warnings") or [])
        entities: list[dict[str, Any]] = []
        if self.entity_extractor is not None:
            try:
                entities, model_warnings = self.entity_extractor.extract(
                    ocr,
                    page_number=page.page_number,
                    width=image.width,
                    height=image.height,
                )
                page_warnings.extend(model_warnings)
            except Exception as exc:
                page_warnings.append(
                    f"layout entity inference failed; generic fallback used: {type(exc).__name__}: {exc}"
                )
        generic = generic_key_value_entities(ocr, page_number=page.page_number)
        entities = _merge_entities(entities, generic)
        relations = infer_relations(entities)
        fields, rule_warnings = extract_rule_fields(ocr, page_number=page.page_number)
        page_warnings.extend(rule_warnings)
        transform = ocr.get("candidate_transform") or _identity_transform(image.width, image.height)
        page_result = {
            "page_number": int(page.page_number),
            "width": int(image.width),
            "height": int(image.height),
            "selected_ocr_orientation": float(ocr.get("orientation", 0.0)) % 360.0,
            "full_text": str(ocr.get("full_text", "")),
            "ocr": {
                "detector_model": str(ocr.get("detector_model", "unavailable")),
                "recognizer_model": str(ocr.get("recognizer_model", "unavailable")),
                "language_route": str(ocr.get("language_route", "general")),
                "mean_confidence": ocr.get("mean_confidence"),
                "words": list(ocr.get("words") or []),
                "lines": list(ocr.get("lines") or []),
                "candidate_scores": list(ocr.get("candidate_scores") or []),
                "provenance_hash": str(ocr.get("provenance_hash", "unavailable")),
                "duration_seconds": max(0.0, float(ocr.get("duration_seconds", 0.0))),
            },
            "entities": entities,
            "key_value_pairs": relations,
            "tables": [],
            "warnings": page_warnings,
            "transforms": {
                "forward": transform["forward"],
                "inverse": transform["inverse"],
            },
        }
        return page_result, fields


def _merge_entities(
    model_entities: Sequence[Mapping[str, Any]], generic_entities: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = [dict(entity) for entity in model_entities]
    existing = {(entity["label"], entity["text"].casefold()) for entity in result}
    for entity in generic_entities:
        key = (entity["label"], entity["text"].casefold())
        if key not in existing:
            result.append(dict(entity))
            existing.add(key)
    return result


def _merge_fields(values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for page_fields in values:
        for field, evidence in page_fields.items():
            grouped.setdefault(field, []).append(dict(evidence))
    merged: dict[str, Any] = {}
    warnings: list[str] = []
    for field, candidates in grouped.items():
        candidates.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        merged[field] = candidates[0]
        if len({str(item.get("value")) for item in candidates}) > 1:
            warnings.append(f"multiple page-level {field} values; selected highest-confidence evidence")
    return merged, warnings


def _detected_languages(pages: Sequence[Mapping[str, Any]]) -> list[str]:
    text = "\n".join(str(page.get("full_text", "")) for page in pages)
    has_thai = any("\u0e00" <= char <= "\u0e7f" for char in text)
    has_latin = any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)
    has_turkish = any(char in "ÇĞİÖŞÜçğıöşü" for char in text)
    result = []
    if has_thai:
        result.append("th")
    if has_turkish:
        result.append("tr")
    elif has_latin:
        result.append("en")
    return result


def _identity_transform(width: int, height: int) -> dict[str, Any]:
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {"forward": matrix, "inverse": matrix, "source_width": width, "source_height": height}


def _failed_page(page: DocumentPage, exc: Exception) -> dict[str, Any]:
    transform = _identity_transform(page.image.width, page.image.height)
    return {
        "page_number": int(page.page_number),
        "width": int(page.image.width),
        "height": int(page.image.height),
        "selected_ocr_orientation": 0.0,
        "full_text": "",
        "ocr": {
            "detector_model": "unavailable",
            "recognizer_model": "unavailable",
            "language_route": "general",
            "mean_confidence": None,
            "words": [],
            "lines": [],
            "candidate_scores": [],
            "provenance_hash": "unavailable",
            "duration_seconds": 0.0,
        },
        "entities": [],
        "key_value_pairs": [],
        "tables": [],
        "warnings": [f"page extraction failed: {type(exc).__name__}: {exc}"],
        "transforms": {"forward": transform["forward"], "inverse": transform["inverse"]},
    }
