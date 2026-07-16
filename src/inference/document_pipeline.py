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
from src.information_extraction.multitask_inference import reconstruct_ocr_tables
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
        calibration_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        enable_kmeans_display: bool = True,
        require_layout_model: bool = False,
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
            preprocessing_profile=str(cfg.get("ocr", {}).get("preprocessing_profile", "original")),
        )
        warnings: list[str] = []
        entity_extractor = None
        configured_checkpoint = cfg.get("layout_model", {}).get("inference_checkpoint")
        checkpoint = Path(layout_checkpoint) if layout_checkpoint else (
            _resolve_configured_path(cfg, configured_checkpoint)
            if configured_checkpoint
            else cfgmod.resolve_path(cfg, "ie_checkpoints") / "layoutxlm_multitask" / "final"
        )
        configured_calibration = cfg.get("layout_model", {}).get("calibration")
        calibration = Path(calibration_path) if calibration_path else (
            _resolve_configured_path(cfg, configured_calibration)
            if configured_calibration
            else cfgmod.project_root(cfg) / "models" / "multitask_calibration.json"
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
                calibration_path=calibration,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            if require_layout_model:
                raise DocumentPipelineError(
                    f"required final layout model failed to initialize: {type(exc).__name__}: {exc}"
                ) from exc
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
        page_document_types: list[dict[str, Any]] = []
        routes: list[str] = []
        confidences: list[float] = []
        warnings = list(self.initialization_warnings)
        for page in pages:
            try:
                result, fields, document_type = self._extract_page(
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
                document_type = {"label": "unknown", "confidence": None}
                warnings.append(f"page {page.page_number} failed and was retained as an empty result")
            output_pages.append(result)
            page_fields.append(fields)
            page_document_types.append(document_type)
            routes.append(str(result["ocr"]["language_route"]))
            if result["ocr"]["mean_confidence"] is not None:
                confidences.append(float(result["ocr"]["mean_confidence"]))

        fields, conflict_warnings = merge_document_fields(page_fields)
        warnings.extend(conflict_warnings)
        document_type, type_confidence, type_warnings = _merge_document_types(
            page_document_types
        )
        warnings.extend(type_warnings)
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
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        model_relations: list[dict[str, Any]] = []
        model_fields: dict[str, Any] = {}
        model_tables: list[dict[str, Any]] = []
        document_type = {"label": "unknown", "confidence": None}
        if self.entity_extractor is not None:
            try:
                model_result = self.entity_extractor.extract(
                    ocr,
                    page_number=page.page_number,
                    width=image.width,
                    height=image.height,
                )
                if isinstance(model_result, Mapping):
                    entities = list(model_result.get("entities") or [])
                    model_relations = list(model_result.get("relations") or [])
                    model_fields = dict(model_result.get("canonical_fields") or {})
                    model_tables = list(model_result.get("tables") or [])
                    document_type = dict(
                        model_result.get("document_type")
                        or {"label": "unknown", "confidence": None}
                    )
                    page_warnings.extend(model_result.get("warnings") or [])
                else:
                    entities, model_warnings = model_result
                    page_warnings.extend(model_warnings)
            except Exception as exc:
                page_warnings.append(
                    f"layout entity inference failed; generic fallback used: {type(exc).__name__}: {exc}"
                )
        if not model_tables:
            try:
                model_tables = reconstruct_ocr_tables(
                    ocr.get("words") or [], page_number=page.page_number
                )
            except Exception as exc:
                page_warnings.append(
                    f"OCR geometry table reconstruction failed independently: {type(exc).__name__}: {exc}"
                )
        generic = generic_key_value_entities(ocr, page_number=page.page_number)
        entities = _merge_entities(entities, generic)
        relations = _merge_relations(model_relations, infer_relations(entities))
        rule_fields, rule_warnings = extract_rule_fields(ocr, page_number=page.page_number)
        page_warnings.extend(rule_warnings)
        fields, field_warnings = merge_page_fields(model_fields, rule_fields)
        page_warnings.extend(field_warnings)
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
            "tables": model_tables,
            "warnings": page_warnings,
            "transforms": {
                "forward": transform["forward"],
                "inverse": transform["inverse"],
            },
        }
        return page_result, fields, document_type


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


def merge_page_fields(
    model_fields: Mapping[str, Any],
    rule_fields: Mapping[str, Any],
    *,
    conflict_margin: float = 0.15,
) -> tuple[dict[str, Any], list[str]]:
    """Fuse model/rule evidence and abstain on unresolved disagreement."""
    merged: dict[str, Any] = {}
    warnings: list[str] = []
    for field in sorted(set(model_fields) | set(rule_fields)):
        model_value = model_fields.get(field)
        rule_value = rule_fields.get(field)
        if model_value is None:
            merged[field] = dict(rule_value)
            continue
        if rule_value is None:
            merged[field] = dict(model_value)
            continue
        if _normalized_field_value(model_value.get("value")) == _normalized_field_value(rule_value.get("value")):
            merged[field] = dict(max(
                (model_value, rule_value),
                key=lambda item: float(item.get("confidence") or 0.0),
            ))
            continue
        model_confidence = float(model_value.get("confidence") or 0.0)
        rule_confidence = float(rule_value.get("confidence") or 0.0)
        if abs(model_confidence - rule_confidence) >= conflict_margin:
            merged[field] = dict(
                model_value if model_confidence > rule_confidence else rule_value
            )
            warnings.append(
                f"model/rule {field} conflict; selected evidence with a calibrated confidence margin"
            )
        else:
            warnings.append(
                f"model/rule {field} conflict was unresolved; abstained"
            )
    return merged, warnings


def merge_document_fields(
    values: Sequence[Mapping[str, Any]],
    *,
    conflict_margin: float = 0.10,
) -> tuple[dict[str, Any], list[str]]:
    """Aggregate page evidence, deduplicate support, and abstain on close conflicts."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for page_fields in values:
        for field, evidence in page_fields.items():
            grouped.setdefault(field, []).append(dict(evidence))
    merged: dict[str, Any] = {}
    warnings: list[str] = []
    for field, candidates in grouped.items():
        by_value: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            by_value.setdefault(
                _normalized_field_value(candidate.get("value")), []
            ).append(candidate)
        ranked = []
        for normalized, supporting in by_value.items():
            best = max(
                supporting,
                key=lambda item: float(item.get("confidence") or 0.0),
            )
            support_bonus = min(0.10, 0.03 * (len(supporting) - 1))
            score = min(1.0, float(best.get("confidence") or 0.0) + support_bonus)
            ranked.append((score, normalized, best))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < conflict_margin:
            warnings.append(
                f"conflicting page-level {field} values were too close; abstained"
            )
            continue
        merged[field] = dict(ranked[0][2])
        if len(ranked) > 1:
            warnings.append(
                f"multiple page-level {field} values; selected evidence with a confidence margin"
            )
    return merged, warnings


def _merge_relations(
    model_relations: Sequence[Mapping[str, Any]],
    rule_relations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(relation) for relation in model_relations]
    existing = {
        (relation.get("source_id"), relation.get("target_id"), relation.get("type"))
        for relation in result
    }
    for relation in rule_relations:
        key = (relation.get("source_id"), relation.get("target_id"), relation.get("type"))
        if key not in existing:
            result.append(dict(relation))
            existing.add(key)
    return result


def _merge_document_types(
    values: Sequence[Mapping[str, Any]],
) -> tuple[str, float | None, list[str]]:
    usable = [
        value for value in values
        if value.get("label") not in {None, "", "unknown"}
        and value.get("confidence") is not None
    ]
    if not usable:
        return "unknown", None, []
    grouped: dict[str, list[float]] = {}
    for value in usable:
        grouped.setdefault(str(value["label"]), []).append(float(value["confidence"]))
    ranked = sorted(
        ((statistics.fmean(scores), label) for label, scores in grouped.items()),
        reverse=True,
    )
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.10:
        return "unknown", ranked[0][0], [
            "page-level document classifications conflicted; abstained"
        ]
    warnings = ["page-level document classifications conflicted; selected confidence winner"] if len(ranked) > 1 else []
    return ranked[0][1], ranked[0][0], warnings


def _normalized_field_value(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _resolve_configured_path(cfg: Mapping[str, Any], value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else cfgmod.project_root(cfg) / path


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
