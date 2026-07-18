"""Lazy PaddleOCR 3.7 adapter with explicit exact local model paths."""
from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from src.ocr.errors import OCRInferenceError, OCRModelMismatch
from src.ocr.environment import configure_external_environment
from src.ocr.model_registry import ModelRegistry
from src.ocr.result_normalizer import normalize_paddle_result


class PaddleOCRAdapter:
    """One explicit detector+recognizer route; no automatic substitution."""

    def __init__(
        self,
        registry: ModelRegistry,
        route: str,
        *,
        device: str = "cpu",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        enable_mkldnn: bool | None = None,
    ) -> None:
        self.registry = registry
        self.route = route
        self.device = device
        self.detector, self.recognizer = registry.route_models(route)
        self.options = {
            "use_doc_orientation_classify": bool(use_doc_orientation_classify),
            "use_doc_unwarping": bool(use_doc_unwarping),
            "use_textline_orientation": bool(use_textline_orientation),
            # PaddleOCR enables oneDNN/MKLDNN on CPU by default. PaddlePaddle
            # 3.3's Linux executor rejects an ArrayAttribute used by the pinned
            # PP-OCRv6 artifacts, so prefer the portable plain-CPU path.
            "enable_mkldnn": (
                not str(device).casefold().startswith("cpu")
                if enable_mkldnn is None
                else bool(enable_mkldnn)
            ),
        }
        self._pipeline: Any = None

    @property
    def paddleocr_version(self) -> str:
        return importlib.metadata.version("paddleocr")

    def initialize(self) -> None:
        if self._pipeline is not None:
            return
        configure_external_environment()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on external env
            raise OCRInferenceError(f"PaddleOCR or a required dependency failed to import: {exc}") from exc
        kwargs = {
            "text_detection_model_name": self.detector.name,
            "text_detection_model_dir": str(self.detector.path),
            "text_recognition_model_name": self.recognizer.name,
            "text_recognition_model_dir": str(self.recognizer.path),
            "device": self.device,
            **self.options,
        }
        try:
            self._pipeline = PaddleOCR(**kwargs)
        except TypeError as exc:
            raise OCRInferenceError(
                "installed PaddleOCR constructor is incompatible with the verified 3.7 adapter arguments"
            ) from exc
        model_names = str(getattr(self._pipeline, "paddlex_config", "")) + repr(self._pipeline)
        # Constructor paths are authoritative. If the installed object exposes
        # its configuration, a conflicting model name is a hard failure.
        for expected in (self.detector.name, self.recognizer.name):
            if model_names and "model_name" in model_names and expected not in model_names:
                raise OCRModelMismatch(f"initialized PaddleOCR pipeline does not expose expected model {expected}")

    def predict(self, image: Image.Image, *, orientation: float = 0.0) -> dict[str, Any]:
        self.initialize()
        started = time.perf_counter()
        try:
            raw = self._pipeline.predict(np.asarray(image.convert("RGB")))
            result = list(raw)
        except Exception as exc:  # pragma: no cover - external engine details
            raise OCRInferenceError(f"PaddleOCR {self.route} inference failed: {exc}") from exc
        return normalize_paddle_result(
            result,
            detector_model=self.detector.name,
            recognizer_model=self.recognizer.name,
            route=self.route,
            orientation=orientation,
            duration_seconds=time.perf_counter() - started,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "detector_model": self.detector.name,
            "detector_artifact_hash": self.detector.artifact_hash,
            "recognizer_model": self.recognizer.name,
            "recognizer_artifact_hash": self.recognizer.artifact_hash,
            "paddleocr_version": self.paddleocr_version,
            "device": self.device,
            **self.options,
        }
