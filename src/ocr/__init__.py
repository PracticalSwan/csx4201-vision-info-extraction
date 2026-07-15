"""Strict multilingual PaddleOCR integration and candidate selection."""

from .model_registry import REQUIRED_MODEL_NAMES, ModelRegistry
from .pipeline import MultilingualOCR

__all__ = ["REQUIRED_MODEL_NAMES", "ModelRegistry", "MultilingualOCR"]
