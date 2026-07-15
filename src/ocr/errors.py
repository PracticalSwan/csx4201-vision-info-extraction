"""OCR-specific failures with no silent model fallback."""


class OCRError(RuntimeError):
    """Base OCR pipeline failure."""


class OCRModelUnavailable(OCRError):
    """A required exact OCR model is absent or invalid."""


class OCRModelMismatch(OCRError):
    """Resolved artifacts do not match the requested model identity."""


class OCRInferenceError(OCRError):
    """PaddleOCR failed to return a normalizable result."""


class OCRCacheError(OCRError):
    """Cached output is corrupt or has stale provenance."""
