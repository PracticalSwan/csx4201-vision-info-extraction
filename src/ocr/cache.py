"""Provenance-complete OCR cache stored outside the repository."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.ocr.errors import OCRCacheError
from src.rotation_common import atomic_write_json, canonical_json, configuration_hash, sha256_file


@dataclass(frozen=True)
class OCRCacheKey:
    input_sha256: str
    detector_model: str
    detector_artifact_hash: str
    recognizer_model: str
    recognizer_artifact_hash: str
    language_route_configuration: Mapping[str, Any]
    orientation_configuration: Mapping[str, Any]
    paddleocr_version: str
    preprocessing_version: str

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        **values: Any,
    ) -> "OCRCacheKey":
        return cls(input_sha256=sha256_file(Path(image_path)), **values)

    @property
    def digest(self) -> str:
        return configuration_hash(self.as_dict(), length=64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_sha256": self.input_sha256,
            "detector_model": self.detector_model,
            "detector_artifact_hash": self.detector_artifact_hash,
            "recognizer_model": self.recognizer_model,
            "recognizer_artifact_hash": self.recognizer_artifact_hash,
            "language_route_configuration": json.loads(canonical_json(self.language_route_configuration)),
            "orientation_configuration": json.loads(canonical_json(self.orientation_configuration)),
            "paddleocr_version": self.paddleocr_version,
            "preprocessing_version": self.preprocessing_version,
        }


class OCRCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: OCRCacheKey) -> Path:
        return self.root / key.digest[:2] / f"{key.digest}.json"

    def get(self, key: OCRCacheKey, *, private: bool = False) -> dict[str, Any] | None:
        # Private OCR text is intentionally not read from or written to the
        # reusable public cache. A separate explicit private-output workflow
        # may retain it under the ignored private root.
        if private:
            return None
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OCRCacheError(f"corrupt OCR cache entry: {path}") from exc
        if payload.get("cache_key") != key.as_dict() or payload.get("cache_digest") != key.digest:
            raise OCRCacheError(f"stale OCR cache provenance: {path}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise OCRCacheError(f"OCR cache result is not an object: {path}")
        return result

    def put(
        self, key: OCRCacheKey, result: Mapping[str, Any], *, private: bool = False
    ) -> Path | None:
        if private:
            return None
        path = self.path_for(key)
        atomic_write_json(path, {
            "cache_version": "1.0",
            "cache_digest": key.digest,
            "cache_key": key.as_dict(),
            "result": dict(result),
        })
        return path
