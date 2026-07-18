"""Exact PaddleOCR model identity and artifact-manifest validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.ocr.errors import OCRModelMismatch, OCRModelUnavailable
from src.rotation_common import sha256_file

DETECTOR_MODEL = "PP-OCRv6_medium_det"
GENERAL_RECOGNIZER_MODEL = "PP-OCRv6_medium_rec"
THAI_RECOGNIZER_MODEL = "th_PP-OCRv5_mobile_rec"
REQUIRED_MODEL_NAMES = (
    DETECTOR_MODEL,
    GENERAL_RECOGNIZER_MODEL,
    THAI_RECOGNIZER_MODEL,
)


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    role: str
    language: str
    path: Path
    files: tuple[dict[str, Any], ...]

    @property
    def artifact_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for item in sorted(self.files, key=lambda value: str(value["path"])):
            digest.update(str(item["path"]).encode("utf-8"))
            digest.update(str(item["sha256"]).encode("ascii"))
        return digest.hexdigest()


class ModelRegistry:
    """Load only verified model identities from model_setup.json."""

    def __init__(self, artifacts: Mapping[str, ModelArtifact], metadata: Mapping[str, Any]) -> None:
        self.artifacts = dict(artifacts)
        self.metadata = dict(metadata)

    @classmethod
    def from_setup(cls, path: str | Path, *, verify_hashes: bool = True) -> "ModelRegistry":
        setup_path = Path(path)
        if not setup_path.is_file():
            raise OCRModelUnavailable(f"model setup manifest not found: {setup_path}")
        payload = json.loads(setup_path.read_text(encoding="utf-8"))
        models = payload.get("models")
        if not isinstance(models, Mapping):
            raise OCRModelMismatch("model setup manifest has no models mapping")
        artifacts: dict[str, ModelArtifact] = {}
        for expected in REQUIRED_MODEL_NAMES:
            item = models.get(expected)
            if not isinstance(item, Mapping):
                raise OCRModelUnavailable(f"required OCR model missing from registry: {expected}")
            if item.get("requested_name") != expected or item.get("resolved_name") != expected:
                raise OCRModelMismatch(f"model identity mismatch for {expected}")
            model_path = Path(str(item.get("resolved_path", ""))).expanduser()
            if not model_path.is_absolute():
                model_path = (setup_path.parent / model_path).resolve()
            if not model_path.is_dir():
                raise OCRModelUnavailable(f"required OCR model directory not found: {model_path}")
            files = item.get("files")
            if not isinstance(files, list) or not files:
                raise OCRModelMismatch(f"model registry has no files for {expected}")
            for file_record in files:
                relative = Path(str(file_record.get("path", "")))
                artifact_file = model_path / relative
                if not artifact_file.is_file():
                    raise OCRModelUnavailable(f"registered model file missing: {artifact_file}")
                if verify_hashes and sha256_file(artifact_file) != file_record.get("sha256"):
                    raise OCRModelMismatch(f"checksum mismatch: {artifact_file}")
            artifacts[expected] = ModelArtifact(
                name=expected,
                role=str(item.get("role", "unknown")),
                language=str(item.get("language", "unknown")),
                path=model_path,
                files=tuple(dict(record) for record in files),
            )
        return cls(artifacts, payload)

    def require(self, name: str) -> ModelArtifact:
        if name not in REQUIRED_MODEL_NAMES:
            raise OCRModelMismatch(f"unapproved OCR model requested: {name}")
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise OCRModelUnavailable(f"required OCR model is unavailable: {name}") from exc

    def route_models(self, route: str) -> tuple[ModelArtifact, ModelArtifact]:
        if route == "general":
            recognizer = GENERAL_RECOGNIZER_MODEL
        elif route == "thai":
            recognizer = THAI_RECOGNIZER_MODEL
        else:
            raise ValueError(f"unsupported OCR route: {route}")
        return self.require(DETECTOR_MODEL), self.require(recognizer)
