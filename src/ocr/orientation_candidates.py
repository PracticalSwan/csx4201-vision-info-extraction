"""K-Means-independent cardinal and optional continuous OCR candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image

from src.information_extraction.geometry import (
    RotationTransform,
    apply_matrix,
    expanded_rotation_transform,
    polygon_to_bbox,
    rotate_image,
)


@dataclass(frozen=True)
class OrientationCandidate:
    angle: float
    image: Image.Image
    transform: RotationTransform
    kind: str


def build_orientation_candidates(
    image: Image.Image,
    *,
    cardinal_angles: Iterable[float] = (0, 90, 180, 270),
    deskew_angle: float | None = None,
) -> list[OrientationCandidate]:
    """Build candidates without accepting or consulting any K-Means output."""
    candidates: list[OrientationCandidate] = []
    seen: set[float] = set()
    values = [(float(angle), "cardinal") for angle in cardinal_angles]
    if deskew_angle is not None:
        values.append((float(deskew_angle), "continuous_deskew"))
    for angle, kind in values:
        transform = expanded_rotation_transform(image.width, image.height, angle)
        if transform.angle in seen:
            continue
        seen.add(transform.angle)
        candidates.append(OrientationCandidate(
            angle=transform.angle,
            image=rotate_image(image, transform),
            transform=transform,
            kind=kind,
        ))
    if 0.0 not in seen:
        transform = expanded_rotation_transform(image.width, image.height, 0.0)
        candidates.insert(0, OrientationCandidate(0.0, image.copy(), transform, "original"))
    return candidates


def restore_original_coordinates(
    result: dict, candidate: OrientationCandidate
) -> dict:
    """Inverse-map candidate OCR polygons into the original page coordinate system."""
    restored = dict(result)
    restored_words = []
    for word in result.get("words", []):
        item = dict(word)
        polygon = apply_matrix(item["polygon"], candidate.transform.inverse)
        polygon = [
            [
                max(0.0, min(float(candidate.transform.source_width), float(x))),
                max(0.0, min(float(candidate.transform.source_height), float(y))),
            ]
            for x, y in polygon
        ]
        item["polygon"] = polygon
        item["bbox"] = polygon_to_bbox(polygon)
        restored_words.append(item)
    restored["words"] = restored_words
    by_id = {word["id"]: word for word in restored_words}
    restored_lines = []
    for line in result.get("lines", []):
        item = dict(line)
        line_words = [by_id[word_id] for word_id in item.get("word_ids", []) if word_id in by_id]
        if line_words:
            item["polygon"] = line_words[0]["polygon"]
            item["bbox"] = line_words[0]["bbox"]
        restored_lines.append(item)
    restored["lines"] = restored_lines
    restored["candidate_transform"] = candidate.transform.as_dict()
    return restored
