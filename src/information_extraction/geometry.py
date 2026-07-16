"""Coordinate-safe document rotation and annotation transformation."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from src.rotation_common import normalize_angle

Point = list[float]
Polygon = list[Point]


@dataclass(frozen=True)
class RotationTransform:
    """Expanded-canvas rotation represented by forward and inverse matrices."""

    angle: float
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    forward: np.ndarray
    inverse: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "angle": self.angle,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "forward": self.forward.tolist(),
            "inverse": self.inverse.tolist(),
        }


def expanded_rotation_transform(width: int, height: int, angle: float) -> RotationTransform:
    """Return the exact 3x3 transform for a visual counterclockwise rotation.

    Image coordinates use a downward-positive y axis. The matrix transforms
    original coordinates into an expanded canvas whose origin is translated so
    all transformed source corners are non-negative.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    normalized = normalize_angle(angle)
    radians = math.radians(normalized)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    linear = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    corners = np.asarray(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    centered = corners - center
    rotated = centered @ linear.T
    minimum = rotated.min(axis=0)
    maximum = rotated.max(axis=0)
    output_width = max(1, int(math.ceil(maximum[0] - minimum[0] - 1e-9)))
    output_height = max(1, int(math.ceil(maximum[1] - minimum[1] - 1e-9)))
    translation = -(linear @ center) - minimum
    forward = np.asarray(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(forward)
    return RotationTransform(
        angle=normalized,
        source_width=int(width),
        source_height=int(height),
        output_width=output_width,
        output_height=output_height,
        forward=forward,
        inverse=inverse,
    )


def apply_matrix(points: Sequence[Sequence[float]], matrix: np.ndarray) -> Polygon:
    """Transform finite 2D points with a homogeneous 3x3 matrix."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ValueError("points contain non-finite coordinates")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3x3 array")
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    transformed = homogeneous @ matrix.T
    if np.any(np.isclose(transformed[:, 2], 0.0)):
        raise ValueError("matrix maps a point to infinity")
    transformed = transformed[:, :2] / transformed[:, 2, None]
    return [[float(x), float(y)] for x, y in transformed]


def clip_polygon(polygon: Sequence[Sequence[float]], width: int, height: int) -> Polygon:
    """Clip polygon points to the expanded image boundary."""
    if width <= 0 or height <= 0:
        raise ValueError("clip dimensions must be positive")
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("polygon must contain at least four 2D points")
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height))
    return points.tolist()


def polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> list[float]:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
        raise ValueError("polygon must contain 2D points")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return [float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])]


def bbox_to_polygon(bbox: Sequence[float]) -> Polygon:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = map(float, bbox)
    if x1 < x0 or y1 < y0:
        raise ValueError("bbox maximum cannot precede minimum")
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rotate_image(image: Image.Image, transform: RotationTransform) -> Image.Image:
    """Rotate an image using the same forward matrix applied to annotations."""
    source = np.asarray(image.convert("RGB"))
    rotated = cv2.warpAffine(
        source,
        transform.forward[:2],
        (transform.output_width, transform.output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rotated, mode="RGB")


def transform_annotation(
    annotation: Mapping[str, Any], transform: RotationTransform
) -> dict[str, Any]:
    """Transform token/entity polygons without mutating the input annotation."""
    output = copy.deepcopy(dict(annotation))
    for collection_name in ("tokens", "entities"):
        for item in output.get(collection_name, []):
            polygon = item.get("polygon")
            if not polygon and item.get("bbox"):
                polygon = bbox_to_polygon(item["bbox"])
            if not polygon:
                raise ValueError(f"{collection_name} item {item.get('id', '<unknown>')} has no geometry")
            transformed = apply_matrix(polygon, transform.forward)
            clipped = clip_polygon(
                transformed, transform.output_width, transform.output_height
            )
            bbox = polygon_to_bbox(clipped)
            if bbox[2] - bbox[0] <= 1e-8 or bbox[3] - bbox[1] <= 1e-8:
                raise ValueError(f"{collection_name} item became degenerate after rotation")
            item["polygon"] = clipped
            item["bbox"] = bbox
    page = output.setdefault("page", {})
    page["width"] = transform.output_width
    page["height"] = transform.output_height
    history = output.setdefault("transformation_history", [])
    history.append(
        {
            "type": "expanded_rotation",
            "angle": transform.angle,
            "forward": transform.forward.tolist(),
            "inverse": transform.inverse.tolist(),
        }
    )
    return output


def rotate_image_and_annotation(
    image: Image.Image, annotation: Mapping[str, Any], angle: float
) -> tuple[Image.Image, dict[str, Any], RotationTransform]:
    transform = expanded_rotation_transform(image.width, image.height, angle)
    return rotate_image(image, transform), transform_annotation(annotation, transform), transform


class DynamicRotation:
    """Deterministic per-example sampler for training-time rotation."""

    def __init__(
        self,
        *,
        seed: int = 42,
        upright_probability: float = 0.2,
        angle_min: float = 0.0,
        angle_max: float = 360.0,
    ) -> None:
        if not 0.0 <= upright_probability <= 1.0:
            raise ValueError("upright_probability must be in [0, 1]")
        if not angle_max > angle_min:
            raise ValueError("angle_max must exceed angle_min")
        self.seed = int(seed)
        self.upright_probability = float(upright_probability)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)

    def angle_for(self, example_id: str, epoch: int = 0) -> float:
        material = f"{self.seed}|{epoch}|{example_id}"
        local_seed = int.from_bytes(material.encode("utf-8"), "little") % (2**63 - 1)
        rng = random.Random(local_seed)
        if rng.random() < self.upright_probability:
            return 0.0
        return normalize_angle(rng.uniform(self.angle_min, self.angle_max))

    def apply(
        self,
        image: Image.Image,
        annotation: Mapping[str, Any],
        *,
        example_id: str,
        epoch: int = 0,
    ) -> tuple[Image.Image, dict[str, Any], RotationTransform]:
        angle = self.angle_for(example_id, epoch)
        rotated_image, rotated_annotation, transform = rotate_image_and_annotation(
            image, annotation, angle
        )
        rotated_annotation["augmentation"] = {
            "seed": self.seed,
            "epoch": int(epoch),
            "example_id": str(example_id),
            "angle": angle,
        }
        return rotated_image, rotated_annotation, transform
