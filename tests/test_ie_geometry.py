from __future__ import annotations

import copy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.information_extraction.geometry import (
    DynamicRotation,
    apply_matrix,
    expanded_rotation_transform,
    rotate_image_and_annotation,
)


def _annotation(width: int = 40, height: int = 20) -> dict:
    return {
        "page": {"width": width, "height": height},
        "tokens": [
            {
                "id": "t1",
                "text": "x",
                "polygon": [[5, 4], [15, 4], [15, 10], [5, 10]],
                "bbox": [5, 4, 15, 10],
                "label": "KEY",
            }
        ],
        "entities": [],
        "relations": [{"id": "r1", "source_id": "t1", "target_id": "t1"}],
    }


@pytest.mark.parametrize("angle", [0, 1, 45, 89, 90, 180, 270, 315, -90])
def test_forward_inverse_round_trip(angle: float) -> None:
    transform = expanded_rotation_transform(40, 20, angle)
    points = [[0, 0], [40, 0], [40, 20], [0, 20], [12.5, 7.25]]
    restored = apply_matrix(apply_matrix(points, transform.forward), transform.inverse)
    assert np.asarray(restored) == pytest.approx(np.asarray(points), abs=1e-7)


def test_cardinal_dimensions_and_identity() -> None:
    identity = expanded_rotation_transform(40, 20, 0)
    assert (identity.output_width, identity.output_height) == (40, 20)
    assert identity.forward == pytest.approx(np.eye(3), abs=1e-10)
    quarter = expanded_rotation_transform(40, 20, 90)
    assert (quarter.output_width, quarter.output_height) == (20, 40)


def test_rotation_preserves_input_relations_and_uses_white_canvas() -> None:
    image = Image.new("RGB", (40, 20), "white")
    ImageDraw.Draw(image).rectangle((5, 4, 15, 10), fill="black")
    source_annotation = _annotation()
    before = copy.deepcopy(source_annotation)
    rotated, annotation, transform = rotate_image_and_annotation(
        image, source_annotation, 45
    )
    assert source_annotation == before
    assert annotation["relations"] == before["relations"]
    assert rotated.size == (transform.output_width, transform.output_height)
    pixels = np.asarray(rotated)
    assert (pixels[0, 0] >= 250).all()
    x0, y0, x1, y1 = annotation["tokens"][0]["bbox"]
    assert 0 <= x0 < x1 <= rotated.width
    assert 0 <= y0 < y1 <= rotated.height


def test_dynamic_rotation_is_seeded_and_records_provenance() -> None:
    image = Image.new("RGB", (40, 20), "white")
    augmenter = DynamicRotation(seed=7, upright_probability=0.0)
    first = augmenter.apply(image, _annotation(), example_id="page-a", epoch=3)
    second = augmenter.apply(image, _annotation(), example_id="page-a", epoch=3)
    assert first[2].angle == second[2].angle
    assert first[1]["augmentation"] == second[1]["augmentation"]
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
