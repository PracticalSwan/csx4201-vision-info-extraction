from __future__ import annotations

import math

import pytest

from src.rotation_common import (
    circular_angular_error,
    get_rotation_zone,
    normalize_angle,
    rotation_filename,
    signed_correction_angle,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (360, 0.0),
        (720, 0.0),
        (-360, 0.0),
        (-1, 359.0),
        (361.5, 1.5),
        (-450, 270.0),
    ],
)
def test_normalize_angle_uses_half_open_circle(value, expected):
    assert normalize_angle(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_normalize_angle_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        normalize_angle(value)


@pytest.mark.parametrize(
    ("angle", "zone"),
    [
        (0, 1),
        (89.999, 1),
        (90, 2),
        (179.999, 2),
        (180, 3),
        (269.999, 3),
        (270, 4),
        (359.999, 4),
        (360, 1),
        (-1, 4),
    ],
)
def test_rotation_zones_use_exact_half_open_boundaries(angle, zone):
    assert get_rotation_zone(angle) == zone


@pytest.mark.parametrize(
    ("predicted", "truth", "error"),
    [(359, 1, 2), (1, 359, 2), (0, 180, 180), (45, 405, 0), (-10, 10, 20)],
)
def test_circular_error_uses_shortest_arc(predicted, truth, error):
    assert circular_angular_error(predicted, truth) == pytest.approx(error)


def test_correction_angle_has_opposite_sign_under_ccw_convention():
    assert signed_correction_angle(0) == 0
    assert signed_correction_angle(45) == -45
    assert signed_correction_angle(359) == -359
    assert (normalize_angle(45 + signed_correction_angle(45))) == pytest.approx(0)
    assert (normalize_angle(359 + signed_correction_angle(359))) == pytest.approx(0)


def test_rotation_filename_records_normalized_angle_and_exact_zone():
    assert rotation_filename("page_a", 90) == "page_a_angle_090_zone_2.png"
    assert rotation_filename("page_a", -1) == "page_a_angle_359_zone_4.png"
