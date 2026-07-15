from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.angle_estimation import apply_angle_correction, estimate_exact_angle
from src.rotation_common import circular_angular_error, get_rotation_zone, normalize_angle


def _asymmetric_document() -> Image.Image:
    image = Image.new("L", (360, 220), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 18, 335, 202), outline=0, width=3)
    draw.rectangle((38, 34, 175, 52), fill=0)
    draw.line((38, 80, 318, 80), fill=0, width=5)
    draw.line((38, 112, 275, 112), fill=0, width=4)
    draw.line((38, 145, 210, 145), fill=0, width=4)
    draw.rectangle((265, 150, 318, 187), outline=0, width=4)
    return image


@pytest.mark.parametrize("angle", [1, 45, 89, 90, 135, 180, 225, 270, 315, 359])
def test_zone_guided_angle_search_handles_boundaries_and_wraparound(angle):
    rotated = _asymmetric_document().rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=255,
    )
    result = estimate_exact_angle(
        np.asarray(rotated),
        get_rotation_zone(angle),
        {
            "coarse_step_degrees": 2.0,
            "fine_step_degrees": 0.25,
            "fine_window_degrees": 3.0,
            "scoring_size": 128,
            "reliability_threshold": 0.0,
            "minimum_ink_fraction": 0.001,
            "minimum_edge_fraction": 0.001,
        },
    )

    assert result["estimated_angle"] is not None
    assert 0.0 <= result["estimated_angle"] < 360.0
    assert result["method"] == "combined_hough_projection_gradient_min_area_rect"
    assert isinstance(result["fallback_used"], bool)
    assert isinstance(result["reliable"], bool)
    assert circular_angular_error(result["estimated_angle"], angle) <= 2.0
    assert result["correction_angle"] == pytest.approx(-result["estimated_angle"])
    assert normalize_angle(result["estimated_angle"] + result["correction_angle"]) == pytest.approx(0)


def test_angle_search_reports_blank_page_failure_explicitly():
    result = estimate_exact_angle(np.full((128, 128), 255, dtype=np.uint8), 1)
    assert result["status"] == "failure"
    assert result["estimated_angle"] is None
    assert result["failure_reason"] == "insufficient_ink"
    assert result["fallback_used"] is True
    assert result["reliable"] is False


def test_predicted_zone_restricts_search_and_resolves_quadrant_ambiguity():
    rotated = _asymmetric_document().rotate(45, expand=True, fillcolor=255)
    search = {"coarse_step_degrees": 5.0, "fine_step_degrees": 1.0, "reliability_threshold": 0.0}
    zone_one = estimate_exact_angle(np.asarray(rotated), 1, search)
    zone_three = estimate_exact_angle(np.asarray(rotated), 3, search)
    assert 0.0 <= zone_one["estimated_angle"] < 90.0
    assert 180.0 <= zone_three["estimated_angle"] < 270.0
    assert circular_angular_error(zone_one["estimated_angle"], zone_three["estimated_angle"]) >= 170.0


def test_pixel_correction_restores_horizontal_long_axis():
    source = np.full((80, 240), 255, dtype=np.uint8)
    source[30:50, 25:215] = 0
    vertical = np.asarray(
        Image.fromarray(source).rotate(90, expand=True, fillcolor=255)
    )
    corrected = apply_angle_correction(vertical, 90, expand=True)
    points = np.argwhere(corrected < 128)
    height = int(points[:, 0].max() - points[:, 0].min() + 1)
    width = int(points[:, 1].max() - points[:, 1].min() + 1)
    assert width > height * 4
