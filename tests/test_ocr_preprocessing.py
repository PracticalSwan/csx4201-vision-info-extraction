from __future__ import annotations

from PIL import Image

from src.ocr.preprocessing import (
    choose_preprocessing_profile,
    preprocess_for_ocr,
)


def test_clean_document_stays_original_under_quality_auto() -> None:
    image = Image.new("RGB", (100, 40), "white")
    for x in range(10, 90):
        for y in range(15, 25):
            image.putpixel((x, y), (0, 0, 0))

    output, decision = preprocess_for_ocr(image, "quality_auto")

    assert output.size == image.size
    assert decision["selected_profile"] == "original"
    assert decision["geometry_preserved"] is True


def test_low_contrast_document_uses_background_normalization() -> None:
    assert choose_preprocessing_profile({
        "contrast_stddev": 5.0, "mean_luminance": 180.0, "edge_mean": 3.0,
    }) == "background_normalized"
    image = Image.new("RGB", (20, 20), (180, 180, 180))
    output, decision = preprocess_for_ocr(image, "quality_auto")
    assert output.mode == "RGB"
    assert decision["selected_profile"] == "background_normalized"


def test_unknown_profile_is_rejected() -> None:
    try:
        preprocess_for_ocr(Image.new("RGB", (10, 10)), "invented")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown preprocessing profile was accepted")
