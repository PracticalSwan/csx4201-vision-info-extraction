"""Size-preserving, bounded OCR preprocessing profiles and quality routing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


PREPROCESSING_PROFILES = (
    "original",
    "grayscale_normalized",
    "adaptive_contrast",
    "denoise",
    "sharpen",
    "background_normalized",
    "quality_auto",
)


def image_quality_features(image: Image.Image) -> dict[str, float]:
    gray = ImageOps.grayscale(image)
    stats = ImageStat.Stat(gray)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return {
        "mean_luminance": float(stats.mean[0]),
        "contrast_stddev": float(stats.stddev[0]),
        "edge_mean": float(ImageStat.Stat(edges).mean[0]),
    }


def choose_preprocessing_profile(features: Mapping[str, Any]) -> str:
    """Choose enhancement only for objectively low-quality inputs."""
    contrast = float(features.get("contrast_stddev", 0.0))
    luminance = float(features.get("mean_luminance", 0.0))
    edge_mean = float(features.get("edge_mean", 0.0))
    if contrast < 22.0 or luminance < 55.0 or luminance > 235.0:
        return "background_normalized"
    if edge_mean < 8.0:
        return "sharpen"
    return "original"


def preprocess_for_ocr(
    image: Image.Image, profile: str = "original"
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply one auditable profile without changing geometry or image size."""
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(f"unsupported OCR preprocessing profile: {profile}")
    source = image.convert("RGB")
    features = image_quality_features(source)
    selected = choose_preprocessing_profile(features) if profile == "quality_auto" else profile
    if selected == "original":
        output = source.copy()
    elif selected == "grayscale_normalized":
        output = ImageOps.grayscale(source).convert("RGB")
    elif selected == "adaptive_contrast":
        output = ImageOps.autocontrast(source, cutoff=1)
    elif selected == "denoise":
        output = source.filter(ImageFilter.MedianFilter(size=3))
    elif selected == "sharpen":
        output = ImageEnhance.Sharpness(source).enhance(1.8)
    elif selected == "background_normalized":
        normalized = ImageOps.autocontrast(ImageOps.grayscale(source), cutoff=2)
        output = normalized.convert("RGB")
    else:  # pragma: no cover - profile validation makes this unreachable
        raise AssertionError(selected)
    if output.size != source.size:
        raise RuntimeError("OCR preprocessing must preserve image geometry")
    return output, {
        "requested_profile": profile,
        "selected_profile": selected,
        "quality_features": features,
        "geometry_preserved": True,
    }
