"""Safe local image/PDF loading for document inference."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class DocumentInputError(ValueError):
    """Raised when a local input cannot safely be decoded."""


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    image: Image.Image


def load_document_pages(
    path: str | Path,
    *,
    max_pages: int | None = None,
    pdf_dpi: int = 200,
    max_pixels: int = 60_000_000,
) -> tuple[str, str, list[DocumentPage]]:
    source = Path(path)
    if not source.is_file():
        raise DocumentInputError(f"input file does not exist: {source}")
    if source.stat().st_size == 0:
        raise DocumentInputError(f"input file is empty: {source}")
    extension = source.suffix.lower()
    if extension == ".pdf":
        pages = _load_pdf(source, max_pages=max_pages, dpi=pdf_dpi, max_pixels=max_pixels)
        source_type = "pdf"
    elif extension in SUPPORTED_IMAGE_EXTENSIONS:
        pages = [_load_image(source, max_pixels=max_pixels)]
        source_type = "image"
    else:
        raise DocumentInputError(
            f"unsupported input extension {source.suffix!r}; expected PDF or {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    document_id = "document_" + _sha256(source)[:16]
    return document_id, source_type, pages


def normalize_image(image: Image.Image, *, max_pixels: int = 60_000_000) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.width < 2 or image.height < 2:
        raise DocumentInputError(f"image is too small for OCR: {image.width}x{image.height}")
    if image.width * image.height > max_pixels:
        raise DocumentInputError(
            f"image has {image.width * image.height:,} pixels; limit is {max_pixels:,}"
        )
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image.copy()


def _load_image(path: Path, *, max_pixels: int) -> DocumentPage:
    try:
        with Image.open(path) as image:
            image.load()
            normalized = normalize_image(image, max_pixels=max_pixels)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DocumentInputError(f"image cannot be decoded: {path.name}: {exc}") from exc
    return DocumentPage(page_number=1, image=normalized)


def _load_pdf(
    path: Path, *, max_pages: int | None, dpi: int, max_pixels: int
) -> list[DocumentPage]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise DocumentInputError("PyMuPDF is required for PDF input") from exc
    if dpi < 72 or dpi > 600:
        raise DocumentInputError("PDF DPI must be between 72 and 600")
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise DocumentInputError(f"PDF cannot be opened: {path.name}: {exc}") from exc
    try:
        if document.needs_pass:
            raise DocumentInputError("encrypted or password-protected PDF is unsupported")
        if document.page_count < 1:
            raise DocumentInputError("PDF contains no pages")
        count = document.page_count if max_pages is None else min(document.page_count, max_pages)
        scale = dpi / 72.0
        pages: list[DocumentPage] = []
        for index in range(count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append(DocumentPage(index + 1, normalize_image(image, max_pixels=max_pixels)))
        return pages
    except DocumentInputError:
        raise
    except Exception as exc:
        raise DocumentInputError(f"PDF rendering failed: {path.name}: {exc}") from exc
    finally:
        document.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
