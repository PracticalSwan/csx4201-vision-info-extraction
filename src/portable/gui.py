"""Local-only Gradio interface for OCR and information extraction."""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .api import ExtractionError, run_extraction
from .results import (
    create_result_archive,
    field_rows,
    ocr_text,
    visualization_files,
)
from .runtime import RuntimeSettings


FIELD_HEADERS = ["Field", "Value", "Confidence", "Method", "Page", "Validation"]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PREVIEW_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
PREVIEW_MAX_PIXELS = 12_000_000
PREVIEW_PDF_DPI = 120
APP_CSS = """
.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
}

#document-upload {
    min-height: 5.75rem;
}

#document-preview {
    min-height: 25rem;
}

#document-preview img {
    object-fit: contain !important;
}

#ocr-text textarea,
#run-log textarea {
    overflow-y: scroll !important;
    overflow-x: auto !important;
    overscroll-behavior: contain;
    resize: none !important;
    scrollbar-gutter: stable;
}

#ocr-text textarea {
    height: 27rem !important;
    min-height: 27rem !important;
    max-height: 27rem !important;
}

#run-log textarea {
    height: 20rem !important;
    min-height: 20rem !important;
    max-height: 20rem !important;
}

#ocr-text textarea::-webkit-scrollbar,
#run-log textarea::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

#ocr-text textarea::-webkit-scrollbar-thumb,
#run-log textarea::-webkit-scrollbar-thumb {
    background: rgba(127, 127, 127, 0.68);
    border: 3px solid transparent;
    border-radius: 8px;
    background-clip: content-box;
}

@media (max-width: 780px) {
    #document-preview {
        min-height: 18rem;
    }

    #ocr-text textarea {
        height: 20rem !important;
        min-height: 20rem !important;
        max-height: 20rem !important;
    }

    #run-log textarea {
        height: 14rem !important;
        min-height: 14rem !important;
        max-height: 14rem !important;
    }
}
"""


def _normalize_preview_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.width < 2 or image.height < 2:
        raise ValueError(f"image is too small: {image.width}x{image.height}")
    pixels = image.width * image.height
    if pixels > PREVIEW_MAX_PIXELS:
        scale = math.sqrt(PREVIEW_MAX_PIXELS / pixels)
        size = (
            max(2, int(image.width * scale)),
            max(2, int(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image.copy()


def _preview_pdf(source: Path) -> Image.Image:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - protected by app requirements
        raise ValueError("PyMuPDF is required for PDF preview") from exc
    try:
        document = fitz.open(source)
    except Exception as exc:
        raise ValueError(f"PDF cannot be opened: {exc}") from exc
    try:
        if document.needs_pass:
            raise ValueError("encrypted or password-protected PDF is unsupported")
        if document.page_count < 1:
            raise ValueError("PDF contains no pages")
        page = document.load_page(0)
        scale = PREVIEW_PDF_DPI / 72.0
        estimated_pixels = page.rect.width * scale * page.rect.height * scale
        if estimated_pixels > PREVIEW_MAX_PIXELS:
            scale *= math.sqrt(PREVIEW_MAX_PIXELS / estimated_pixels)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        return _normalize_preview_image(image)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF preview failed: {exc}") from exc
    finally:
        document.close()


def _preview_document(uploaded: str | None):
    if not uploaded:
        return None, "Upload an image or PDF to preview it here."
    source = Path(uploaded)
    try:
        if not source.is_file():
            raise ValueError("uploaded file no longer exists")
        if source.suffix.casefold() == ".pdf":
            preview = _preview_pdf(source)
            preview_kind = "first PDF page"
        elif source.suffix.casefold() in PREVIEW_IMAGE_SUFFIXES:
            with Image.open(source) as image:
                image.load()
                preview = _normalize_preview_image(image)
            preview_kind = "image"
        else:
            raise ValueError(f"unsupported file type: {source.suffix or '(none)'}")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, f"Preview unavailable: {exc}"
    return preview, f"**{source.name}** · Previewing the {preview_kind} locally."


def _on_document_change(uploaded: str | None):
    preview, preview_note = _preview_document(uploaded)
    if uploaded:
        status = (
            f"Ready to extract **{Path(uploaded).name}**. "
            "Select **Extract document** to run the local model."
        )
    else:
        status = "Choose a document, then select **Extract document**."
    return preview, preview_note, status, [], "", None, [], None, ""


def _clean_log_line(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line)


def _run_gui(
    uploaded: str | None,
    language: str,
    device: str,
    max_pages: float | None,
    *,
    settings: RuntimeSettings,
):
    if not uploaded:
        raise ValueError("Choose an image or PDF first.")
    log_tail: list[str] = []

    def on_log(line: str) -> None:
        log_tail.append(_clean_log_line(line))
        del log_tail[:-200]

    try:
        run = run_extraction(
            uploaded,
            settings=settings,
            language=language,
            device=device,
            max_pages=int(max_pages) if max_pages else None,
            on_log=on_log,
        )
    except ExtractionError as exc:
        raise RuntimeError(str(exc)) from exc
    archive = create_result_archive(run.output_dir)
    pages = len(run.payload.get("pages") or [])
    display_output = (
        Path(settings.output_root.name or "outputs") / run.output_dir.name
    ).as_posix()
    status = (
        f"### Complete\n"
        f"Processed **{pages} page{'s' if pages != 1 else ''}** locally. "
        f"Results are stored locally under `{display_output}`."
    )
    return (
        status,
        field_rows(run.payload),
        ocr_text(run.payload),
        run.payload,
        visualization_files(run.output_dir),
        str(archive),
        "\n".join(log_tail),
    )


def build_app(settings: RuntimeSettings | None = None):
    import gradio as gr

    runtime = settings or RuntimeSettings.load()
    with gr.Blocks(
        title="OCR Model — Local Document Extraction",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# OCR Model\n"
            "Run the finished OCR + LayoutXLM information-extraction pipeline on "
            "an image or PDF. Processing and outputs stay on this computer. "
            "**No OpenAI API key is used.**"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                uploaded = gr.File(
                    label="Document",
                    file_types=[
                        ".pdf",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".tif",
                        ".tiff",
                        ".bmp",
                        ".webp",
                    ],
                    type="filepath",
                    height=112,
                    elem_id="document-upload",
                )
                preview = gr.Image(
                    label="Document preview",
                    interactive=False,
                    height=400,
                    buttons=["fullscreen"],
                    placeholder="Upload an image or PDF to preview it here.",
                    elem_id="document-preview",
                )
                preview_note = gr.Markdown(
                    "Upload an image or PDF to preview it here.",
                    elem_id="document-preview-note",
                )
            with gr.Column(scale=4, min_width=320):
                language = gr.Dropdown(
                    choices=["auto", "general", "thai", "en", "tr", "th"],
                    value="auto",
                    label="Language",
                )
                device = gr.Dropdown(
                    choices=["cpu", "gpu:0"],
                    value=runtime.device if runtime.device in {"cpu", "gpu:0"} else "cpu",
                    label="Device",
                )
                max_pages = gr.Number(
                    value=0,
                    minimum=0,
                    precision=0,
                    label="Maximum PDF pages (0 = all)",
                )
                run_button = gr.Button("Extract document", variant="primary")
        status = gr.Markdown("Choose a document, then select **Extract document**.")
        with gr.Tabs():
            with gr.Tab("Extracted fields"):
                fields = gr.Dataframe(
                    headers=FIELD_HEADERS,
                    datatype=["str", "str", "number", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("OCR text"):
                text = gr.Textbox(
                    label="OCR text",
                    lines=18,
                    max_lines=18,
                    interactive=False,
                    autoscroll=False,
                    buttons=["copy"],
                    elem_id="ocr-text",
                )
            with gr.Tab("Full JSON"):
                result_json = gr.JSON()
            with gr.Tab("Visual confirmation"):
                gallery = gr.Gallery(
                    label="Page overlays",
                    columns=2,
                    object_fit="contain",
                    height=600,
                )
            with gr.Tab("Run log"):
                log = gr.Textbox(
                    label="Run log",
                    lines=12,
                    max_lines=12,
                    interactive=False,
                    autoscroll=False,
                    elem_id="run-log",
                )
        archive = gr.File(
            label="Download complete local result (.zip)",
            interactive=False,
            height=90,
        )

        def run_handler(
            uploaded_value: str | None,
            language_value: str,
            device_value: str,
            max_pages_value: float | None,
        ):
            return _run_gui(
                uploaded_value,
                language_value,
                device_value,
                max_pages_value,
                settings=runtime,
            )

        uploaded.change(
            _on_document_change,
            inputs=uploaded,
            outputs=[
                preview,
                preview_note,
                status,
                fields,
                text,
                result_json,
                gallery,
                archive,
                log,
            ],
            show_progress="hidden",
            queue=False,
            trigger_mode="always_last",
        )
        run_button.click(
            run_handler,
            inputs=[uploaded, language, device, max_pages],
            outputs=[status, fields, text, result_json, gallery, archive, log],
            show_progress="minimal",
            concurrency_limit=1,
            trigger_mode="once",
            scroll_to_output=False,
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("OCR_MODEL_HOST", "127.0.0.1"),
        help="listen host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OCR_MODEL_PORT", "7860")),
    )
    args = parser.parse_args(argv)
    settings = RuntimeSettings.load()
    settings.output_root.mkdir(parents=True, exist_ok=True)
    blocked = [
        str(path)
        for path in (
            settings.home / "data" / "raw" / "private",
            settings.home / "private_outputs",
        )
        if path.exists()
    ]
    build_app(settings).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=args.host in {"127.0.0.1", "localhost"},
        allowed_paths=[str(settings.output_root)],
        blocked_paths=blocked,
        show_error=True,
        footer_links=["gradio", "settings"],
        mcp_server=False,
        max_file_size="100mb",
        css=APP_CSS,
    )
    return 0
