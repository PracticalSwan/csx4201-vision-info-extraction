"""Local-only Gradio interface for OCR and information extraction."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .api import ExtractionError, run_extraction
from .results import (
    create_result_archive,
    field_rows,
    ocr_text,
    visualization_files,
)
from .runtime import RuntimeSettings


FIELD_HEADERS = ["Field", "Value", "Confidence", "Method", "Page", "Validation"]


def _run_gui(
    uploaded: str | None,
    language: str,
    device: str,
    max_pages: float | None,
    *,
    settings: RuntimeSettings,
    progress: Any,
):
    if not uploaded:
        raise ValueError("Choose an image or PDF first.")
    progress(0.02, desc="Checking local runtime")
    log_tail: list[str] = []

    def on_log(line: str) -> None:
        log_tail.append(line)
        del log_tail[:-12]

    progress(0.08, desc="Running OCR and layout extraction locally")
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
    progress(0.94, desc="Preparing local results")
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
    progress(1.0, desc="Complete")
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
    ) as demo:
        gr.Markdown(
            "# OCR Model\n"
            "Run the finished OCR + LayoutXLM information-extraction pipeline on "
            "an image or PDF. Processing and outputs stay on this computer. "
            "**No OpenAI API key is used.**"
        )
        with gr.Row():
            uploaded = gr.File(
                label="Document",
                file_types=[
                    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
                ],
                type="filepath",
            )
            with gr.Column():
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
                text = gr.Textbox(lines=18, interactive=False, buttons=["copy"])
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
                log = gr.Textbox(lines=12, interactive=False)
        archive = gr.File(label="Download complete local result (.zip)", interactive=False)

        def run_handler(
            uploaded_value: str | None,
            language_value: str,
            device_value: str,
            max_pages_value: float | None,
            progress=gr.Progress(),
        ):
            return _run_gui(
                uploaded_value,
                language_value,
                device_value,
                max_pages_value,
                settings=runtime,
                progress=progress,
            )

        run_button.click(
            run_handler,
            inputs=[uploaded, language, device, max_pages],
            outputs=[status, fields, text, result_json, gallery, archive, log],
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
    )
    return 0
