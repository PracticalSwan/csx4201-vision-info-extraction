"""Local STDIO MCP server exposing only consent-gated OCR result summaries."""

from .review import (
    list_reviewable_results as catalog_results,
    prepare_review_payload as build_review_payload,
)
from .runtime import RuntimeSettings


def create_server():
    from mcp.server.fastmcp import FastMCP

    settings = RuntimeSettings.load()
    mcp = FastMCP(
        "OCR Model Local Review",
        instructions=(
            "Read-only bridge to completed local OCR results. Never extract a "
            "document or return file paths. Call list_reviewable_results first. "
            "Call prepare_review_payload only after the user explicitly confirms "
            "the selected fields and separately confirms any OCR text."
        ),
    )

    @mcp.tool()
    def list_reviewable_results() -> dict:
        """List opaque result IDs and field names without values, text, or paths."""
        return catalog_results(settings.output_root)

    @mcp.tool()
    def prepare_review_payload(
        document_id: str,
        confirmed_cloud_review: bool = False,
        selected_fields: list[str] | None = None,
        include_ocr_text: bool = False,
        max_text_chars: int = 0,
    ) -> dict:
        """Prepare explicitly approved, bounded fields for GPT-5.6 suggestions.

        Set confirmed_cloud_review true only after direct user confirmation in
        the current conversation. Set include_ocr_text true only after separate
        confirmation and use the smallest useful max_text_chars value.
        """
        return build_review_payload(
            settings.output_root,
            document_id,
            confirmed_cloud_review=confirmed_cloud_review,
            selected_fields=selected_fields,
            include_ocr_text=include_ocr_text,
            max_text_chars=max_text_chars,
        )

    return mcp


def main() -> None:
    create_server().run(transport="stdio")
