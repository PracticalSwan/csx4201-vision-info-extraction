"""End-to-end local document information-extraction pipeline."""

from .document_pipeline import DocumentPipeline, DocumentPipelineError

__all__ = ["DocumentPipeline", "DocumentPipelineError"]
