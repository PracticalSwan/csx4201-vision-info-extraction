# OCR Model — OpenAI Build Week portable release

Portable, local-only OCR and document information extraction for Windows and
the Docker-backed macOS route. The archive includes the exact final LayoutXLM
and PaddleOCR weights, one-step CLI and GUI launchers, diagnostics, a synthetic
demo sample, manifests, and setup instructions. It does not use an OpenAI API
key.

This release repairs the local GUI with:

- an image preview and first-page PDF preview;
- exactly one compact extraction progress indicator;
- independently scrollable OCR-text and run-log panes;
- stale-result clearing when a new document is uploaded; and
- ANSI-free run logs.

Verified before publication:

- clean 1,153,302,135-byte archive with one `OCR_Model/` root and no raw or
  private data;
- archive SHA-256
  `f6a057e5c37c6036bd1d4ad6c247aa0895e893d87fe17f997fd011e0c5064f9e`;
- source commit
  `5b2c964f0affea209aefc03f6ce03183c7dd88de`;
- native Windows GPU doctor, CLI extraction, and packaged-GUI extraction;
- rebuilt Docker Linux/AMD64 CPU doctor and full extraction with exact field
  values/methods, OCR text, entity/relation triplets, and rotation parity;
- host test suite: 243 passed and 2 environment-dependent skips; and
- final LayoutXLM checkpoint SHA-256
  `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`.

Physical Apple hardware was not available. macOS support is the Docker Desktop
`linux/amd64` CPU path and may be slow on Apple Silicon.
