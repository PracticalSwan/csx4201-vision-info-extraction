# OCR Model — OpenAI Build Week portable release

Portable, local-only OCR and document information extraction for Windows and
the Docker-backed macOS route. The archive includes the exact final LayoutXLM
and PaddleOCR weights, one-step CLI and GUI launchers, diagnostics, a synthetic
demo sample, manifests, setup instructions, the MIT license for original
code/documentation, and the solo-maintainer contribution policy. It does not
use an OpenAI API key.

This release repairs the local GUI with:

- an image preview and first-page PDF preview;
- exactly one compact extraction progress indicator;
- independently scrollable OCR-text and run-log panes;
- stale-result clearing when a new document is uploaded; and
- ANSI-free run logs.

Verified before publication:

- clean 1,153,305,967-byte archive with one `OCR_Model/` root and no raw or
  private data;
- archive SHA-256
  `e8fc8229235c42436a487d687f75888c5f7713a69cd94c95459b55fb0d046dc6`;
- source commit
  `20739bfb3d6ff4b3b03f973aa8040ca152353a64`;
- ZIP CRC, single-root, required-file, and privacy audits;
- portable doctor with OCR, LayoutXLM, and K-Means import probes;
- full CPU extraction of the bundled sample with the expected five fields;
- native Windows GPU doctor, CLI extraction, and packaged-GUI extraction;
- rebuilt Docker Linux/AMD64 CPU doctor and full extraction with exact field
  values/methods, OCR text, entity/relation triplets, and rotation parity;
- host test suite: 243 passed and 2 environment-dependent skips; and
- final LayoutXLM checkpoint SHA-256
  `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`.

Physical Apple hardware was not available. macOS support is the Docker Desktop
`linux/amd64` CPU path and may be slow on Apple Silicon.
