# Private GitHub Release handoff

The owner explicitly approved the private weights-included Release on
2026-07-20. Publication remains pending only while the repaired GUI package is
rebuilt and reverified.

## Approved release

- Repository: `PracticalSwan/csx4201-vision-info-extraction` (private)
- Tag: `v1.0.0-build-week`
- Title: `OCR Model - OpenAI Build Week portable release`
- Asset: `D:\OCR_Model.zip`
- Sidecar: `D:\OCR_Model.zip.sha256`
- Exact size: 1,153,107,272 bytes
- Archive SHA-256:
  `b8b73db81a9d7751b21fc744c3245dbd95fd8b27deca6cbc1f5383c0bcd8ce83`
- Source commit recorded inside the archive:
  `38b85755e3641d452e5fb9d8e8363815ac581696`

## Proposed release notes

Portable, local-only OCR and document information extraction for Windows and
the Docker-backed macOS route. The archive includes the exact final LayoutXLM
and PaddleOCR weights, one-step CLI, local GUI, diagnostics, a synthetic demo
sample, manifests, and setup instructions. It does not use an OpenAI API key.

Verified before publication:

- clean archive with one `OCR_Model/` root and no raw/private data;
- native Windows GPU doctor and full sample extraction;
- Docker Linux/AMD64 CPU build and full sample extraction;
- identical field values, OCR text, entity triplets, relation triplets, and
  rotation cluster/zone across those two runs;
- final LayoutXLM checkpoint SHA-256
  `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`.

Physical Apple hardware was not available. macOS support is the verified
Docker Desktop `linux/amd64` CPU path and may be slow on Apple Silicon.

## Publication guard

Before upload, reconfirm that the repository is still private and both judge
access paths remain read-only. Do not attach owner outputs, `.runtime`,
`runtime.local.json`, raw data, Gmail material, or private inventories.
