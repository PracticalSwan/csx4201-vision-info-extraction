# Private GitHub Release handoff

The owner explicitly approved the private weights-included Release on
2026-07-20. It was published and remotely reverified on the same date:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week>

## Approved release

- Repository: `PracticalSwan/csx4201-vision-info-extraction` (private)
- Tag: `v1.0.0-build-week`
- Title: `OCR Model - OpenAI Build Week portable release`
- Asset: `D:\OCR_Model.zip`
- Sidecar: `D:\OCR_Model.zip.sha256`
- Exact size: 1,153,302,135 bytes
- Archive SHA-256:
  `f6a057e5c37c6036bd1d4ad6c247aa0895e893d87fe17f997fd011e0c5064f9e`
- Source commit recorded inside the archive:
  `5b2c964f0affea209aefc03f6ce03183c7dd88de`

## Published release notes

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

## Remote verification

- Repository visibility: private
- Release state: published, not draft, not prerelease
- Tag target:
  `5b2c964f0affea209aefc03f6ce03183c7dd88de`
- Remote ZIP state: uploaded
- Remote ZIP size: 1,153,302,135 bytes
- Remote digest:
  `sha256:f6a057e5c37c6036bd1d4ad6c247aa0895e893d87fe17f997fd011e0c5064f9e`
- Devpost judge account: active pull-only access
- OpenAI judge invitation: pending acceptance, read-only, not expired

## Publication guard

Before any replacement upload, reconfirm that the repository is still private
and both judge access paths remain read-only. Do not attach owner outputs, `.runtime`,
`runtime.local.json`, raw data, Gmail material, or private inventories.
