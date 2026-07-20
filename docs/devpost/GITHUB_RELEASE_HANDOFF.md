# Public GitHub Release record

Release:

<https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week>

## Published payload

- Repository: `PracticalSwan/csx4201-vision-info-extraction` (public)
- Tag: `v1.0.0-build-week`
- Title: `OCR Model - OpenAI Build Week portable release`
- Asset: `OCR_Model.zip`
- Sidecar: `OCR_Model.zip.sha256`
- Exact size: 1,153,305,967 bytes
- Archive SHA-256:
  `e8fc8229235c42436a487d687f75888c5f7713a69cd94c95459b55fb0d046dc6`
- Source commit recorded inside the archive:
  `20739bfb3d6ff4b3b03f973aa8040ca152353a64`

The archive contains one `OCR_Model/` root, the trained LayoutXLM and
PaddleOCR weights, one-step CLI and GUI launchers, a synthetic sample,
manifests, setup guides, the MIT license for original code/documentation, and
the solo-maintainer contribution policy.

## Verification

- source tree clean at build;
- ZIP SHA-256 recomputed and sidecar matched;
- all 180 ZIP entries passed CRC validation;
- no duplicate entries and no path outside the single package root;
- privacy audit passed with no raw data, private outputs, credentials,
  `.runtime`, `runtime.local.json`, or user output;
- final LayoutXLM checkpoint SHA-256 matched
  `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`;
- doctor checks and OCR/LayoutXLM/K-Means import probes passed;
- a full CPU extraction of the bundled sample completed with the expected
  currency, customer, organization, reference, and total fields;
- the pre-existing Windows GPU and Docker Linux/AMD64 CPU parity evidence
  remains valid because the rebuilt source change only adds publication files
  to the package.

Physical Apple hardware was not available. The documented macOS route is
Docker Desktop with a CPU-only `linux/amd64` image and may be slow on Apple
Silicon.

## Replacement guard

Build replacements in an isolated directory named `OCR_Model`; do not rebuild
over the owner's installed `D:\OCR_Model`. Keep the current Release asset until
the replacement passes all checks. Never attach `.runtime`,
`runtime.local.json`, outputs, raw/private data, Gmail material, or private
inventories.
