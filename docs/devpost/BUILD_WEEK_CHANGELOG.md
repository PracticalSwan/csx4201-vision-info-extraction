# Build Week extension changelog

The trained information-extraction model and its final evaluation existed
before OpenAI Build Week. The following product and GPT-5.6 integration work is
the submission's post–July 13 extension.

## July 21, 2026

- Published the corrected 2:54 narrated demo and verified that it plays
  without authentication.
- Submitted Devpost entry `1102544` in the Work & Productivity track with
  Thailand, the public repository, testing instructions, and `/feedback`
  Session ID.
- Made the repository public with a reader-first README, MIT license for
  original code/documentation, and a solo-maintainer contribution policy.
- Rebuilt the weights-included Release from clean commit `e47023d`, added the
  license and contribution files, reran the privacy/ZIP/doctor checks, and
  completed a full CPU extraction with the bundled sample.

## July 20, 2026

- Repaired the local GUI so image uploads and first PDF pages render in a
  document-preview panel without changing the extraction input.
- Replaced stacked output overlays plus callback progress with one compact
  Gradio-owned loading indicator.
- Added bounded, independently scrollable OCR-text and run-log panes, removed
  terminal color escapes, and reset stale outputs whenever the input changes.
- Added image/PDF preview, overflow, event-configuration, and log-sanitization
  regressions plus live Chrome screenshots from real GPU extraction.

## July 19, 2026

- Added a relocatable runtime resolver that reuses the verified pipeline
  without rewriting its inference logic.
- Added one-command CLI entry points and a local Gradio GUI.
- Added structured result presentation, visualization browsing, and local
  result archives.
- Added Windows CPU/GPU setup and launch scripts.
- Added a CPU-only Docker Compose path pinned to `linux/amd64` for Windows and
  macOS Docker Desktop.
- Validated that Docker path with full CPU inference and a semantic comparison
  against the native GPU result.
- Added a reproducible `D:\OCR_Model` builder that copies exact model artifacts,
  rewrites only portable path metadata, generates SHA-256 manifests, and
  excludes private/raw data.
- Replaced cross-version loading of the display-only scikit-learn pickles with
  a hash-bound numeric inference artifact; all 7,520 public feature rows kept
  identical cluster labels.
- Added the `$review-ocr-document` Codex skill.
- Added a local read-only STDIO MCP server with opaque IDs, private-result
  exclusion, explicit field selection, confirmation, and separately bounded
  OCR text.
- Added Devpost submission copy, video script, judge-access guide, eligibility
  checklist, and accuracy/privacy disclosures.

## Evidence to show judges

- Git commits after July 13 containing the files above
- GUI and CLI runs against the bundled synthetic sample
- Sanitized screenshots under `docs/devpost/assets/`
- `MODEL_MANIFEST.json` and doctor output
- Docker build/run evidence
- Codex task with GPT-5.6 skill invocation and consent sequence
- `/feedback` Session ID from that demonstrated task

The extension uses the user's signed-in Codex/GPT-5.6 session. It does not add
an OpenAI API key or make OpenAI API calls from the OCR application.
