# Optional Codex/GPT-5.6 review (no API key)

The OCR Model performs extraction locally. Its optional Build Week extension
lets GPT-5.6 review selected output fields through Codex without adding an
OpenAI API call to the application.

## What is installed

- `.agents/skills/review-ocr-document/`: a repo skill that enforces consent,
  minimal disclosure, untrusted-document handling, and suggestions-only output.
- `mcp_server.py`: a local STDIO MCP server with two read-only tools.
  - `list_reviewable_results` returns opaque result IDs and field names only.
  - `prepare_review_payload` returns only explicitly selected field values after
    confirmation. OCR text is separate, opt-in, and capped at 4,000 characters.

The MCP server cannot accept a source path, cannot run OCR, cannot return raw
files, and cannot overwrite extraction results. Results marked private and
results in private output folders are excluded.

## Windows installation

After `setup_windows.bat` succeeds:

```powershell
.\install_codex_integration.ps1
```

Open this folder in Codex, select GPT-5.6, and invoke:

```text
$review-ocr-document
```

Codex will list opaque result IDs, ask which fields to review, request direct
permission before sending selected values to the signed-in GPT-5.6 session, and
ask separately before including any OCR text.

## macOS installation

Build the Docker service once with `bash launch_macos.command`, stop it, then
run:

```bash
bash install_codex_integration_macos.command
```

The registered STDIO command starts the MCP server in the same local Docker
runtime.

## Data boundary

Local only unless explicitly approved:

- source image/PDF bytes
- filenames and absolute paths
- all unselected fields
- OCR text by default
- private Gmail inputs and private outputs

Eligible for the user's signed-in Codex session only after confirmation:

- the chosen opaque result ID
- explicitly selected field names and values
- confidence/method/page metadata for those fields
- an optional, separately approved OCR excerpt of at most 4,000 characters

GPT-5.6 produces observations, inconsistency flags, and normalization
suggestions. The local JSON remains the source of truth and is never changed by
the skill.

## Build Week evidence

For the demo, show:

1. a local extraction in the GUI;
2. the opaque catalog response;
3. the field-selection and consent prompt;
4. GPT-5.6 suggestions;
5. the unchanged local `document_result.json`.

The submitted Build Week entry uses `/feedback` Session ID
`019f7669-11fd-7923-ad68-ea1a09bd7d74`, captured from the demonstrated Codex
task. Do not put an API key, access token, source document, or private OCR text
in a future demo or submission update.
