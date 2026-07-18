# Devpost submission copy

## Project name

OCR Model: Local Document Intelligence

## Tagline

Turn scans and PDFs into structured, reviewable data on your own computer.

## Track

Work & Productivity

## Short description

OCR Model extracts text, document fields, entities, relations, tables, and
display-only rotation zones from images and PDFs. It runs locally through a
one-command CLI or a browser GUI, and ships with its trained weights for
Windows and Docker-backed macOS use. A Build Week Codex skill and local MCP
server let GPT-5.6 review only the fields a user explicitly approves, with no
OpenAI API key and no raw-document upload.

## Inspiration

Document extraction projects often stop at a notebook or a model checkpoint.
That makes them hard to test, hard to share, and risky for documents that may
contain personal information. I wanted the finished course model to behave like
a usable tool while keeping local extraction authoritative and making any
cloud-assisted review deliberate.

## What it does

The user selects an image or PDF. The local pipeline evaluates cardinal
orientation candidates, routes text through general or Thai PaddleOCR models,
runs the fine-tuned LayoutXLM multitask checkpoint, adds evidence-based field
rules and generic key/value relations, validates the result schema, and writes
JSON plus page overlays.

The GUI presents extracted fields, OCR text, full JSON, visualizations, and a
downloadable result archive. The CLI performs the same full extraction in one
command.

The optional `$review-ocr-document` skill lists completed results through
opaque IDs. It asks which fields the user wants reviewed and requests consent
before the selected values enter the signed-in GPT-5.6 Codex session. OCR text
requires separate consent and is bounded. GPT-5.6 flags uncertainty and
inconsistencies as suggestions; it never overwrites the local result.

## How I built it

- PaddleOCR 3.7 with pinned detector/general/Thai recognizer artifacts
- a four-epoch fine-tuned LayoutXLM multitask checkpoint
- Python, PyMuPDF, Pillow, OpenCV, scikit-learn, and JSON Schema
- Gradio 6 for the local GUI
- separate OCR and layout Python environments to avoid framework conflicts
- Docker Compose with a CPU-only `linux/amd64` runtime for Windows/macOS
- a repo Codex skill and local STDIO MCP server for GPT-5.6 review

The Build Week work wraps the existing academic model in a relocatable product
surface and adds the Codex/GPT-5.6 review loop. It does not use an OpenAI API
key.

## Challenges

Paddle and PyTorch can load incompatible GPU libraries in one environment, so
the application launches the verified OCR worker and LayoutXLM worker in
separate Python processes. Shipping the model also required replacing
machine-specific paths with runtime overrides, preserving exact model hashes,
and ensuring private training/evaluation material never enters the bundle.

The cloud-review boundary was another design challenge. The MCP server was
intentionally limited: it has no extraction-by-path tool, returns no paths or
filenames, excludes private outputs, requires explicit field selection, and
keeps OCR text off by default.

## Accomplishments

- One-command full extraction instead of a multi-step research workflow
- Local GUI with field, text, JSON, visualization, and archive views
- Included model weights with SHA-256 manifest
- Windows native setup and Docker-backed macOS path
- No API-key requirement
- Consent-gated GPT-5.6 assistance with unchanged local results
- Privacy checks that exclude raw/private datasets from the distributable

## What I learned

Packaging is part of model quality. A strong reference-token score does not
remove OCR bottlenecks, environment drift, provenance requirements, or the need
for human review. Separating local extraction from optional reasoning made the
system easier to explain and gave users a clear boundary around what can leave
their computer.

## What's next

- Test the Docker package on physical Intel and Apple Silicon Macs
- Improve end-to-end OCR quality on natural, rotated, and multilingual pages
- Expand relation supervision beyond FUNSD
- Add signed desktop installers after the academic prototype is stable

## Accuracy disclosure

Reference-token entity F1 is 0.9813. Bounded end-to-end entity F1 is
0.1314–0.1830 because OCR remains the main bottleneck. Relation quality is
limited by FUNSD-only supervision. The K-Means rotation zone is display-only.
This is an academic/noncommercial prototype and requires human verification
for consequential decisions.
