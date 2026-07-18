# Three-minute demo script

Target length: 2:35–2:50. Record at 1080p with audible narration. Use only the
bundled synthetic sample and the sanitized screenshots in `assets/`.

## 0:00–0:20 — Problem and promise

Show the title slide, then the local GUI.

Narration:

> OCR Model turns scans and PDFs into structured, reviewable data on your own
> computer. I converted my finished course model into a one-command tool for
> Windows and Docker-backed macOS, with no OpenAI API key.

## 0:20–1:10 — Local extraction

Select `samples/unknown_upright.png` and click **Extract document**. If the live
run is too slow, start with a completed result, then show a short sped-up clip.
Open the fields, OCR text, JSON, and visual confirmation tabs.

Narration points:

- PaddleOCR handles general/Thai text and orientation candidates.
- The fine-tuned LayoutXLM checkpoint predicts entities, document type,
  canonical fields, and relations.
- The local pipeline validates JSON and creates page overlays.
- The shown rotation zone is diagnostic only.
- All extraction stays local.

## 1:10–2:15 — Codex and GPT-5.6

Show Codex with GPT-5.6 selected. Invoke `$review-ocr-document`.

1. Show `list_reviewable_results` returning an opaque ID and field names.
2. Pick two fields.
3. Show the skill asking consent.
4. Approve only those fields; decline OCR text for the shortest demo.
5. Show GPT-5.6 flagging uncertainty or suggesting normalization.
6. Return to `document_result.json` and show that it did not change.

Narration:

> For Build Week I added a Codex skill and a local STDIO MCP server. The server
> exposes no source-file tool, paths, filenames, or private results. GPT-5.6
> sees only fields I select after confirmation and returns suggestions without
> changing the local model output. This uses my signed-in Codex session, not an
> API key.

## 2:15–2:45 — Portability, evidence, and limits

Show the portable folder, `MODEL_MANIFEST.json`, and the README.

Narration:

> The package includes exact OCR and LayoutXLM weights, hash verification,
> Windows launchers, and a CPU Docker path for macOS. I also disclose the main
> limitation: reference-token entity F1 is 0.9813, but bounded end-to-end F1 is
> 0.1314 to 0.1830 because OCR remains the bottleneck. This is an academic,
> noncommercial prototype that requires human review.

End on the GUI/result screenshot with the project name.

## Recording checklist

- Video is publicly visible on YouTube; unlisted/private is not sufficient.
- YouTube URL works in a signed-out/private window.
- Duration is no more than 3:00.
- Narration explicitly says **Codex**, **GPT-5.6**, and **no API key**.
- No desktop notification, username, absolute path, email, token, or private
  document is visible.
- `/feedback` is run in the demonstrated Codex task after recording.
