# Three-minute demo script

Target length: 2:35–2:50. Record at 1080p with audible narration. Use only the
bundled synthetic sample and the sanitized screenshots in `assets/`.

## Before recording

1. Start the portable GUI with `D:\OCR_Model\launch_windows.bat`.
2. Load only `D:\OCR_Model\samples\unknown_upright.png`.
3. Open the Codex task that will demonstrate `$review-ocr-document`, select
   GPT-5.6, and keep the safe result ready.
4. Close email, messaging, file explorers showing unrelated folders, password
   managers, and any private documents. Turn on Windows **Do not disturb**.
5. Set browser and app zoom so the GUI fields and consent prompt are readable
   at 1080p.

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

## Record and upload on Windows

The recommended built-in workflow is Microsoft Clipchamp:

1. Open Clipchamp and create a new video.
2. Choose **Record & create** > **Screen**.
3. Allow microphone access, select the microphone, and choose the entire
   screen so the recording can move between the GUI and Codex.
4. Record the timeline above in one take or in short sections. Keep the final
   edit between 2:35 and 2:50 so there is margin below the 3:00 limit.
5. Trim pauses and setup time. Do not cut away the consent prompt or the
   evidence that the local JSON stays unchanged.
6. Export a 1080p MP4 and play it once from beginning to end. Confirm that
   narration is audible, all important text is readable, and no notification
   or private information appears.

Microsoft's current screen-recording instructions are at
<https://support.microsoft.com/en-US/Clipchamp/how-to-make-a-screen-recording>.

Upload the finished MP4:

1. Open <https://studio.youtube.com/> and choose **Create** > **Upload
   videos**.
2. Use the title **OCR Model: Local Document Intelligence | OpenAI Build Week
   Demo**.
3. Complete YouTube's audience choice truthfully, then set **Visibility** to
   **Public** and publish. Public is recommended for sharing the project. The
   organizer's July 18 update also permits **Unlisted**; never use Private.
4. Open the resulting URL in a signed-out or InPrivate window and play the
   first few seconds. The link must work without an account.
5. Send that public YouTube URL back to the Codex task. It will be attached to
   Devpost through the Devpost connector.

YouTube's current upload instructions are at
<https://support.google.com/youtube/answer/57407?hl=en>.

## Recording checklist

- Video is Public (recommended) or Unlisted, as permitted by the organizer's
  July 18 update; Private is not sufficient.
- YouTube URL works in a signed-out/private window.
- Duration is no more than 3:00.
- Narration explicitly says **Codex**, **GPT-5.6**, and **no API key**.
- No desktop notification, username, absolute path, email, token, or private
  document is visible.
- `/feedback` is run in the demonstrated Codex task after recording.
