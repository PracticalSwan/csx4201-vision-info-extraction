# OpenAI Build Week submission guide

## Confirmed project choices

- Entrant: solo
- Track: Work & Productivity
- Country of residence: Thailand
- Repository: private
- OpenAI API key: not used
- Judge access changes: approved by the owner

## Registration and submission

The authenticated Devpost connector has already created and populated the live
project:

- Project ID: `1350784`
- Project: [OCR Model: Local Document Intelligence](https://devpost.com/software/ocr-model-local-document-intelligence)
- Saved content: title, tagline, refreshed repaired-GUI write-up, technologies,
  private GitHub repository/Release links, and a refreshed privacy-safe GUI
  thumbnail
- Hackathon state: registered, but not yet submitted to OpenAI Build Week

The live Devpost editor was rechecked in Chrome on 2026-07-20. The public
project page is populated, but the Build Week entry still reports **Draft,
1/5 steps done**. Its Additional info page has no saved submitter type,
country, category, repository URL, testing instructions, or `/feedback`
Session ID. Thailand is now owner-confirmed and will be written with the
remaining values by the Devpost connector. A published Devpost project page is
not evidence of a submitted hackathon entry.

The remaining flow is:

1. Record and publish a YouTube video no longer than three minutes. Public is
   recommended; the organizer's July 18 update also permits Unlisted. It needs
   audible narration explaining the product and showing how Codex and GPT-5.6
   are used. Follow `VIDEO_SCRIPT.md`.
2. In the Codex task used for the demo, run `/feedback` and copy the returned
   Session ID.
3. Send the public YouTube URL and `/feedback` Session ID to the
   Codex task.
4. Recheck every item in `REQUIREMENTS_CHECKLIST.md`. The connector will use
   **Individual**, **Thailand**, **Work & Productivity**, and the existing private
   repository URL, then perform the final submission.

The published deadline is **July 21, 2026 at 5:00 PM Pacific Time**, which is
**July 22, 2026 at 7:00 AM in Bangkok**. Submit earlier so repository access and
the video can be checked while there is still time.

Official references:

- <https://openai.devpost.com/rules>
- <https://openai.devpost.com/details/faqs>
- <https://openai.devpost.com/updates/45371-tuesday-last-minute-tips>
- <https://openai.devpost.com/>

## Existing-project eligibility

The trained OCR/layout model existed before Build Week. The submission must
therefore distinguish the pre-existing model from the meaningful Build Week
extension made after July 13:

- relocatable Windows and Docker/macOS runtime;
- local one-command CLI and GUI;
- included, hash-manifested model artifacts;
- a local consent-gated MCP server;
- the `$review-ocr-document` Codex skill;
- GPT-5.6 review that produces bounded suggestions without an API key.

Do not present the original model training as Build Week work. Use
`BUILD_WEEK_CHANGELOG.md` to explain the extension and show Git history.

## Actions that remain intentionally manual

- Record narration and publish the final YouTube video.
- Run the final real GPT-5.6 skill demo and `/feedback`.
- Review the already confirmed Thailand value.
- Review the final values before the Devpost connector submits them.

The connector can complete the final Devpost submission, but it cannot invent
the `/feedback` Session ID or public video URL.

The owner approved publishing the weights-included bundle as a private GitHub
Release for invited judges. It is published at
<https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week>;
its build, upload, digest, and judge-access verification are tracked in
`GITHUB_RELEASE_HANDOFF.md`.
