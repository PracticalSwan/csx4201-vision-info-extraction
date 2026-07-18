# OpenAI Build Week submission guide

## Confirmed project choices

- Entrant: solo
- Track: Work & Productivity
- Repository: private
- OpenAI API key: not used
- Judge access changes: approved by the owner

## Registration and submission

1. Sign in at <https://openai.devpost.com/>.
2. Join the hackathon if the page still shows a **Join hackathon** action.
3. Open the submission manager:
   <https://devpost.com/submit-to/30223-openai-build-week/manage/submissions>.
4. Create a project and use the copy in `SUBMISSION_COPY.md`.
5. Select exactly **Work & Productivity**.
6. Add the private GitHub repository and grant repository access to both judge
   email addresses listed in `JUDGE_ACCESS.md`.
7. Upload/link a public YouTube video no longer than three minutes. It needs
   audible narration explaining the product and showing how Codex and GPT-5.6
   are used. Follow `VIDEO_SCRIPT.md`.
8. In the Codex task used for the demo, run `/feedback` and place the returned
   Session ID in the Devpost field.
9. Recheck every item in `REQUIREMENTS_CHECKLIST.md`.
10. Preview the submission, confirm there is no private content, and submit.

The published deadline is **July 21, 2026 at 5:00 PM Pacific Time**, which is
**July 22, 2026 at 7:00 AM in Bangkok**. Submit earlier so repository access and
the video can be checked while there is still time.

Official references:

- <https://openai.devpost.com/rules>
- <https://openai.devpost.com/details/faqs>
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
- Review the final Devpost preview.
- Click the irreversible final **Submit** button.

Those actions require the owner's voice/account confirmation and should not be
automated silently.
