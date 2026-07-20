# Information-Extraction Task Tracker

## Preserved baseline

- [x] Freeze rotation artifacts, reports, and historical metrics.
- [x] Re-run raw-data, artifact-reload, privacy, and 20-check rotation verification.
- [x] Add a failure-isolated, display-only K-Means wrapper.
- [x] Keep the unreliable exact-angle experiment disabled for inference.

## OCR and storage

- [x] Create D:-backed OCR and layout environments with a 15 GiB reserve gate.
- [x] Isolate Paddle GPU from CUDA PyTorch on Windows.
- [x] Download and hash the exact detector, general, and Thai models.
- [x] Implement general/Thai/auto routing and OCR-evidence orientation scoring.
- [x] Add provenance-aware public OCR caching and private-cache exclusion.
- [x] Verify general, Thai, rotated, image, PDF, and multipage paths.

## Data and model

- [x] Define the universal JSON and annotation schemas.
- [x] Normalize SROIE, FUNSD, FATURA, and supported CORU annotations.
- [x] Build public/private-safe information-extraction and model manifests.
- [x] Implement continuous rotation plus polygon/box transformation.
- [x] Prepare the aligned public final model dataset with Gmail fit rows 0.
- [x] Implement the Detectron2-free LayoutXLM text + 2D-layout model.
- [x] Train the four-epoch final public model on CUDA and verify exact reload.
- [x] Calibrate temperatures and abstention thresholds on `dev_calibration` only.
- [x] Implement entity inference, key-value relations, canonical rules, and generic fallback.

## Inference and evaluation

- [x] Implement image, PDF, multipage, rotated, multilingual, and unknown-type CLI inference.
- [x] Validate and atomically write schema-compliant JSON.
- [x] Run rule and OCR baselines.
- [x] Run the required 18-angle layout grid and 72-case end-to-end angle grid.
- [x] Report recognized-text coverage and one-to-one polygon detection precision/recall/F1.
- [x] Execute the fixed 100-page zero-fit-row CORU holdout and report its limitations.
- [x] Run aggregate-only private Gmail operational inference.
- [x] Generate and cryptographically revalidate synthetic image/rotation/Thai/multipage evidence.
- [x] Add unit/regression tests and environment-specific test partitions.
- [x] Run the locked public in-domain test once without post-test tuning.
- [x] Train and package the final working academic pre-model over the eligible public corpus.
- [ ] Run professor-approved labeled Thai and leave-one-dataset-out benchmarks.

## Finalization

- [x] Update README, summary, requirements, design, setup, IE, evaluation, privacy, memory, and lessons.
- [x] Run final combined verification after documentation and cleanup stabilize.
- [x] Complete the independent final review and fix every confirmed finding.
- [x] Inspect staged content, commit the feature branch, merge it, and sync GitHub `main`.

The only unchecked research item requires professor-approved data and protocol;
it is not a hidden implementation failure. Final publication completed through
GitHub PR #1 at merge commit `b38ebc2fc3de8975c03ef9ea5fe66334f40bd137`.

## Portable release and OpenAI Build Week

- [x] Add one-command full-document extraction through CLI and local Gradio GUI.
- [x] Add a Windows setup/launch path and a Docker-based macOS launch path.
- [x] Package the exact OCR and LayoutXLM weights into a privacy-audited
  `D:\OCR_Model.zip` release with a SHA-256 sidecar.
- [x] Add no-API-key Codex integration through a consent-gated local MCP server
  and repository skill.
- [x] Prepare Work & Productivity submission copy, requirements evidence,
  judge-access guidance, a three-minute video script, and privacy-safe
  screenshots.
- [x] Verify native Windows GPU extraction, Docker Linux/AMD64 CPU extraction,
  and exact output parity on the safe validation document.
- [x] Replace cross-version K-Means pickle loading with hash-bound numeric
  inference parameters and verify zero cluster-label differences over all
  7,520 public feature rows.
- [x] Complete one independent review, fix all three confirmed findings, and
  complete the same reviewer's single follow-up with zero remaining findings.
- [x] Repair the GUI image/PDF preview, duplicate progress surfaces,
  scrollable OCR/log panes, stale-input state, and terminal log rendering;
  verify the final behavior with real browser screenshots.
- [x] Confirm the entrant country as Thailand.
- [x] Publish and remotely verify the owner-approved weights-included public
  GitHub Release at `v1.0.0-build-week`.
- [x] Publish the narrated 2:54 public demo:
  <https://youtu.be/8BV8LnbK1GI>.
- [x] Run the final demonstrated Codex skill flow and save `/feedback`
  Session ID `019f7669-11fd-7923-ad68-ea1a09bd7d74`.
- [x] Invite the Devpost judge account with read-only repository access.
- [x] Complete GitHub passkey confirmation and invite
  `build-week-event@openai.com` with read-only access.
- [x] Populate and refresh Devpost project `1350784` with repaired-GUI
  submission copy, technologies, public repository/Release links, and a
  privacy-safe thumbnail through the connector.
- [x] Submit Devpost entry `1102544` after recording the public video,
  providing the Session ID, and receiving the owner's explicit rules/terms
  agreement. The live manager and connector both report `Submitted`.
