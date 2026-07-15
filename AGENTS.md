# AGENTS.md — Project source of truth (cross-host)

> **How this file fits together with the others**
> - This is the **shared, cross-host** source of truth for the `CSX4201/Project` workspace. It is read by Codex, Claude Code, and Gemini CLI alike.
> - `CLAUDE.md` (sibling file) layers Claude Code-specific behavior on top of this one. The two **complement** each other: shared rules live here, Claude-only rules live there.
> - User-level global rules still apply (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`).
> - **Priority when rules conflict:** project files > global files > host defaults. For Claude Code specifically, `CLAUDE.md` > `AGENTS.md` at equal scope.

## Priority system
**MUST** > **SHOULD** > **OPTIONAL**. At equal priority, the narrower scope wins.

---

## Project identity
- **Course:** CSX4201 — Artificial Intelligence Concepts (Assumption University, Bangkok). Spring 2026.
- **Workspace:** `c:\Assumption University\CSX4201\Project`
- **Data root:** `data/raw/` (organized 2026-07-13). The original `vision_info_extraction_data/` is now an empty husk (0 files).
- **Domain (confirmed by professor, 2026-07-13):** vision information extraction — build a model that extracts information correctly and accurately from images, documents, and any file that contains information. The datasets on disk are scanned forms, receipts, and invoices (OCR + IE + possibly DocVQA).
- **Status:** Dataset organization and the bounded rotation baseline remain verified. On 2026-07-15 the repository added a complete information-extraction inference and smoke-training lifecycle: D:-backed PaddleOCR/LayoutXLM environments, exact general/Thai OCR models, public annotation normalization, dynamic geometry transforms, public-only layout smoke training, image/PDF/multipage inference, schema validation, bounded evaluation, and aggregate-only private testing. Final-review regressions added real rotated phrase recovery, automatic Thai multipage routing, one-to-one polygon detection metrics, and hash-bound integration evidence. Treat the smoke checkpoint as lifecycle proof, not a final-quality model: public text-detection F1 is 0.4146, entity F1 is about 0.01, and relation F1 is 0. K-Means remains display-only, and the failed exact-angle estimator is disabled for inference.

## Project goal and model requirements (confirmed by professor, 2026-07-13)

The professor has now stated the project goal. Treat it as authoritative for downstream work; do not re-litigate it without the user.

- **Goal:** Build a **vision pre-model** that extracts information correctly and accurately from pictures, files, and anything that contains information.
- **Rotation-zone clustering (required auxiliary feature):** group each document by its rotation angle into four zones for display/diagnostics. It is included alongside the main pre-model but must not control OCR or extraction:

| Zone (cluster) | Rotation angle range | Example |
|----------------|----------------------|---------|
| 1              | 0 to 90 degrees      | 45 degrees -> Zone 1 |
| 2              | 90 to 180 degrees    |         |
| 3              | 180 to 270 degrees   |         |
| 4              | 270 to 360 degrees   |         |

- The rotation requirement has an executed baseline. It uses deterministic balanced augmentation, K-Means with four clusters, a training-only Hungarian cluster-to-zone mapping, and an experimental zone-guided exact-angle search. Its held-out quality is weak, so inference exposes only the K-Means display value and disables exact-angle correction by default. This implementation does not resolve the professor's open method questions.
- **Open sub-questions to confirm with the professor (do NOT assume answers):**
  - Angle-range boundaries: is exactly 90 / 180 / 270 degrees in the lower or upper zone? Default convention if unspecified: half-open intervals [0,90), [90,180), [180,270), [270,360).
  - Method: the professor said "clustering," but the four zones are predefined angular quadrants. Confirm whether this means K-Means (k=4) on the estimated rotation angle, or a deterministic 4-way classifier / regression head. These are different implementations.
  - Angle source: zoning requires a per-document rotation-angle estimate first (orientation estimation). Confirm the angle-estimation approach before zoning.
  - What "pre-model" denotes: interpreted as a preprocessing/precursor model (orientation handling that runs before extraction). Confirm with the professor.

## Verified workspace structure (2026-07-13)
```text
data/
  raw/                     # original public/private inputs; read-only and ignored
  metadata/                # organization plus page/split/rotation/feature manifests
  processed/
    private/page_images/   # anonymized private PDF renders; ignored
    rotated_images/full/   # 8,332 bounded full-profile rotations; ignored
    features/full/         # raw/transformed feature caches; ignored
  splits/                  # train, validation, test, private_test page manifests
models/kmeans_rotation/    # scaler, PCA, K-Means, mapping, provenance
reports/                   # preparation, features, K-Means, angles, verification
schemas/                   # versioned inference-output JSON Schema
scripts/                   # organization plus rotation-stage CLI entry points
src/                       # organization, rotation, OCR, IE, inference, evaluation
tests/                     # synthetic tests; 158 pass, 1 environment-dependent skip
```
Large OCR/layout assets live below `D:\CSX4201\vision-info-extraction-assets` in separate Python 3.10 environments. Raw totals remain 128,793 files and 35,459,126,772 bytes. The bounded rotation run generated 8,332 rotations with 0 failures and 2,083 rows per zone. Rotation verification passes 20/20 checks; the expanded development suite passes 158 tests with one skip. OCR-runtime and CUDA-layout partitions pass 53 and 3 tests respectively.

---

## Session-start protocol (MUST, every session)
1. **Read `AGENT_MEMORY.md`.** Treat its contents as hints written at a point in time. **Verify any fact against the live source before relying on it** (paths may have moved, files may have been added/removed). Reading it is orientation; it does **not** count as completing a task.
2. **Read `LESSONS.md`.** Do not repeat a mistake that has already been recorded there.
3. If anything in either file contradicts the current filesystem or the user's current request, trust the live evidence and the user — then update the file.

## Core workflow (MUST)
- **Classify the task first:** advisory/review, read-only investigation, local mutation, cross-workspace mutation, or document/writing. Behavior differs by type.
- **Read before edit.** Never edit a file you have not seen.
- **Loop:** Read -> Plan minimal change -> Implement -> Test/verify -> Refine only if needed.
- **Minimal diffs.** Change only what the task requires. No reformatting or "clean up" outside scope.
- **Evidence over assumption.** Prefer tests, execution, and logs over inference. State which verification level was performed (static review / local execution / test execution / live external verification / inference).
- **Surface uncertainty immediately.** Distinguish verified fact from inference in every report.
- **No implied work.** Do not claim an edit, validation, sync, or doc update happened unless it actually did.

## Data and git hygiene
- The repo is initialized on `main` with an existing private GitHub remote. Before every commit or push:
  - `.gitignore` already excludes OS junk (`__MACOSX/`, `.DS_Store`), Python build/venv artifacts, and IDE folders — keep it that way.
  - Large binaries are present (`.zip`, `.pdf`, image archives). Before committing, decide policy: **Git LFS**, gitignore + documented download steps, or commit directly. Do not blindly push hundreds of MB.
  - Derived rotations, features, private page renders, private operational manifests, and large operational manifests are local ignored outputs. Classical artifacts under `models/kmeans_rotation/` are legitimate for this stage, but verify their provenance and contents before staging.
  - Public reports contain public row-level predictions and private aggregate-only metrics. Even sanitized outputs require a final privacy review before publication.

## Privacy (CRITICAL — read before any commit or external upload)
- `data/raw/private/gmail/` contains **real personal financial and legal documents** (invoices, agreements, risk disclosures, terms, fee details). (Moved here from the original `vision_info_extraction_data/gmail_private_test/`.)
- **Default posture:** these MUST NOT be uploaded to a public GitHub repo. `.gitignore` excludes `data/raw/`, `data/**/gmail/`, and `private_file_inventory.csv` by default.
- The commitable `file_inventory.csv` anonymizes Gmail filenames (`gmail_<id>.<ext>`); real filenames live only in the gitignored `private_file_inventory.csv`.
- Before any commit/push, confirm the repo visibility. If the user explicitly needs Gmail data in the repo, require a **private** repo and explicit opt-in. Surface this risk every time a commit/share action is pending.
- Beyond this folder, watch for personal identities, names, emails, account numbers, and signatures inside any image/PDF before generating derived artifacts (annotations, cropped images, logs) that might be shared.

## Pending from user (blockers for fuller documentation)
- [x] High-level project goal — CONFIRMED 2026-07-13 by professor (vision info-extraction pre-model + rotation-zone clustering). See *Project goal and model requirements*.
- [ ] Confirm whether the implemented provisional canonical fields and document types match the professor's final required scope.
- [ ] Rotation-zone open sub-questions (boundary inclusivity; clustering vs classification; angle-estimation source; meaning of "pre-model"). The current baseline uses half-open zones, K-Means, and a zone-guided handcrafted estimator only as provisional implementation choices.
- [ ] Confirm official quality thresholds and test protocol. Current smoke and natural CORU-holdout results are not final benchmarks.
- [ ] Is `gmail_private_test` the private leaderboard set? Should derived outputs be derived from it at all?
- [ ] Target deliverable: final trained model, notebook, report, or competition submission?
- [x] Repo visibility for GitHub — confirmed private before the 2026-07-15 publication pass; recheck before every future upload.
- [x] README.md — updated for the implemented information-extraction smoke lifecycle (2026-07-15), while preserving historical rotation metrics and open research decisions.

> When the user provides the above, update this section, `AGENT_MEMORY.md`, and then `README.md`.
