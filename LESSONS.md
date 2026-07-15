# LESSONS.md — Mistakes and lessons log

> **READ RULE (MUST):** Read this at the start of every session. Do not repeat a mistake that is already recorded here.
> **How to use:** When a mistake happens or a non-obvious lesson is learned, add a row. Be specific about cause and the durable corrective action — not blame.
> Keep entries concise and chronological (newest at the bottom).

## Format
| Date | Context | What went wrong | Root cause | Corrective action (durable) |
|------|---------|-----------------|------------|------------------------------|
| (example) YYYY-MM-DD | task / area | one-line description | why it happened | the rule to follow next time |

## Log
| Date | Context | What went wrong | Root cause | Corrective action (durable) |
|------|---------|-----------------|------------|------------------------------|
| 2026-07-13 | Workflow init | (none yet — file created during initialization) | — | When the user gives project context, record any early mistakes here. |
| 2026-07-13 | Dataset organization | `inspect_data.py` crashed printing Thai Gmail filenames on the Windows cp1252 console. | stdout used the OEM code page; non-ASCII paths could not be encoded. | Reconfigure `sys.stdout/stderr` to UTF-8 (`errors="replace"`) at the top of every CLI script. |
| 2026-07-13 | Dataset discovery | Public datasets not found initially. | `_locate_named_child` and `_score_candidate` tested the folder name for exact membership in the hint tuple (`name in tuple`), so `sroie_receipts` never matched hint `sroie`. | Match by substring (`any(hint in name for hint in hints)`). |
| 2026-07-13 | Audit performance | Inventory over 129k files / 34 GB was far too slow single-threaded and even threaded when PIL `Image.load()` fully decoded large CORU receipt images. | Full pixel decode of ~31 GB of images dominated runtime. | Use header + `Image.verify()` (no full decode) for validation; thread the file work (hashing releases the GIL). Document the validation level. |
| 2026-07-13 | Near-duplicate scan | The scan hung for minutes inside one dimension bucket of ~9k same-size FATURA images (O(n²) pairwise). | Bucketing by (w,h) created a giant bucket. | Cap per-bucket pairwise work (truncate + log) and cap total images hashed; write core metadata before the optional near-dup step so a slow scan can never lose the inventory. |
| 2026-07-13 | FATURA pairing | First audit reported 20,000 "unmatched annotations" — a false alarm. | Stem-exact matching missed COCO/HF annotation files named `<image>_coco_test.json` / `<image>_hugg_test.json`. | For multi-format datasets, match an annotation to an image when the image stem is the annotation stem or a `_`-delimited prefix of it. |
| 2026-07-13 | Rotation materialization | A literal full-corpus, full-angle run would exceed the machine's safe disk budget. | The 22,086 usable public pages imply 416,028 rotations and an empirical estimate of 219.08 GiB new space, versus 16.12 GiB free at the final gate and a 10 GiB reserve. | Run an empirical smoke profile first, enforce a free-space reserve, and label any deterministic corpus cap explicitly as a bounded full-angle run. |
| 2026-07-13 | Unsupervised rotation model | Four K-Means clusters did not recover the four desired quadrants reliably; held-out accuracy stayed near 38% and exact-angle median error was 90 degrees. | Handcrafted orientation features and document symmetry can group axis-equivalent or content-similar pages rather than the predefined angle labels. | Report raw clustering and mapped metrics separately, preserve boundary/confidence evidence, and do not promote the baseline without a method change and professor confirmation. |
| 2026-07-13 | Stage transition | Organization-only dependency comments, ignore rules, and forbidden-output checks became stale when modeling artifacts were introduced. | Lifecycle controls encoded the old stage as permanent policy instead of distinguishing approved rotation outputs from still-forbidden OCR/neural outputs. | When a project enters a new stage, audit requirements, ignore rules, verifiers, README, memory, and tests together before calling the stage complete. |
| 2026-07-13 | Privacy review | Real private filenames entered otherwise synthetic test fixtures, while the verifier scanned only generated artifact roots. | Test data was copied from live private examples and privacy coverage was directory-limited. | Use invented fixture names and scan committable source, tests, docs, config, metadata, reports, and models against the private-name inventory. |
| 2026-07-13 | Derived artifact reuse | Existing page and rotation PNGs were reused after settings changed and then relabeled with new manifest hashes. | Reuse validated image readability but did not bind the file to its source hash and configuration. | Embed source/configuration provenance in materialized PNGs and regenerate atomically whenever provenance does not match. |

## Recurring watch-items (promote here if a lesson repeats)
- When a CLI must print filesystem paths on Windows, force UTF-8 stdout up front.
- When a Python pipeline processes a large image corpus, never use `Image.load()` for validation; use `Image.open().verify()`.
- Per-bucket or per-group duplicate comparisons must be size-capped to stay bounded.
- Treat a four-cluster result as a hypothesis until mapped held-out metrics show that clusters correspond to the required four zones.
