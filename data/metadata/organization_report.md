# Organization Report

Overall verification: **PASS** (20/20 checks passed).
Organization mode used: `move`.

## 1. Original directory structure found
Data originally lived under `vision_info_extraction_data/` with the layout recorded in `AGENT_MEMORY.md`: `public_train/{coru_receipts, docvqa_samples, fatura_invoices, funsd_forms, sroie_receipts}`, `gmail_private_test/`, plus empty `public_test/`, `augmented_rotated/`, and `metadata/` placeholders. Several datasets carried both source archives (`.zip`) and extracted folders; FUNSD also carried macOS `__MACOSX/` junk.

## 2-3. Dataset roots detected and identification confidence
| Dataset | Source | Confidence | Current path |
|---------|--------|------------|--------------|
| sroie | public | high | `C:\Assumption University\CSX4201\Project\data\raw\public\sroie` |
| funsd | public | high | `C:\Assumption University\CSX4201\Project\data\raw\public\funsd` |
| fatura | public | high | `C:\Assumption University\CSX4201\Project\data\raw\public\fatura` |
| coru | public | high | `C:\Assumption University\CSX4201\Project\data\raw\public\coru` |
| gmail | private | high | `C:\Assumption University\CSX4201\Project\data\raw\private\gmail` |

## 4-5. Organization actions
- `moved` **sroie**: `C:\Assumption University\CSX4201\Project\vision_info_extraction_data\public_train\sroie_receipts` -> `C:\Assumption University\CSX4201\Project\data\raw\public\sroie` (files 2926->2926 hashes-ok)
- `moved` **funsd**: `C:\Assumption University\CSX4201\Project\vision_info_extraction_data\public_train\funsd_forms` -> `C:\Assumption University\CSX4201\Project\data\raw\public\funsd` (files 810->810 hashes-ok)
- `moved` **fatura**: `C:\Assumption University\CSX4201\Project\vision_info_extraction_data\public_train\fatura_invoices` -> `C:\Assumption University\CSX4201\Project\data\raw\public\fatura` (files 40005->40005 hashes-ok)
- `moved` **coru**: `C:\Assumption University\CSX4201\Project\vision_info_extraction_data\public_train\coru_receipts` -> `C:\Assumption University\CSX4201\Project\data\raw\public\coru` (files 85026->85026 hashes-ok)
- `moved` **gmail**: `C:\Assumption University\CSX4201\Project\vision_info_extraction_data\gmail_private_test` -> `C:\Assumption University\CSX4201\Project\data\raw\private\gmail` (files 26->26 hashes-ok)

## 6-10. Per-dataset counts, size, images, PDFs, annotations
| Dataset | Files | Size | Images | PDFs | Annotations | Unreadable | Empty |
|---------|-------|------|--------|------|-------------|-----------|-------|
| sroie | 2926 | 1.8 GB | 973 | 0 | 1949 | 0 | 0 |
| funsd | 810 | 42.6 MB | 398 | 0 | 398 | 398 | 0 |
| fatura | 40005 | 798.4 MB | 10000 | 0 | 30003 | 0 | 0 |
| coru | 85026 | 30.4 GB | 41062 | 0 | 13807 | 10 | 6 |
| gmail | 26 | 6.0 MB | 0 | 26 | 0 | 0 | 0 |

## 11-12. Corrupted and empty files
- invalid_json: 203
- corrupted_image: 199
- empty_file: 6

## 13-14. Duplicates
- Exact duplicate groups: 3260
- Near-duplicate groups: 42

## 15. Missing image-annotation pairs
- 5315 unmatched files recorded (see `unmatched_files.csv`).

## 16. Unresolved files / issues
- sroie: 30 exact duplicate groups
- sroie: 32 near-duplicate groups
- sroie: 2 pretrained-model artifact(s) preserved (out of scope)
- funsd: 398 unreadable/invalid files
- funsd: 2 exact duplicate groups
- funsd: 2 near-duplicate groups
- fatura: 8 near-duplicate groups
- coru: 10 unreadable/invalid files
- coru: 6 empty files
- coru: 3228 exact duplicate groups
- coru: 1212 unmatched images, 4103 unmatched annotations

## 17. Gmail private-file count
- Gmail private files: 26 (receipts/invoices/legal_financial_docs/unclassified). Real filenames are kept only in the gitignored `private_file_inventory.csv`.

## 18. Privacy actions applied
- Gmail documents moved under `data/raw/private/gmail/`.
- Public inventory anonymizes Gmail filenames.
- `.gitignore` excludes all raw data and the private inventory.
- Public/private separation verified.

## 19. Verification results
- [PASS] datasets-discovered: 5 datasets (need >=4)
- [PASS] dataset-path:sroie: C:\Assumption University\CSX4201\Project\data\raw\public\sroie exists=True
- [PASS] dataset-path:funsd: C:\Assumption University\CSX4201\Project\data\raw\public\funsd exists=True
- [PASS] dataset-path:fatura: C:\Assumption University\CSX4201\Project\data\raw\public\fatura exists=True
- [PASS] dataset-path:coru: C:\Assumption University\CSX4201\Project\data\raw\public\coru exists=True
- [PASS] dataset-path:gmail: C:\Assumption University\CSX4201\Project\data\raw\private\gmail exists=True
- [PASS] public-private-separation: no private file under public tree
- [PASS] gmail-is-private: gmail root=C:\Assumption University\CSX4201\Project\data\raw\private\gmail
- [PASS] metadata-exists:file_inventory.csv: C:\Assumption University\CSX4201\Project\data\metadata\file_inventory.csv
- [PASS] metadata-exists:private_file_inventory.csv: C:\Assumption University\CSX4201\Project\data\metadata\private_file_inventory.csv
- [PASS] metadata-exists:data_sources.csv: C:\Assumption University\CSX4201\Project\data\metadata\data_sources.csv
- [PASS] metadata-exists:dataset_summary.json: C:\Assumption University\CSX4201\Project\data\metadata\dataset_summary.json
- [PASS] metadata-exists:processing_errors.csv: C:\Assumption University\CSX4201\Project\data\metadata\processing_errors.csv
- [PASS] metadata-exists:duplicate_report.csv: C:\Assumption University\CSX4201\Project\data\metadata\duplicate_report.csv
- [PASS] metadata-exists:unmatched_files.csv: C:\Assumption University\CSX4201\Project\data\metadata\unmatched_files.csv
- [PASS] inventory-paths-resolve: 128767/128767 resolved
- [PASS] no-files-lost: all dataset counts match summary
- [PASS] raw-bytes-unchanged: 100 sample hashes stable across move
- [PASS] duplicate-report-consistent: 3302 duplicate groups well-formed
- [PASS] no-disallowed-neural-or-ocr-outputs: no OCR outputs or neural checkpoints outside data/raw; classical rotation artifacts allowed

## 20. Remaining manual decisions
- Review any low-confidence Gmail classifications in `unclassified/`.
- Decide GitHub repo visibility before any commit (Gmail requires a private repo or must stay uncommitted).
- Decide large-binary policy (Git LFS vs. documented downloads) for the public `.zip`/image archives before the first push.
- Confirm the bundled SROIE pretrained model is intentionally kept (it is preserved and unused at this stage).