# Privacy Redaction Notes

This file records the privacy rules applied during dataset organization and audit.
It is safe to commit (it contains no private filenames or content).

## 1. Gmail files are private test data

All files under `data/raw/private/gmail/` are real personal documents used only for
private evaluation. They are never treated as public training data.

## 2. Gmail files must not be committed to Git

`.gitignore` excludes:
- `data/raw/`
- `data/raw/private/`
- `data/**/gmail/`
- `data/metadata/private_file_inventory.csv` and any `private_*.csv` / `private_*.json`

The real Gmail inventory is written to `private_file_inventory.csv` (gitignored).
The commitable `file_inventory.csv` anonymizes Gmail filenames.

## 3. Sensitive content that may be present in Gmail files

- full names
- email addresses
- home or billing addresses
- phone numbers
- order IDs, invoice IDs, receipt IDs
- account numbers
- payment details and card fragments
- tax information
- other financial information

## 4. Gmail filenames may themselves be sensitive

Examples of identifying patterns observed (categories only — no real names here):
receipt numbers, invoice numbers, account/fee reference numbers, base64-like tokens,
and localized (Thai) document titles.

Rule: real Gmail filenames are replaced by anonymized identifiers
(`gmail_<file_id>.<ext>`) in every public report. Only the category subfolder
(receipts / invoices / legal_financial_docs / unclassified) is preserved.

## 5. Reporting rules

- Private files must not appear in screenshots without redaction.
- Private document text must not appear in public reports.
- Private documents must not be uploaded to external services.
- Later model evaluation should report only **aggregate** private-test results
  (counts, metrics), never per-document private content.

## 6. Public/private separation

- Public datasets live under `data/raw/public/`.
- Gmail documents live under `data/raw/private/gmail/`.
- `verify_data.py` enforces that no private file appears under a public tree and
  that the Gmail root resolves under a private path.

## 7. Classification of Gmail documents

Gmail documents are classified by **filename keyword only** (contents are never
inspected at this stage):

- receipt terms: `receipt`, `rcpt`, `ใบเสร็จ`
- invoice terms: `invoice`, `inv-`, `e-invoice`, `tax_invoice`, `ใบกำกับภาษี`
- legal/financial terms: `agreement`, `terms`, `policy`, `risk`, `disclosure`,
  `regulation`, `fee`, `complaint`, `privacy`, `execution`, `margin`,
  `declaration`, `regulations`, `disclaimer`, `conflicts`

Documents matching no keyword are placed in `unclassified/` with low confidence
and left for manual review. They are never deleted.
