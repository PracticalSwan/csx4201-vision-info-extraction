# Local private-document testing

Private Gmail documents are an operational test set only. This command runs
the fixed final checkpoint locally; it cannot train, calibrate, select a
checkpoint, or change thresholds.

Run it from the OCR environment after final training and calibration:

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
$checkpoint = 'D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final'
& $ocr scripts/run_private_test.py `
  --input-root 'data\raw\private\gmail' `
  --output-root 'D:\CSX4201\vision-info-extraction-assets\private-evaluation\final-model' `
  --language auto --device gpu:0 --private-output `
  --checkpoint $checkpoint --recursive --continue-on-error `
  --no-private-visualizations --aggregate-report `
  --manual-review-csv --force
```

Use `--limit N` for a bounded run, or repeat `--file <relative-path>` to select
specific files under `--input-root`. A selected file cannot escape that root.
`--max-pages N` bounds multipage documents. The command requires the output
root to remain below the configured ignored private root on D:.
The general `extract_document.py` command enforces the same input boundary: a
path under any configured Gmail root cannot run without `--private-output`.

Detailed local output uses anonymous IDs:

```text
private-evaluation/final-model/
  aggregate_report.json
  aggregate_report.md
  manual_review.csv
  documents/
    private_000001/
      document_result.json
      pages/
```

`manual_review.csv` is private because predicted values and evidence page
numbers are included for correction. Keep it on D: and never copy it into a
tracked report directory. The optional public aggregate report contains only
counts, timings, route/document-type totals, the checkpoint hash, and explicit
no-content declarations. It contains no source filename, OCR text, image, or
per-document prediction.

A nonzero exit means at least one selected document failed or no document
completed. With `--continue-on-error`, anonymous error records remain local so
the other documents can finish. Never use the aggregate result to tune the
model; there is no private ground truth, so it is not an accuracy estimate.

The final bounded operational check on 2026-07-17 used `--limit 2` after all
model and calibration choices were fixed. It completed 2/2 documents and two
pages with zero failures. The public aggregate reports 47.5 OCR words, 6.0
entities, 1.0 relation, and 4.0 non-null fields on average, plus the exact
checkpoint hash. It explicitly declares that it contains no filenames, OCR
text, images, or per-document predictions. These counts are operational only
and were not used to change the model.
