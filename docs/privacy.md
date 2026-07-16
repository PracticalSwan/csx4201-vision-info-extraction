# Privacy and Publication Rules

`data/raw/private/gmail/` contains real personal financial and legal material.
It is local private-test data, not training data or publication material.

## Prohibited

- committing or uploading private source files, rendered pages, OCR caches,
  text, filenames, paths, identifiers, previews, or per-document predictions;
- using Gmail to fit models, routing thresholds, rules, checkpoint selection,
  hyperparameters, or examples;
- sending private files to an external service;
- moving a private output into a public report root.

## Allowed local operation

Private inference may run locally with `--private-output`. Detailed results
must remain below the ignored D: private root. A committed report may contain
aggregate counts and timings only and must explicitly declare that it contains
no filenames, OCR text, images, or per-document predictions.

The general extraction CLI resolves every input against the four configured
Gmail roots before opening it. A matching input is rejected unless
`--private-output` is present; that mode then requires the destination to stay
below the ignored D: private root. The caller cannot opt out of either guard.

## Before staging or pushing

1. Verify repository visibility and remote ownership.
2. Run both data and information-extraction verifiers.
3. Inspect `git status --ignored` and every staged path.
4. Inspect the staged diff for private filename/text fragments, secrets,
   `.env` content, credentials, source paths, and generated previews.
5. Reject unexpectedly large files and model/checkpoint/cache artifacts.
6. Confirm the training and checkpoint reports still show Gmail fit rows 0.

Ignore rules are a safeguard, not authorization to publish. A successful scan
means no known match was found in the checked surface; it does not prove that
an image/PDF is safe without human review.
