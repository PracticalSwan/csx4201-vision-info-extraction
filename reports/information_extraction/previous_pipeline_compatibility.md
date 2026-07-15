# Previous Pipeline Compatibility

Verified 2026-07-15 before and during information-extraction integration.

## Preserved and reused

- stable page/document IDs, public splits, dataset/component names, and
  public/private flags;
- image/PDF rendering conventions, counterclockwise angles, expanded white
  canvases, and page-size metadata;
- configuration loader, atomic JSON/CSV writes, logging, and rotation helpers;
- HOG/Hough/projection feature artifacts, scaler, PCA, K-Means, Hungarian
  mapping, reports, and the experimental exact-angle artifacts.

## Verification evidence

- baseline tests passed before extension; the expanded suite later passed 153
  tests with one environment-dependent skip, then 158 after final-review
  regressions;
- `verify_data.py` preserved 128,793 files, 35,459,126,772 bytes, and sampled
  raw hashes;
- `verify_rotation_data.py --profile full --complete` passed 20/20 checks;
- saved rotation artifacts reloaded; command help and Python compilation
  passed;
- privacy checks found no known private names in committable surfaces.
- the permitted final review revalidated the three correction areas with 13
  focused tests and found no remaining reproducible completion blocker.

## Conflicts found and fixed

1. The old docs described OCR/neural work as out of scope. They now separate
   the historical rotation result from the implemented IE smoke lifecycle.
2. Paddle CUDA and CUDA PyTorch cannot coexist safely in one Windows process.
   Separate D:-backed OCR/layout environments and a JSONL worker boundary were
   added.
3. The optional K-Means adapter imported a plotting-heavy module, making OCR
   require `matplotlib`. Centroid-margin calculation was made local and
   dependency-light.
4. OpenCV 5.0 shadowed PaddleX's OpenCV 4.10 build and lacked
   `HOGDescriptor`. Requirements now pin 4.10.0.84.
5. Preserved scikit-learn 1.8 artifacts load under the Python 3.10 maximum of
   scikit-learn 1.7.2 with a compatibility warning. A public sample produced
   the same cluster/zone and confidence within 1e-10 across both runtimes. The
   branch remains display-only and failure-isolated.
6. CORU pages without OCR reference tokens inflated CER when edit insertions
   were divided by one. Evaluation now excludes unavailable references and
   reports coverage.
7. The first public IE manifest redacted private paths but retained private
   source hashes and duplicate-group fingerprints. Those fields are now blank
   in all 203 public-facing private rows and enforced by the complete verifier;
   full values remain only in the ignored operational manifest.
8. The initial integration report was static evidence. A tracked runner now
   generates D:-backed fixtures/outputs and a text-free report bound to source,
   model, configuration, checkpoint, fixture, and output hashes; the verifier
   independently checks schema and semantics.
9. The rotated OCR smoke allowed empty output. It now requires the real
   cardinal selector to recover a known phrase with exact model identities.
10. The automatic Thai retry threshold missed a real synthetic Thai PDF page,
    and Thai script ratio could exceed one. The calibrated retry threshold and
    bounded script denominator now select the exact Thai route.

## Intentionally preserved behavior

Historical metrics and artifacts were not retrained, deleted, hidden, or
relabeled. K-Means remains a four-zone display result. The exact-angle result
remains available as failed experimental evidence and is disabled in the main
pipeline.
