# Final vision information-extraction pre-model card

## Model

The checkpoint is a LayoutXLM-initialized multilingual text-plus-normalized-2D-layout encoder with trained entity, document-type, canonical-evidence, and real relation heads. PaddleOCR runs in an isolated process path with exact general and Thai recognizers. The K-Means model is display-only.

Checkpoint: `D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm_multitask\final`.
Checkpoint model SHA-256: `34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189`.
Training: 4 completed epochs, 7812 optimizer steps; best epoch 4 with selection score 0.8241600322529928.
License inherited from the base checkpoint: `CC-BY-NC-SA-4.0`.

## Training data and privacy

Public fit pages: 11172; examples: 11684; Gmail/private fit rows: 0.
Private Gmail documents are operational test only and never train, calibrate, or select the model.

## Measured quality

Held-out entity micro-F1: raw 0.9806603808577088; calibrated/abstained 0.9812706023374288; raw macro-F1 0.7290191953730021.
Held-out relation F1: raw 0.4667681380010147; calibrated/abstained 0.46324007807417045.
Held-out canonical evidence F1: raw 0.9792278412180533; calibrated/abstained 0.9814241915569551.
Bounded upright end-to-end OCR text coverage: 0.4027889972234873; WER: 0.6998877146245568.
CORU unseen-domain answer-text recall: 0.7853036740814796 on 100 sampled pages.

## Intended use

Local extraction from images and PDFs containing receipts, invoices, forms, and unfamiliar documents. Outputs include OCR evidence, generic entities and relations, canonical fields with abstention, and geometry-based tables.

## Limitations

This is a bounded academic pre-model, not a production or high-stakes decision system. The visual backbone is unavailable on this Windows runtime, relation supervision is sparse, Thai quality lacks a labeled public benchmark, arbitrary-angle end-to-end evaluation is bounded, and low-confidence/unsupported fields return null. Review financial and legal outputs against the source document.
