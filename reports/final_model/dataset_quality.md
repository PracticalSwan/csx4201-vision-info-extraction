# Final model dataset quality

Build: `final-6be3e0b46b0a4e4c`.
Normalized public pages: 12433.
Usable public fit pages: 11172.
Usable examples across streams: 11684.
Private/Gmail fit rows: **0**.

The ground-truth stream is immutable. PaddleOCR examples are inference-realistic; hybrid examples are training-only and preserve supervised targets with explicit masks.
CORU remains wholly unseen-domain evaluation data because it has QA answers but no compatible token polygons.

| Dataset | Normalized | Fit pages | Examples | Splits |
|---|---:|---:|---:|---|
| coru | 1261 | 0 | 0 | unseen_domain_test=1261 |
| fatura | 10000 | 10000 | 10184 | dev_calibration=594, dev_select=996, test_in_domain=1584, train=6826 |
| funsd | 199 | 199 | 351 | dev_calibration=10, dev_select=20, test_in_domain=30, train=139 |
| sroie | 973 | 973 | 1149 | dev_calibration=49, dev_select=97, test_in_domain=146, train=681 |

Alignment and target-retention details are machine-readable in `alignment_metrics.json`.
