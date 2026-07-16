# Final public error analysis

This analysis uses executed public test predictions. Counts below come from the bounded end-to-end angle grid; no private text or filenames are included.

| Root-cause signal | Count |
|---|---:|
| entity f1 below 0 70 with reference | 54 |
| no table detected | 39 |
| canonical field miss | 36 |
| detection f1 below 0 70 | 36 |
| ocr coverage below 0 70 | 36 |
| relation f1 below 0 60 with reference | 18 |
| empty ocr | 0 |

## Weakest supported classes on the full locked layout test

| Head | Class | Support | F1 |
|---|---|---:|---:|
| entity | B-HEADER | 85 | 0.1277 |
| entity | I-HEADER | 173 | 0.4543 |
| entity | I-QUESTION | 796 | 0.6803 |
| entity | B-ANSWER | 564 | 0.7107 |
| entity | B-QUESTION | 613 | 0.7882 |
| relation | QUESTION_ANSWER | 595 | 0.3874 |
| relation | OTHER_RELATION | 183 | 0.7366 |
| canonical evidence | document_title | 730 | 0.9376 |
| canonical evidence | total_amount | 3038 | 0.9558 |
| canonical evidence | organization_name | 3902 | 0.9638 |
| canonical evidence | email | 1854 | 0.9739 |
| canonical evidence | address | 14841 | 0.9829 |

## Locked-test dataset slices

| Dataset | Examples | Entity F1 | Relation F1 | Canonical evidence F1 |
|---|---:|---:|---:|---:|
| fatura | 1584 | 0.9994639811581255 | 0.0 | 0.9841247580848737 |
| funsd | 30 | 0.745405537858368 | 0.4667681380010147 | 0.0 |
| sroie | 146 | 0.8680535765175263 | 0.0 | 0.8254109284762328 |

## Lowest bounded OCR coverage angles

| Angle | Pages | Recognized-text coverage | WER | Entity F1 |
|---:|---:|---:|---:|---:|
| 45 | 3 | 0.4026224138485081 | 0.7789880295143452 | 0.16370035193564605 |
| 0 | 3 | 0.4027889972234873 | 0.6998877146245568 | 0.17720042680959824 |
| 90 | 3 | 0.4027889972234873 | 0.7009133556501977 | 0.18301423662248406 |
| 60 | 3 | 0.40369467889854044 | 0.7693043798306957 | 0.18077430180990053 |
| 135 | 3 | 0.40376744369869755 | 0.7660010165273322 | 0.17023629964806433 |

These are bounded diagnostic signals, not all ground-truth error labels: for example, `no table detected` records output availability because the sampled annotations do not provide a compatible table benchmark. Implemented mitigations include cardinal-plus-polygon fine deskew, real PaddleOCR/hybrid training streams, class-weighted multi-task loss, calibrated abstention, arithmetic validation, and geometry table fallback. Remaining misses stay visible in the measured metrics.

Held-out entity micro-F1: 0.9806603808577088.
Held-out calibrated entity micro-F1: 0.9812706023374288.
Held-out relation F1: 0.4667681380010147.
Held-out calibrated relation F1: 0.46324007807417045.
Held-out canonical evidence F1: 0.9792278412180533.
Held-out calibrated canonical evidence F1: 0.9814241915569551.

Known bottlenecks: relation labels exist only in FUNSD; the Windows runtime has no compatible Detectron2 visual backbone; CORU QA has no token polygons; and the public corpus has no compatible labeled Thai benchmark. These constraints are not treated as successes.
