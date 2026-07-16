# Public annotation mapping report

The adapters preserve raw annotations unchanged and write normalized JSON to the configured generated-data root.

## Mapping

- SROIE polygon OCR plus entity JSON: tokens are retained; company, address, date, and total become canonical fields.
- FUNSD form JSON: header, question, answer, and other labels map to the universal entity set; links become relations.
- FATURA LayoutLM-HF plus original JSON: words and boxes are retained; non-background field tags map to VALUE and table tags to TABLE_CELL.
- CORU receipt QA: full-document question-answer supervision is retained for later OCR alignment. KIE YOLO regions are excluded from supervised fitting because no verified text/class map is present.
- CORU OCR line crops and CSV-only item records remain excluded because they are not full-document image examples.

## Counts

- Source pages inspected: 52636
- Normalized usable pages: 12433
- Conversion errors: 135
- Private Gmail fit rows: 0
- Usable by dataset: `{"coru": 1261, "fatura": 10000, "funsd": 199, "sroie": 973}`
- Usable by split: `{"test": 2857, "train": 6575, "validation": 3001}`
- Exclusions: `{"line crops are not full document pages": 30148, "malformed_source_annotation": 4, "missing_source_annotation": 1212, "source image is unreadable, invalid, or empty": 199, "unsupported_annotation_semantics": 8437}`

## Known alignment boundary

CORU QA answers have no source token polygons. They remain valid field/relation supervision only after local OCR alignment passes the configured coverage gate. They are not silently treated as token labels.
