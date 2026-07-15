# Information-Extraction Workflow

## Normalize annotations

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/normalize_ie_annotations.py --force
& $ocr scripts/verify_ie_annotations.py
```

The normalizer writes public schema records below
`data/processed/normalized_ie_annotations` (ignored bulk data), a public-safe
manifest, mapping report, schema, summary, and categorized error CSV. It never
modifies raw annotations. The private operational manifest is ignored and does
not contain training-eligible rows.

## Prepare aligned model data

```powershell
& $ocr scripts/prepare_model_dataset.py --profile smoke --device gpu:0 --force
```

PaddleOCR tokens are aligned to source entities using geometry and normalized
text. Examples below the configured 0.70 coverage are excluded. Every output
records the source page, OCR route/model hashes, label map, split, and privacy
status. Gmail rows are rejected from training by construction.

## Train

```powershell
$layout = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
& $layout scripts/train_layout_model.py --profile smoke --device cuda
```

The model uses multilingual token and normalized 2D-box embeddings. Windows
are length 512 with overlap. Training applies deterministic dynamic rotations,
mixed precision, gradient accumulation, clipping, validation selection, and
early stopping. The final save includes tokenizer, label maps, model weights,
relation head, and state/provenance. Reloaded logits must match within the
recorded tolerance.

Current checkpoint:

```text
D:\CSX4201\vision-info-extraction-assets\checkpoints\layoutxlm\smoke
```

This is a smoke checkpoint. Its three public training examples and 10.07%
validation token accuracy do not support a quality claim.

## Inference behavior

The main process selects OCR orientation and route. A persistent layout worker
returns token labels and merged entities in original page coordinates.
Geometry rules generate key-value candidates. Canonical rules select only
evidence-backed values and warn on conflicts. Unknown types additionally parse
generic `key: value` text.

All results validate against the JSON Schema before atomic write. No field is
invented: unavailable canonical fields are `null`.

## Reproducible integration evidence

```powershell
& $ocr scripts/run_integration_smoke.py --device gpu:0
python scripts/verify_information_extraction.py --complete
```

The runner creates only synthetic image, 45-degree, Thai, and two-page PDF
fixtures below the ignored external D: root. It writes full outputs there and
commits a text-free report with assertions and SHA-256 evidence. Complete
verification re-hashes source/model/config/checkpoint/fixture/output artifacts,
revalidates output schemas, and independently checks result semantics.
