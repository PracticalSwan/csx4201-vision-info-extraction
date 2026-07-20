# OCR Model portable package

This package runs the finished PaddleOCR + LayoutXLM information-extraction
pipeline on local images and PDFs. It includes the trained LayoutXLM checkpoint,
the three pinned PaddleOCR model directories, calibration, rotation-display
artifacts, schemas, launchers, and a safe synthetic sample. It does not include
raw training data, private Gmail documents, private outputs, credentials, or an
OpenAI API integration.

Extraction is local and does not require an OpenAI API key. The optional Codex
review workflow uses the user's signed-in Codex session and is described in
`docs/CODEX_INTEGRATION.md`.

## Windows quick start

Requirements:

- Windows 10 or 11, 64-bit
- Python 3.10 from python.org, with the `py` launcher enabled
- Internet access for the one-time dependency installation
- At least 20 GB of free disk space for Python environments and Docker caches

Run:

1. Extract `OCR_Model.zip` to a normal writable folder.
2. Double-click `setup_windows.bat`. The default setup is CPU-only and works
   without NVIDIA hardware.
3. Double-click `launch_windows.bat`.
4. Select an image/PDF and click **Extract document**.

The upload card previews the complete selected image or the first page of a
PDF before extraction. Selecting a different document clears the previous
status and results. A run uses one compact loading indicator; it does not cover
each output tab with separate spinners.

After setup, the model can also be run in one command:

```powershell
.\run_cli.bat "C:\path\to\document.pdf"
```

Or with the lightweight app Python:

```powershell
.\.runtime\app\Scripts\python.exe .\extract_document.py "C:\path\to\image.png"
```

The owner of the original development machine can reuse the already verified
OCR/layout environments without altering them:

```powershell
.\setup_windows.ps1 -Device gpu -ReuseExisting
```

This writes only `runtime.local.json` and the lightweight app environment inside
the package. It does not modify the trained checkpoint or existing OCR/layout
environments.

## macOS quick start

Native macOS Paddle/LayoutXLM environments are not claimed or tested. The
supported macOS path uses Docker Desktop:

1. Install and start Docker Desktop.
2. Extract the package.
3. In Terminal, run:

   ```bash
   cd /path/to/OCR_Model
   bash launch_macos.command
   ```

4. Open <http://127.0.0.1:7860>.

The Compose service is pinned to `linux/amd64` so it can run on Intel Macs and,
through Docker Desktop emulation, Apple Silicon Macs. It is CPU-only and can be
slow. The container and host port are local; the GUI is published only at
`127.0.0.1`.

The exact `linux/amd64` image was built and exercised on Windows Docker
Desktop: all readiness probes passed and the full bundled sample produced the
same field values, OCR text, entity triplets, and relation triplets as the
native GPU run. No physical Intel or Apple Silicon Mac was available, so that
host-specific launch remains explicitly untested.

## Outputs

Every run creates a timestamped folder under `outputs/` containing:

- `document_result.json`: schema-validated full result
- `pages/page_###_ocr.json`: OCR words, lines, confidence, and provenance
- `pages/page_###_entities.json`: learned and generic entities
- `pages/page_###_relations.json`: key/value relations
- `pages/page_###_visualization.png`: visual overlay
- `portable_run.log`: local diagnostic log

The GUI shows a field table, combined OCR text, full JSON, page visualizations,
and a downloadable ZIP of that run. The OCR text and run-log panes have fixed
heights with independent vertical scrolling, so long output remains usable.
Terminal color sequences are removed from the displayed log.

In the GUI, **Maximum PDF pages = 0** means process every page.

## Readiness and troubleshooting

Run the fast check:

```powershell
.\.runtime\app\Scripts\python.exe .\doctor.py
```

Run framework imports too:

```powershell
.\.runtime\app\Scripts\python.exe .\doctor.py --probe
```

`MODEL_MANIFEST.json` records the included model filenames, sizes, and SHA-256
hashes. Every published `OCR_Model.zip` is accompanied by
`OCR_Model.zip.sha256`. On Windows, recompute and compare the digest before
extracting:

```powershell
Get-FileHash .\OCR_Model.zip -Algorithm SHA256
Get-Content .\OCR_Model.zip.sha256
```

Users and judges can download the archive and checksum from the public
[`v1.0.0-build-week` Release](https://github.com/PracticalSwan/csx4201-vision-info-extraction/releases/tag/v1.0.0-build-week).

The package includes the project's MIT `LICENSE` and `CONTRIBUTING.md`.
LayoutXLM-derived weights and other third-party components retain the upstream
licenses documented in `docs/THIRD_PARTY_NOTICES.md`.

The final layout checkpoint must have this SHA-256:

```text
34c7a26e78d6285a2739e1b61839eadfd0e686ccbcf57f9cb47997c12cef2189
```

The package keeps the original scikit-learn joblib files for provenance, but
the display-only rotation branch loads `models/kmeans_rotation/inference_params.npz`.
That hash-bound numeric export avoids unsupported cross-version pickle loading
and matched the original scaler/PCA/K-Means output on all 7,520 public
train/validation/test feature rows with zero cluster-label differences.

If Windows blocks a downloaded archive, right-click the ZIP, choose
**Properties**, select **Unblock**, then extract it again. If port 7860 is in
use, launch `app.py --port 7861`.

For CPU portability, the adapter disables Paddle's oneDNN/MKLDNN optimization;
the pinned PP-OCRv6 models otherwise trigger a PaddlePaddle 3.3 Linux executor
error. This changes the execution backend, not the bundled weights or output
schema.

## Privacy and limitations

- Do not put private documents in a folder that will be re-zipped or shared.
- Outputs remain local unless the user deliberately approves a bounded Codex
  review payload.
- The K-Means rotation zone is display-only and never controls OCR/extraction.
- The failed exact-angle estimator remains disabled.
- Reference-token entity F1 is 0.9813, while bounded end-to-end entity F1 is
  0.1314–0.1830 because OCR remains the main bottleneck. Relation learning is
  limited by FUNSD-only supervision.
- This is an academic/noncommercial model package, not a production system.
  Human review is required for financial, legal, or other consequential use.

See `docs/THIRD_PARTY_NOTICES.md` before redistributing the package.
