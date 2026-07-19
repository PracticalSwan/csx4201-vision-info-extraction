# OCR and Layout Environment Setup

## Why there are two environments

PaddlePaddle GPU and CUDA PyTorch load incompatible cuDNN DLLs in one Windows
process. `scripts/setup_ie_environment.ps1` therefore creates two Python 3.10
virtual environments on D::

```text
D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr
D:\CSX4201\vision-info-extraction-assets\environments\ie-layout
```

The OCR environment contains PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, PaddleX
3.7.2, and CPU-only PyTorch 2.8.0 required by PaddleX/ModelScope. The layout
environment contains PyTorch 2.8.0+cu128, Transformers 4.57.6, SentencePiece,
and Accelerate.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ie_environment.ps1
```

The script verifies Python 3.10, C:/D: capacity, packages, independent GPU
runtimes, and D:-backed cache variables. Use limits sparingly: run a bounded
profile before a large download, alignment job, or training run.
It discovers Python through `py -3.10`; pass `-Python310 <path>` only when the
Windows launcher is unavailable.

The configured external root is:

```text
D:\CSX4201\vision-info-extraction-assets
```

It contains environments, Paddle/Hugging Face/Torch/pip caches, temporary
files, OCR cache, aligned datasets, checkpoints, generated documents, and
private outputs. These are ignored by Git.

## Download and verify models

```powershell
$ocr = 'D:\CSX4201\vision-info-extraction-assets\environments\ie-ocr\Scripts\python.exe'
& $ocr scripts/download_ocr_models.py
& $ocr scripts/verify_ocr_models.py --device gpu:0
& $ocr scripts/print_environment_report.py `
  --layout-python 'D:\CSX4201\vision-info-extraction-assets\environments\ie-layout\Scripts\python.exe'
```

Required exact identities:

- detector: `PP-OCRv6_medium_det`;
- general recognizer: `PP-OCRv6_medium_rec`;
- Thai recognizer: `th_PP-OCRv5_mobile_rec`.

The download report records directories and per-file SHA-256 hashes. Model
verification re-hashes those files before initializing the routes. A partial,
missing, or changed artifact fails closed.

The rotated checks are end-to-end selector assertions. A 90-degree fixture
must return nonempty OCR, select a nonzero cardinal correction, and recover
`INVOICE`, `TOTAL`, and `123.45`. A separate 17-degree fixture must recover the
same phrase through automatic fine deskew with reliable line evidence. An
empty engine call cannot pass either check.

The final public dev-only preprocessing ablation selected the `original`
profile. Grayscale normalization and optional Paddle orientation modules tied
its alignment score but did not improve it; denoising reduced coverage.
Raster-to-PDF tests tied across 200/250/300 DPI, so 200 DPI remains the lower
cost default. The locked test and private documents were never used for this
selection.

## CPU mode

Pass `--device cpu` to OCR verification/inference and `--device cpu` to the
layout training script where supported. CPU mode is slower. Do not install a
CUDA Torch wheel into the OCR environment; doing so reintroduces the DLL
collision.

## Troubleshooting

- Storage gate failure: free space or change the configured external root;
  never bypass the 15 GiB reserve.
- Paddle/Torch DLL error: verify that the command uses the correct interpreter
  and that the layout worker is a subprocess.
- Missing model/hash mismatch: rerun `download_ocr_models.py`; do not copy an
  unverified partial cache into place.
- CUDA unavailable: rerun the environment report and inspect driver/runtime
  probes before falling back to CPU.
- K-Means maintenance artifacts remain in their original scikit-learn 1.8
  joblib form. Inference uses the hash-bound `inference_params.npz` numeric
  export, which was checked against all 7,520 public train/validation/test
  feature rows and avoids cross-version pickle loading in Python 3.10.
