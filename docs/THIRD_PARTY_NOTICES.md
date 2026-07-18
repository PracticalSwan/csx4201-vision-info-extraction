# Distribution and third-party notices

This portable package is intended for academic and noncommercial evaluation.
It contains original project code plus model artifacts with separate upstream
terms. No raw datasets or private Gmail documents are distributed.

## LayoutXLM checkpoint

The fine-tuned checkpoint is derived from `microsoft/layoutxlm-base`. The
project configuration records the model license as
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**:

<https://creativecommons.org/licenses/by-nc-sa/4.0/>

This includes attribution, noncommercial-use, and share-alike conditions. Do
not use or redistribute the checkpoint for commercial purposes without
confirming that the required rights are available.

## PaddleOCR model artifacts

The included model cards for:

- `PP-OCRv6_medium_det`
- `PP-OCRv6_medium_rec`
- `th_PP-OCRv5_mobile_rec`

declare the **Apache License 2.0**:

<https://www.apache.org/licenses/LICENSE-2.0>

Their original `README.md` model cards are retained inside each bundled model
directory.

## Runtime dependencies

The setup scripts install pinned or bounded Python dependencies from their
official package indexes. Each dependency keeps its own license. Docker,
Python, PyTorch, PaddlePaddle, PaddleOCR, Transformers, Gradio, the MCP Python
SDK, and the remaining packages are not relicensed by this project.

## Original project code

No blanket open-source license has been granted for the original project code.
Recipients may run this package only within the permission given by its owner
and the upstream model licenses. Before broader publication, the owner should
choose and add an explicit code license compatible with the model's
noncommercial/share-alike obligations.
