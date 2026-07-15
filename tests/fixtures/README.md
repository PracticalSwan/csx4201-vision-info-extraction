# tests/fixtures

Synthetic dataset fixtures are generated at runtime inside pytest's `tmp_path`
by `tests/conftest.py` (see `build_sroie`, `build_funsd`, `build_fatura`,
`build_coru`, `build_gmail`). This keeps the fixtures tiny, deterministic, and
fully isolated from the real data under `data/raw/`.

Drop any future static fixture files (e.g., hand-crafted malformed PDFs) into
this directory and reference them via `Path(__file__).parent / "fixtures"`.
