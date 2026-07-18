# OCR review contract

The local OCR and LayoutXLM pipeline is the system of record. The MCP server
does not accept source file paths and cannot run extraction. It scans only the
configured local output folder and assigns opaque IDs.

The catalog excludes results marked `processing.private_output=true` and
results below folders named `private`, `private_outputs`, or
`private-evaluation`. Catalog responses contain no filenames, paths, field
values, or OCR text.

A review payload requires an explicit field list and confirmation that those
values may enter the user's signed-in Codex/GPT-5.6 session. OCR text is off by
default, needs a separate confirmation, and is capped at 4,000 characters.

GPT-5.6 may flag inconsistencies, explain uncertainty, or suggest normalized
values. It must not claim to have changed the extraction, and it must not write
back to the local result. No OpenAI API key is used by this workflow.
