#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI is not available on PATH."
  exit 1
fi
codex mcp remove ocr_model >/dev/null 2>&1 || true
codex mcp add ocr_model --env "OCR_MODEL_HOME=/workspace" -- \
  docker compose -f "$ROOT/docker-compose.yml" run --rm --no-deps -T ocr-model \
  /opt/venvs/app/bin/python /workspace/mcp_server.py
echo "Installed local MCP server: ocr_model"
echo 'Open this folder in Codex, select GPT-5.6, and invoke $review-ocr-document.'
echo "No OpenAI API key was requested or configured."
