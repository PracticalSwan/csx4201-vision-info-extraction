#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker Desktop is required. Install and start it, then run this file again."
  exit 1
fi
echo "Building/starting the local CPU runtime. First launch can take a while."
echo "When ready, open http://127.0.0.1:7860"
docker compose up --build
