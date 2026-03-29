#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m src.cli quota-sync "$@"
