#!/usr/bin/env bash
# scripts/setup-classifier-build-venv.sh
# Creates the build venv used by nightly_retrain_classifier.sh to fine-tune
# ModernBERT. This venv needs torch + datasets + accelerate + onnx export deps.
#
# Default location: ~/.local/state/nexus-router/venvs/router-classifier-build/
# Override with NEXUS_ROUTER_CLASSIFIER_BUILD_VENV env var.
#
# Idempotent: detects Python minor-version drift and recreates cleanly.
set -euo pipefail

VENV_DIR="${NEXUS_ROUTER_CLASSIFIER_BUILD_VENV:-$HOME/.local/state/nexus-router/venvs/router-classifier-build}"
PYTHON_BIN="${NEXUS_ROUTER_BUILD_PYTHON:-/usr/bin/python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[classifier-build-venv] missing base python: $PYTHON_BIN" >&2
  exit 1
fi

TARGET_MM="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "[classifier-build-venv] target=$VENV_DIR python=$PYTHON_BIN ($TARGET_MM)"

if [[ -d "$VENV_DIR" ]]; then
  if [[ ! -d "$VENV_DIR/lib/python$TARGET_MM/site-packages" ]] \
    || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 \
    || ! "$VENV_DIR/bin/python" -c "import sys; raise SystemExit(0 if f'{sys.version_info[0]}.{sys.version_info[1]}' == '$TARGET_MM' else 1)" >/dev/null 2>&1; then
    echo "[classifier-build-venv] existing venv is drifted/broken — recreating for Python $TARGET_MM"
    rm -rf "$VENV_DIR"
  fi
fi

mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet --upgrade pip

# Install build/retrain stack plus ONNX export packages (onnx + onnxscript for torch.onnx.export).
"$VENV_DIR/bin/pip" install --quiet \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$ROOT_DIR/requirements.txt" \
  onnx \
  onnxscript

echo "[classifier-build-venv] installed:"
"$VENV_DIR/bin/pip" list 2>/dev/null \
  | grep -iE '^(torch|transformers|datasets|accelerate|numpy|tokenizers|onnx|onnxruntime|onnxscript|pyyaml|httpx)\s' \
  || true

echo "[classifier-build-venv] OK"
