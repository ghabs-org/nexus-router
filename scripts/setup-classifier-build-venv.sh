#!/usr/bin/env bash
# scripts/setup-classifier-build-venv.sh
# Creates the build venv used by nightly_retrain_classifier.sh to fine-tune
# ModernBERT. This venv needs torch + datasets + accelerate (NOT just the
# lean ONNX runtime deps in setup-classifier-venv.sh).
#
# Default location: ~/.local/state/nexus-router/venvs/router-classifier-build/
# Override with NEXUS_ROUTER_CLASSIFIER_BUILD_VENV env var.
#
# Idempotent: safe to re-run to repair a drifted venv (will pip-install --upgrade).
set -euo pipefail

VENV_DIR="${NEXUS_ROUTER_CLASSIFIER_BUILD_VENV:-$HOME/.local/state/nexus-router/venvs/router-classifier-build}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[classifier-build-venv] target=$VENV_DIR"

mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

# Install build/retrain stack from repo requirements (includes torch + datasets + accelerate).
# CPU-only torch extra-index keeps the wheel small and avoids CUDA on this box.
"$VENV_DIR/bin/pip" install --quiet \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$ROOT_DIR/requirements.txt"

echo "[classifier-build-venv] installed:"
"$VENV_DIR/bin/pip" list 2>/dev/null \
  | grep -iE '^(torch|transformers|datasets|accelerate|numpy|tokenizers|onnxruntime|pyyaml)\s' \
  || true

echo "[classifier-build-venv] OK"
