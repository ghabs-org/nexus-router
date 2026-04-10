#!/usr/bin/env bash
# scripts/setup-classifier-venv.sh
# Creates a lean venv for the local ONNX classifier runtime.
#
# Default location: ~/.local/lib/nexus-router/classifier-venv/
# Override with NEXUS_ROUTER_LOCAL_CLASSIFIER_VENV env var.
#
# This path is intentionally:
#   - Outside the git repo (not a tracked artifact)
#   - Outside ~/.local/state/nexus-router/ (not included in backups)
#   - Rebuildable any time from this script
#
# Usage: bash scripts/setup-classifier-venv.sh
set -euo pipefail

VENV_DIR="${NEXUS_ROUTER_LOCAL_CLASSIFIER_VENV:-$HOME/.local/lib/nexus-router/classifier-venv}"

echo "[classifier-venv] Creating lean ONNX runtime venv at $VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
# Runtime-only deps: onnxruntime for inference, tokenizers for tokenisation
"$VENV_DIR/bin/pip" install --quiet onnxruntime tokenizers numpy
echo "[classifier-venv] Done: $("$VENV_DIR/bin/python" -c 'import onnxruntime; print("onnxruntime", onnxruntime.__version__)')"
echo ""
echo "To use: export NEXUS_ROUTER_LOCAL_CLASSIFIER_VENV=$VENV_DIR"
echo "Or add NEXUS_ROUTER_LOCAL_CLASSIFIER_VENV=$VENV_DIR to your .env"
