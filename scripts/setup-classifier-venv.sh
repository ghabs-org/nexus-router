#!/usr/bin/env bash
# scripts/setup-classifier-venv.sh
# Creates a lean venv for the local ONNX classifier runtime.
# Intentionally placed OUTSIDE ~/.local/state/nexus-router/ so it is
# never included in backups and can be rebuilt from scratch any time.
#
# Usage: bash scripts/setup-classifier-venv.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${NEXUS_CLASSIFIER_VENV:-$REPO_DIR/classifier-venv}"

echo "[classifier-venv] Creating lean ONNX runtime venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
# Runtime-only deps: onnxruntime for inference, tokenizers for tokenisation
"$VENV_DIR/bin/pip" install --quiet onnxruntime tokenizers numpy
echo "[classifier-venv] Done: $("$VENV_DIR/bin/python" -c 'import onnxruntime; print("onnxruntime", onnxruntime.__version__)')"
echo ""
echo "Set NEXUS_ROUTER_LOCAL_CLASSIFIER_VENV=$VENV_DIR in your environment if the"
echo "classifier uses a separate venv for inference. Or add to nexus-router/.env."
