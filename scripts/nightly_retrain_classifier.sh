#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_VENV="${NEXUS_ROUTER_CLASSIFIER_BUILD_VENV:-$HOME/.local/state/nexus-router/venvs/router-classifier-build}"
PYTHON_BIN="$BUILD_VENV/bin/python"
STATE_ROOT="${NEXUS_ROUTER_STATE_ROOT:-$HOME/.local/state/nexus-router}"
DATA_DIR="$STATE_ROOT/artifacts/router-classifier/data"
REPORT_DIR="$STATE_ROOT/reports"
LOG_DIR="$STATE_ROOT/logs"
STAMP_DIR="$STATE_ROOT/retrain"
EXPORT_PATH="${NEXUS_ROUTER_TRAINING_EXPORT:-/tmp/router-training-nightly.jsonl}"
TRAIN_FILE="$DATA_DIR/train.jsonl"
EVAL_FILE="$DATA_DIR/eval.jsonl"

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$LOG_DIR" "$STAMP_DIR"
LOG_FILE="$LOG_DIR/nightly-retrain-classifier-$(date -u +%F).log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRY_RUN="${1:-}"
echo "[nightly-retrain] start $(date -u +%FT%TZ)"
echo "[nightly-retrain] root=$ROOT_DIR"
echo "[nightly-retrain] build_venv=$BUILD_VENV"
echo "[nightly-retrain] mode=${DRY_RUN:-live}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[nightly-retrain] missing build python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

$PYTHON_BIN scripts/feedback_calibration_report.py > "$REPORT_DIR/calibration-$(date -u +%F).json"
$PYTHON_BIN scripts/export_classifier_training_data.py --output "$EXPORT_PATH" --min-samples 1

$PYTHON_BIN - <<'PY'
import json, random, pathlib
src = pathlib.Path("/tmp/router-training-nightly.jsonl")
out_dir = pathlib.Path.home() / ".local/state/nexus-router/artifacts/router-classifier/data"
out_dir.mkdir(parents=True, exist_ok=True)
rows=[]
with src.open() as f:
    for line in f:
        line=line.strip()
        if line:
            rows.append(json.loads(line))
if not rows:
    raise SystemExit("no training rows exported")
random.Random(42).shuffle(rows)
split=max(1, int(len(rows)*0.85))
train=rows[:split]
eval_=rows[split:]
(out_dir/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in train)+'\n')
(out_dir/'eval.jsonl').write_text('\n'.join(json.dumps(r) for r in eval_)+'\n')
print({'total': len(rows), 'train': len(train), 'eval': len(eval_)})
PY

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[nightly-retrain] dry-run complete before train/export/deploy"
  exit 0
fi

$PYTHON_BIN scripts/train_router_classifier.py
$PYTHON_BIN scripts/export_router_classifier_onnx.py
bash "$ROOT_DIR/scripts/refresh_registry.sh"

echo "$(date -u +%FT%TZ)" > "$STAMP_DIR/last_success_utc.txt"
echo "[nightly-retrain] done $(date -u +%FT%TZ)"
