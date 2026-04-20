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
EXPORT_PATH="${NEXUS_ROUTER_TRAINING_EXPORT:-$DATA_DIR/export.jsonl}"
TRAIN_FILE="$DATA_DIR/train.jsonl"
EVAL_FILE="$DATA_DIR/eval.jsonl"
SPLIT_SUMMARY="$REPORT_DIR/training-split-$(date -u +%F).json"

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

EXPORT_PATH="$EXPORT_PATH" TRAIN_FILE="$TRAIN_FILE" EVAL_FILE="$EVAL_FILE" SPLIT_SUMMARY="$SPLIT_SUMMARY" $PYTHON_BIN - <<'PY'
import json, os, random, pathlib
src = pathlib.Path(os.environ["EXPORT_PATH"])
train_path = pathlib.Path(os.environ["TRAIN_FILE"])
eval_path = pathlib.Path(os.environ["EVAL_FILE"])
summary_path = pathlib.Path(os.environ["SPLIT_SUMMARY"])
train_path.parent.mkdir(parents=True, exist_ok=True)
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
train_path.write_text('\n'.join(json.dumps(r) for r in train)+'\n')
eval_path.write_text('\n'.join(json.dumps(r) for r in eval_)+'\n')
summary = {'total': len(rows), 'train': len(train), 'eval': len(eval_), 'export_path': str(src), 'train_file': str(train_path), 'eval_file': str(eval_path)}
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
print(summary)
PY

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[nightly-retrain] dry-run complete before train/export/deploy"
  exit 0
fi

$PYTHON_BIN scripts/train_router_classifier.py
$PYTHON_BIN scripts/export_router_classifier_onnx.py
bash "$ROOT_DIR/scripts/refresh_registry.sh"

if command -v docker >/dev/null 2>&1; then
  docker compose -f "$ROOT_DIR/deploy/docker-compose.yml" up -d nexus-router
fi

echo "$(date -u +%FT%TZ)" > "$STAMP_DIR/last_success_utc.txt"
echo "[nightly-retrain] done $(date -u +%FT%TZ)"
