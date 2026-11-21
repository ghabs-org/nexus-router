#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_VENV="${NEXUS_ROUTER_CLASSIFIER_BUILD_VENV:-$HOME/.local/state/nexus-router/venvs/router-classifier-build}"
PYTHON_BIN="$BUILD_VENV/bin/python"
STATE_ROOT="${NEXUS_ROUTER_STATE_ROOT:-$HOME/.local/state/nexus-router}"
DATA_DIR="$STATE_ROOT/artifacts/router-classifier/data"
ONNX_ROOT="$STATE_ROOT/artifacts/router-classifier"
LIVE_ONNX="$ONNX_ROOT/onnx"
CANDIDATE_ONNX="$ONNX_ROOT/onnx.candidate"
PREV_ONNX="$ONNX_ROOT/onnx.prev"
CHECKPOINT_DIR="$ONNX_ROOT/checkpoints"
SCORER_DIR="$STATE_ROOT/artifacts/scorer"
REPORT_DIR="$STATE_ROOT/reports"
LOG_DIR="$STATE_ROOT/logs"
STAMP_DIR="$STATE_ROOT/retrain"
EXPORT_PATH="${NEXUS_ROUTER_TRAINING_EXPORT:-$DATA_DIR/export.jsonl}"
TRAIN_FILE="$DATA_DIR/train.jsonl"
EVAL_FILE="$DATA_DIR/eval.jsonl"
SPLIT_SUMMARY="$REPORT_DIR/training-split-$(date -u +%F).json"
MIN_NEW_VERIFIED="${NEXUS_ROUTER_MIN_NEW_VERIFIED_FEEDBACK:-15}"

export NEXUS_ROUTER_STATE_ROOT="$STATE_ROOT"
export NEXUS_ROUTER_DB_PATH="${NEXUS_ROUTER_DB_PATH:-$STATE_ROOT/data/router.sqlite}"
export NEXUS_ROUTER_ARTIFACTS_DIR="${NEXUS_ROUTER_ARTIFACTS_DIR:-$STATE_ROOT/artifacts}"
export TOKENIZERS_PARALLELISM=false

DRY_RUN=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    *) echo "[nightly-retrain] unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DATA_DIR" "$REPORT_DIR" "$LOG_DIR" "$STAMP_DIR" "$SCORER_DIR" "$ONNX_ROOT"
LOG_FILE="$LOG_DIR/nightly-retrain-classifier-$(date -u +%F).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[nightly-retrain] start $(date -u +%FT%TZ)"
echo "[nightly-retrain] root=$ROOT_DIR"
echo "[nightly-retrain] build_venv=$BUILD_VENV"
echo "[nightly-retrain] mode=$([[ "$DRY_RUN" == 1 ]] && echo dry-run || echo live) force=$FORCE"

# Self-heal: if python binary is missing/broken or required ML packages fail, rebuild venv.
if [[ ! -x "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c "import torch, transformers, accelerate, onnxruntime, onnx" >/dev/null 2>&1; then
  echo "[nightly-retrain] build venv missing/drifted or incomplete — bootstrapping"
  bash "$ROOT_DIR/scripts/setup-classifier-build-venv.sh"
fi

cd "$ROOT_DIR"

$PYTHON_BIN scripts/feedback_calibration_report.py \
  --db "$NEXUS_ROUTER_DB_PATH" \
  > "$REPORT_DIR/calibration-$(date -u +%F).json"

$PYTHON_BIN scripts/export_classifier_training_data.py \
  --verified-only \
  --output "$EXPORT_PATH" \
  --split-dir "$DATA_DIR" \
  --eval-ratio "${RETRAIN_EVAL_RATIO:-0.15}" \
  --seed "${RETRAIN_SEED:-42}" \
  --min-samples 1

cp "$DATA_DIR/split_summary.json" "$SPLIT_SUMMARY"

TOTAL_VERIFIED="$("$PYTHON_BIN" -c "import json; print(json.load(open('$SPLIT_SUMMARY'))['total_verified'])")"
PREV_VERIFIED=0
if [[ -f "$STAMP_DIR/last_verified_count.txt" ]]; then
  PREV_VERIFIED="$(cat "$STAMP_DIR/last_verified_count.txt" 2>/dev/null || echo 0)"
fi
DELTA_VERIFIED=$(( TOTAL_VERIFIED - PREV_VERIFIED ))
echo "[nightly-retrain] verified_records=$TOTAL_VERIFIED prev_verified=$PREV_VERIFIED delta=$DELTA_VERIFIED min_new=$MIN_NEW_VERIFIED"

if [[ "$DRY_RUN" != 1 && "$FORCE" != 1 && -f "$STAMP_DIR/last_success_utc.txt" && -f "$STAMP_DIR/last_verified_count.txt" ]]; then
  if (( DELTA_VERIFIED < MIN_NEW_VERIFIED )); then
    echo "[nightly-retrain] skipping train: only $DELTA_VERIFIED new verified samples (< $MIN_NEW_VERIFIED); pass --force to override"
    exit 0
  fi
fi

SCORER_CANDIDATE="$SCORER_DIR/fitted_weights.candidate.json"
SCORER_ACTIVE="$SCORER_DIR/fitted_weights.active.json"
if ! $PYTHON_BIN scripts/fit_scoring_weights.py \
  --candidate-output "$SCORER_CANDIDATE" \
  --active-output "$SCORER_ACTIVE" \
  --min-samples-per-task 40 \
  --activate-on-golden-pass; then
  echo "[nightly-retrain] WARN: fit_scoring_weights exited non-zero; continuing classifier pipeline"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[nightly-retrain] dry-run complete before train/export/eval"
  exit 0
fi

$PYTHON_BIN scripts/train_router_classifier.py \
  --train-file "$TRAIN_FILE" \
  --eval-file "$EVAL_FILE" \
  --output-dir "$CHECKPOINT_DIR" \
  --epochs "${RETRAIN_EPOCHS:-4.0}" \
  --batch-size "${RETRAIN_BATCH_SIZE:-4}" \
  --grad-accum "${RETRAIN_GRAD_ACCUM:-4}" \
  --max-length "${RETRAIN_MAX_LENGTH:-192}" \
  --learning-rate "${RETRAIN_LR:-1e-4}" \
  --freeze-layers "${RETRAIN_FREEZE_LAYERS:-18}" \
  --seed "${RETRAIN_SEED:-42}" \
  --class-weighting balanced

rm -rf "$CANDIDATE_ONNX"
$PYTHON_BIN scripts/export_router_classifier_onnx.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --onnx-dir "$CANDIDATE_ONNX"

EVAL_REPORT="$REPORT_DIR/classifier-eval-$(date -u +%F).json"
set +e
$PYTHON_BIN scripts/eval_router_classifier.py \
  --eval-file "$EVAL_FILE" \
  --onnx-dir "$CANDIDATE_ONNX" \
  --baseline-onnx-dir "$LIVE_ONNX" \
  --min-macro-f1 "${RETRAIN_MIN_MACRO_F1:-0.0}" \
  --min-macro-f1-delta "${RETRAIN_MIN_MACRO_F1_DELTA:-0.0}" \
  --report-out "$EVAL_REPORT"
EVAL_RC=$?
set -e

if [[ "$EVAL_RC" == 0 ]]; then
  if [[ -d "$LIVE_ONNX" ]]; then
    rm -rf "$PREV_ONNX"
    mv "$LIVE_ONNX" "$PREV_ONNX"
  fi
  mv "$CANDIDATE_ONNX" "$LIVE_ONNX"
  echo "[nightly-retrain] promoted candidate ONNX -> $LIVE_ONNX (previous saved at $PREV_ONNX)"

  if command -v docker >/dev/null 2>&1 && docker inspect nexus-router-1 >/dev/null 2>&1; then
    docker restart nexus-router-1
    READY=0
    for _ in $(seq 1 30); do
      if curl -sf http://127.0.0.1:7771/health >/dev/null 2>&1; then
        READY=1
        break
      fi
      sleep 1
    done
    if [[ "$READY" != 1 ]]; then
      echo "[nightly-retrain] ERROR: nexus-router-1 failed health check after restart" >&2
      exit 1
    fi
    echo "[nightly-retrain] nexus-router-1 restarted and healthy"
  fi

  echo "$(date -u +%FT%TZ)" > "$STAMP_DIR/last_success_utc.txt"
  echo "$TOTAL_VERIFIED" > "$STAMP_DIR/last_verified_count.txt"
elif [[ "$EVAL_RC" == 2 ]]; then
  echo "[nightly-retrain] promotion blocked by eval gate; live model unchanged"
  rm -rf "$CANDIDATE_ONNX"
else
  echo "[nightly-retrain] ERROR: eval_router_classifier exited $EVAL_RC" >&2
  rm -rf "$CANDIDATE_ONNX"
  exit "$EVAL_RC"
fi

echo "[nightly-retrain] done $(date -u +%FT%TZ)"
