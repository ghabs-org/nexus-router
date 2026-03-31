# Local classifier foundation (ModernBERT)

This repo now has a **build/export scaffold** for a local router classifier based
on `answerdotai/ModernBERT-base`.

## Env split

We are intentionally keeping two different Python environments:

- **Runtime / production base:** `/home/ubuntu/.openclaw/workspace/.venvs/memsearch`
  - good place for `onnxruntime` inference once a classifier artifact exists
  - should stay lean and focused on serving / routing
- **Build / export env:** `/home/ubuntu/.openclaw/workspace/.venvs/router-classifier-build`
  - used for training, Hugging Face model loading, and ONNX export
  - allowed to carry heavier packages like `torch`, `transformers`, and `optimum`

This keeps training/export concerns out of the production runtime.

## Shared HF cache

Both environments should reuse the workspace cache:

- `HF_HOME=/home/ubuntu/.openclaw/workspace/.cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/home/ubuntu/.openclaw/workspace/.cache/huggingface/hub`

The build scripts default to those paths.

## Labels

The v1 local classifier mirrors the router's existing top-level task labels:

- `coding`
- `code_review`
- `reasoning`
- `summarization`
- `fast_utility`
- `long_context`
- `vision`
- `general_chat`

Those constants live in `src/local_classifier_labels.py`.

## Training path

Prepare JSONL files under `artifacts/router-classifier/data/`:

- `train.jsonl`
- `eval.jsonl`

Then run:

```bash
source /home/ubuntu/.openclaw/workspace/.venvs/router-classifier-build/bin/activate
python scripts/train_router_classifier.py --dry-run
python scripts/train_router_classifier.py
```

The dry run validates labels, dataset shape, tokenizer/model loading, and output paths
without starting a training job.

## Export path

After training, export to ONNX with:

```bash
source /home/ubuntu/.openclaw/workspace/.venvs/router-classifier-build/bin/activate
python scripts/export_router_classifier_onnx.py \
  --checkpoint-dir artifacts/router-classifier/checkpoints \
  --onnx-dir artifacts/router-classifier/onnx
```

Expected production usage later:

1. build env fine-tunes ModernBERT and exports ONNX
2. ONNX artifact is copied/versioned as a router asset
3. runtime env (`memsearch`) loads the ONNX model with `onnxruntime`
4. router server now prefers the local classifier as the primary first-pass classifier when a valid ONNX artifact is present
5. existing heuristic + LLM classification paths remain as safe fallback layers for unavailable / low-confidence / ambiguous local cases

## Runtime behavior

The live routing stack now uses the local ONNX classifier first when the artifact is present and passes the runtime confidence gates.

Default runtime gates:

- minimum confidence: `0.68`
- minimum top-1 vs top-2 margin: `0.12`

If those gates are not met, the router falls through to heuristic and then LLM classification.

Relevant env vars:

- `NEXUS_ROUTER_LOCAL_CLASSIFIER_DIR`
- `NEXUS_ROUTER_LOCAL_CLASSIFIER_MODEL_FILE`
- `NEXUS_ROUTER_LOCAL_CLASSIFIER_MIN_CONFIDENCE`
- `NEXUS_ROUTER_LOCAL_CLASSIFIER_MARGIN`
- `NEXUS_ROUTER_LOCAL_CLASSIFIER_MAX_LENGTH`
