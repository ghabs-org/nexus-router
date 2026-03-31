#!/usr/bin/env python3
"""Export a trained router classifier checkpoint to ONNX.

Input should be a fine-tuned sequence-classification checkpoint created by
train_router_classifier.py (or an equivalent Hugging Face checkpoint directory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_classifier_labels import (  # noqa: E402
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_ONNX_DIR,
    DEFAULT_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))

    model = ORTModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, cache_dir=args.cache_dir)
    model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)

    produced = sorted(p.name for p in onnx_dir.iterdir())
    print(
        json.dumps(
            {
                "status": "exported",
                "checkpoint_dir": str(checkpoint_dir),
                "onnx_dir": str(onnx_dir),
                "files": produced,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
