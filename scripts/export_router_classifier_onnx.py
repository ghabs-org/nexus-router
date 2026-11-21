#!/usr/bin/env python3
"""Export a trained router classifier checkpoint to ONNX and verify parity.

Uses `optimum.onnxruntime` when installed and compatible; otherwise exports
directly with `torch.onnx.export`. Verifies input/output shapes and numerical
parity against PyTorch using `onnxruntime`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
except ImportError:
    ORTModelForSequenceClassification = None  # type: ignore[assignment]

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_classifier_labels import (  # noqa: E402
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_ONNX_DIR,
    DEFAULT_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser.parse_args()


def _load_tokenizer(checkpoint_dir: Path, cache_dir: str):
    try:
        return AutoTokenizer.from_pretrained(checkpoint_dir, cache_dir=cache_dir)
    except Exception:
        return AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, cache_dir=cache_dir)


def _export_with_torch(checkpoint_dir: Path, onnx_dir: Path, tokenizer) -> str:
    import torch

    class _OnnxLogitsWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval().cpu()
    wrapper = _OnnxLogitsWrapper(model).eval()

    dummy = tokenizer("Compare these two architectures for failover risk.", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    onnx_file = onnx_dir / "model.onnx"

    export_kwargs = dict(
        args=(input_ids, attention_mask),
        f=str(onnx_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    try:
        with torch.no_grad():
            torch.onnx.export(wrapper, **export_kwargs, dynamo=False)
        backend = "torch.onnx.export(dynamo=False)"
    except Exception:
        with torch.no_grad():
            torch.onnx.export(wrapper, **export_kwargs, dynamo=True)
        backend = "torch.onnx.export(dynamo=True)"

    # Keep model.onnx self-contained if external data files were generated.
    external_data = Path(f"{onnx_file}.data")
    if external_data.exists():
        import onnx

        proto = onnx.load(str(onnx_file), load_external_data=True)
        onnx.save_model(proto, str(onnx_file), save_as_external_data=False)
        external_data.unlink(missing_ok=True)

    model.config.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)
    return backend


def _verify_onnx_parity(checkpoint_dir: Path, onnx_dir: Path, tokenizer) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch

    onnx_file = onnx_dir / "model.onnx"
    if not onnx_file.exists():
        raise FileNotFoundError(f"missing exported ONNX model: {onnx_file}")

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    if not {"input_ids", "attention_mask"}.issubset(input_names):
        raise RuntimeError(f"unexpected ONNX inputs: {sorted(input_names)}")

    sample = tokenizer("Compare these two architectures for latency and failover risk.", return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval().cpu()
    with torch.no_grad():
        pt_logits = model(input_ids=sample["input_ids"], attention_mask=sample["attention_mask"]).logits.numpy()

    ort_logits = session.run(
        None,
        {
            "input_ids": sample["input_ids"].numpy().astype(np.int64),
            "attention_mask": sample["attention_mask"].numpy().astype(np.int64),
        },
    )[0]
    if ort_logits.shape != pt_logits.shape:
        raise RuntimeError(f"ONNX output shape mismatch: {ort_logits.shape} != {pt_logits.shape}")

    max_abs_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    if not np.allclose(pt_logits, ort_logits, atol=1e-3, rtol=1e-3):
        raise RuntimeError(f"ONNX parity check failed: max_abs_diff={max_abs_diff:.6f}")
    return max_abs_diff


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))

    tokenizer = _load_tokenizer(checkpoint_dir, args.cache_dir)
    backend = None
    if ORTModelForSequenceClassification is not None:
        try:
            model = ORTModelForSequenceClassification.from_pretrained(checkpoint_dir, export=True)
            model.save_pretrained(onnx_dir)
            tokenizer.save_pretrained(onnx_dir)
            backend = "optimum.onnxruntime"
        except Exception as exc:
            print(f"[export_onnx] optimum backend failed ({exc}); falling back to torch.onnx.export", file=sys.stderr)

    if backend is None:
        backend = _export_with_torch(checkpoint_dir, onnx_dir, tokenizer)

    max_abs_diff = _verify_onnx_parity(checkpoint_dir, onnx_dir, tokenizer)
    produced = sorted(p.name for p in onnx_dir.iterdir())
    print(
        json.dumps(
            {
                "status": "exported",
                "backend": backend,
                "max_abs_diff": round(max_abs_diff, 6),
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
