#!/usr/bin/env python3
"""Fine-tune ModernBERT for Nexus Router top-level task classification.

Expected dataset format: JSONL files with records like:
{"text": "please fix this failing pytest", "label": "coding"}

This script is intentionally v1/simple: it gives us a repeatable training path
without wiring the local classifier into production routing yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import inspect
from collections import Counter
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_classifier_labels import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MAX_SEQUENCE_LENGTH,
    ROUTER_TASK_LABELS,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            label = item.get("label")
            if not text:
                raise ValueError(f"{path}: line {idx} missing text")
            if label not in LABEL_TO_ID:
                raise ValueError(
                    f"{path}: line {idx} label={label!r} not in {sorted(ROUTER_TASK_LABELS)}"
                )
            rows.append({"text": text, "label": LABEL_TO_ID[label]})
    if not rows:
        raise ValueError(f"{path}: no usable rows found")
    return rows


def tokenize_dataset(dataset: Dataset, tokenizer):
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        ),
        batched=True,
    )
    if "text" in tokenized.column_names:
        tokenized = tokenized.remove_columns(["text"])
    return tokenized


def compute_class_weights(rows: list[dict], *, smoothing: float = 1.0) -> list[float]:
    counts = Counter(int(row["label"]) for row in rows)
    n_labels = len(ROUTER_TASK_LABELS)
    total = float(sum(counts.values()))
    weights: list[float] = []
    for label_idx in range(n_labels):
        c = float(counts.get(label_idx, 0.0))
        # Inverse-frequency weighting with smoothing to avoid exploding values.
        weight = total / max(1.0, (c + smoothing) * n_labels)
        weights.append(weight)
    # Normalize around 1.0 so learning rate remains intuitive.
    mean = sum(weights) / max(1, len(weights))
    return [w / mean for w in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=f"{DEFAULT_DATA_DIR}/train.jsonl")
    parser.add_argument("--eval-file", default=f"{DEFAULT_DATA_DIR}/eval.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--class-weighting",
        choices=["off", "balanced"],
        default="balanced",
        help="Class weighting strategy for rare-label recall (default: balanced)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))

    train_rows = load_jsonl(Path(args.train_file))
    eval_rows = load_jsonl(Path(args.eval_file))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        num_labels=len(ROUTER_TASK_LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    train_ds = tokenize_dataset(Dataset.from_list(train_rows), tokenizer)
    eval_ds = tokenize_dataset(Dataset.from_list(eval_rows), tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = compute_class_weights(train_rows)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "train_rows": len(train_rows),
                    "eval_rows": len(eval_rows),
                    "model_name": args.model_name,
                    "num_labels": len(ROUTER_TASK_LABELS),
                    "output_dir": args.output_dir,
                    "class_weighting": args.class_weighting,
                    "class_weights": class_weights,
                },
                indent=2,
            )
        )
        return 0

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )
    training_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_params:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    if class_weights is not None:
        import torch

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                weight_tensor = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)
                loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
    # attach tokenizer for compatibility with some Trainer utilities
    trainer.tokenizer = tokenizer
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({"status": "trained", "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
