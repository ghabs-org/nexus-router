#!/usr/bin/env python3
"""
scripts/export_classifier_training_data.py

Export labelled routing decisions from router.sqlite for classifier training.

Usage:
    python scripts/export_classifier_training_data.py [--output training_data.jsonl] [--min-samples N]

Output: JSONL file, one record per line:
    {"text": "...", "label": "coding", "classifier_source": "llm", "confidence": 0.82}

Only includes records where:
  - message_text is present (non-null, non-empty)
  - classifier_source is "llm" or "explicit" (trusted labels)
  - task_type is a known label
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

try:
    from src.db import _connect
    from src.local_classifier_labels import ROUTER_TASK_LABELS
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    from db import _connect  # type: ignore
    from local_classifier_labels import ROUTER_TASK_LABELS  # type: ignore

TRUSTED_SOURCES = {"llm", "explicit"}
VALID_LABELS = set(ROUTER_TASK_LABELS)


def export(output_path: str, min_samples: int = 1) -> None:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT message_text, task_type, classifier_source,
                   classifier_confidence, has_code, has_image, has_diff,
                   has_logs, estimated_tokens
            FROM routing_decisions
            WHERE message_text IS NOT NULL
              AND TRIM(message_text) != ''
              AND classifier_source IN ('llm', 'explicit')
              AND task_type IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()

    label_counts: Counter = Counter()
    written = 0

    with open(output_path, "w") as f:
        for row in rows:
            label = row["task_type"]
            if label not in VALID_LABELS:
                continue
            record = {
                "text": row["message_text"].strip(),
                "label": label,
                "classifier_source": row["classifier_source"],
                "confidence": float(row["classifier_confidence"] or 0.0),
                "signals": {
                    "has_code": bool(row["has_code"]),
                    "has_image": bool(row["has_image"]),
                    "has_diff": bool(row["has_diff"]),
                    "has_logs": bool(row["has_logs"]),
                    "estimated_tokens": int(row["estimated_tokens"] or 0),
                },
            }
            f.write(json.dumps(record) + "\n")
            label_counts[label] += 1
            written += 1

    print(f"Exported {written} records to {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // 5)
        print(f"  {label:20s} {count:5d}  {bar}")

    rare = [l for l, c in label_counts.items() if c < min_samples]
    if rare:
        print(f"\n⚠ Labels with < {min_samples} samples (may need augmentation): {rare}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classifier training data from router.sqlite")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL path")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Warn if any label has fewer than this many samples")
    args = parser.parse_args()
    export(args.output, args.min_samples)


if __name__ == "__main__":
    main()
