#!/usr/bin/env python3
"""
scripts/export_classifier_training_data.py

Export labelled routing decisions from router.sqlite for classifier training.

Usage:
    python scripts/export_classifier_training_data.py [--output training_data.jsonl] [--min-samples N]

Output: JSONL file, one record per line:
    {"text": "...", "label": "coding", "classifier_source": "local", "confidence": 0.82}

Training labels are resolved as:
  - `route_feedback.corrected_task` when present and non-empty
  - otherwise `routing_decisions.task_type`

Only includes records where:
  - message_text is present (non-null, non-empty)
  - classifier_source is "local"
  - outcome_success is not NULL (turn completed, success or failure)
  - resolved label is a known label
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

try:
    from src.db import _connect
    from src.local_classifier_labels import ROUTER_TASK_LABELS
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    from db import _connect  # type: ignore
    from local_classifier_labels import ROUTER_TASK_LABELS  # type: ignore

VALID_LABELS = set(ROUTER_TASK_LABELS)


def _is_low_information_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return True
    if len(normalized) <= 12:
        return True
    trivial = {
        "ok", "okay", "ok thanks", "thanks", "thank you", "yes", "yes please",
        "done", "great", "hello", "hello?", "hi", "sure",
    }
    return normalized in trivial


def export(output_path: str, min_samples: int = 1) -> dict[str, object]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT
                   rd.message_text,
                   COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type) AS task_type,
                   rd.classifier_source,
                   rd.classifier_confidence,
                   rd.has_code,
                   rd.has_image,
                   rd.has_diff,
                   rd.has_logs,
                   rd.estimated_tokens
            FROM routing_decisions rd
            LEFT JOIN (
                SELECT decision_id, corrected_task, MAX(created_at) AS created_at
                FROM route_feedback
                WHERE corrected_task IS NOT NULL AND TRIM(corrected_task) != ''
                GROUP BY decision_id
            ) latest_rf ON latest_rf.decision_id = rd.id
            LEFT JOIN route_feedback rf
              ON rf.decision_id = latest_rf.decision_id
             AND rf.created_at = latest_rf.created_at
            WHERE rd.message_text IS NOT NULL
              AND TRIM(rd.message_text) != ''
              AND rd.classifier_source = 'local'
              AND rd.outcome_success IS NOT NULL
              AND COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type) IS NOT NULL
            ORDER BY rd.created_at DESC
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
            text = row["message_text"].strip()
            if label == "reasoning" and _is_low_information_text(text):
                continue
            record = {
                "text": text,
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
            repeat = 2 if label == "reasoning" else 1
            for _ in range(repeat):
                f.write(json.dumps(record) + "\n")
                label_counts[label] += 1
                written += 1

    summary = {
        "output_path": str(Path(output_path).resolve()),
        "rows_scanned": len(rows),
        "records_written": written,
        "label_distribution": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "rare_labels": [],
    }

    print(f"Exported {written} records to {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // 5)
        print(f"  {label:20s} {count:5d}  {bar}")

    rare = [lbl for lbl, c in label_counts.items() if c < min_samples]
    summary["rare_labels"] = rare
    if rare:
        print(f"\n⚠ Labels with < {min_samples} samples (may need augmentation): {rare}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classifier training data from router.sqlite")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL path")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Warn if any label has fewer than this many samples")
    args = parser.parse_args()
    summary = export(args.output, args.min_samples)
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
