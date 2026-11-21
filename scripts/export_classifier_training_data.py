#!/usr/bin/env python3
"""
scripts/export_classifier_training_data.py

Export labelled routing decisions from router.sqlite for classifier training.

Usage:
    python scripts/export_classifier_training_data.py [--output training_data.jsonl] [--min-samples N]

Training labels are resolved as:
  - `route_feedback.corrected_task` (latest feedback per decision) when non-empty
  - otherwise `routing_decisions.task_type`

Inclusion rules:
  - Always include rows with non-empty `message_text` where the latest feedback
    has a non-empty `corrected_task` OR `verdict == 'correct'`, regardless of
    `classifier_source` or `outcome_success`.
  - Always exclude rows where latest feedback has `verdict == 'wrong'` and no
    `corrected_task`.
  - When `verified_only=False`, rows without feedback are included only when
    `classifier_source == 'local'` and `outcome_success IS NOT NULL`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).parent.parent

try:
    from src.db import _connect
    from src.local_classifier_labels import ROUTER_TASK_LABELS
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    from db import _connect  # type: ignore
    from local_classifier_labels import ROUTER_TASK_LABELS  # type: ignore

VALID_LABELS = set(ROUTER_TASK_LABELS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_low_information_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if len(normalized) <= 12:
        return True
    trivial = {
        "ok", "okay", "ok thanks", "thanks", "thank you", "yes", "yes please",
        "done", "great", "hello", "hello?", "hi", "sure",
    }
    return normalized in trivial


def split_stratified(
    rows: list[dict[str, object]],
    *,
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Deduplicate by normalized text and perform a stratified train/eval split.

    Labels with fewer than 2 samples stay in train so rare classes never appear
    exclusively in eval. Guarantees zero normalized-text overlap between splits.
    """
    seen: set[str] = set()
    unique_rows: list[dict[str, object]] = []
    for row in rows:
        norm = _normalize_text(str(row.get("text", "")))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique_rows.append(row)

    by_label: dict[str, list[dict[str, object]]] = {}
    for row in unique_rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    ratio = max(0.0, min(0.5, float(eval_ratio)))

    for label in sorted(by_label):
        group = list(by_label[label])
        random.Random(f"{seed}:{label}").shuffle(group)
        if len(group) < 2 or ratio <= 0.0:
            train_rows.extend(group)
            continue
        n_eval = max(1, min(len(group) - 1, int(round(len(group) * ratio))))
        eval_rows.extend(group[:n_eval])
        train_rows.extend(group[n_eval:])

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(eval_rows)

    train_texts = {_normalize_text(str(r["text"])) for r in train_rows}
    eval_texts = {_normalize_text(str(r["text"])) for r in eval_rows}
    overlap = train_texts & eval_texts
    if overlap:
        raise RuntimeError(f"train/eval split leakage detected for {len(overlap)} prompt(s)")

    return train_rows, eval_rows


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def export(
    output_path: str,
    min_samples: int = 1,
    *,
    verified_only: bool = False,
    split_dir: str | Path | None = None,
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    """Export labelled decisions to JSONL and optionally write stratified train/eval splits."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            WITH latest_rf AS (
                SELECT
                    decision_id,
                    verdict,
                    corrected_task,
                    ROW_NUMBER() OVER (
                        PARTITION BY decision_id
                        ORDER BY created_at DESC, id DESC
                    ) AS rn
                FROM route_feedback
            )
            SELECT
                   rd.id AS decision_id,
                   rd.created_at,
                   rd.message_text,
                   COALESCE(NULLIF(TRIM(rf.corrected_task), ''), rd.task_type) AS task_type,
                   rf.verdict AS feedback_verdict,
                   rf.corrected_task AS corrected_task,
                   rd.classifier_source,
                   rd.classifier_confidence,
                   rd.has_code,
                   rd.has_image,
                   rd.has_diff,
                   rd.has_logs,
                   rd.estimated_tokens
            FROM routing_decisions rd
            LEFT JOIN latest_rf rf
              ON rf.decision_id = rd.id
             AND rf.rn = 1
            WHERE rd.message_text IS NOT NULL
              AND TRIM(rd.message_text) != ''
              AND (
                    (rf.corrected_task IS NOT NULL AND TRIM(rf.corrected_task) != '')
                 OR LOWER(TRIM(COALESCE(rf.verdict, ''))) = 'correct'
                 OR (
                        rf.decision_id IS NULL
                    AND rd.classifier_source = 'local'
                    AND rd.outcome_success IS NOT NULL
                 )
              )
              AND COALESCE(NULLIF(TRIM(rf.corrected_task), ''), rd.task_type) IS NOT NULL
            ORDER BY rd.created_at DESC, rd.id DESC
            """
        ).fetchall()
    finally:
        conn.close()

    candidates: list[tuple[bool, dict[str, object]]] = []
    for row in rows:
        label = str(row["task_type"]).strip()
        if label not in VALID_LABELS:
            continue
        text = row["message_text"].strip()
        if label == "reasoning" and _is_low_information_text(text):
            continue
        verdict = str(row["feedback_verdict"] or "").strip().lower()
        corrected = str(row["corrected_task"] or "").strip()
        is_verified = bool(corrected) or verdict == "correct"
        if verified_only and not is_verified:
            continue
        record: dict[str, object] = {
            "decision_id": row["decision_id"],
            "text": text,
            "label": label,
            "verified": is_verified,
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
        candidates.append((is_verified, record))

    # Prefer human-verified labels over unverified decisions if identical text appears again.
    candidates.sort(key=lambda item: 0 if item[0] else 1)

    seen_texts: set[str] = set()
    records: list[dict[str, object]] = []
    deduped_dropped = 0
    for _is_verified, record in candidates:
        norm = _normalize_text(str(record["text"]))
        if norm in seen_texts:
            deduped_dropped += 1
            continue
        seen_texts.add(norm)
        records.append(record)

    label_counts: Counter = Counter(str(r["label"]) for r in records)
    verified_count = sum(1 for r in records if r["verified"])
    out_file = Path(output_path)
    _write_jsonl(out_file, records)

    split_info: dict[str, object] | None = None
    if split_dir is not None:
        split_root = Path(split_dir)
        train_rows, eval_rows = split_stratified(records, eval_ratio=eval_ratio, seed=seed)
        train_path = split_root / "train.jsonl"
        eval_path = split_root / "eval.jsonl"
        summary_path = split_root / "split_summary.json"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(eval_path, eval_rows)
        split_info = {
            "total": len(records),
            "total_verified": verified_count,
            "train": len(train_rows),
            "eval": len(eval_rows),
            "eval_ratio": eval_ratio,
            "seed": seed,
            "export_path": str(out_file.resolve()),
            "train_file": str(train_path.resolve()),
            "eval_file": str(eval_path.resolve()),
            "train_label_distribution": dict(sorted(Counter(str(r["label"]) for r in train_rows).items())),
            "eval_label_distribution": dict(sorted(Counter(str(r["label"]) for r in eval_rows).items())),
        }
        summary_path.write_text(json.dumps(split_info, indent=2) + "\n")

    rare = sorted(lbl for lbl, c in label_counts.items() if c < min_samples)
    summary: dict[str, object] = {
        "output_path": str(out_file.resolve()),
        "rows_scanned": len(rows),
        "records_written": len(records),
        "verified_records": verified_count,
        "unverified_records": len(records) - verified_count,
        "deduped_dropped": deduped_dropped,
        "label_distribution": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "rare_labels": rare,
    }
    if split_info is not None:
        summary["split"] = split_info

    print(
        f"Exported {len(records)} unique records to {output_path} "
        f"(verified={verified_count}, unverified={len(records) - verified_count}, deduped_dropped={deduped_dropped})"
    )
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // 5)
        print(f"  {label:20s} {count:5d}  {bar}")

    if rare:
        print(f"\n⚠ Labels with < {min_samples} samples (may need augmentation): {rare}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classifier training data from router.sqlite")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL path")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Warn if any label has fewer than this many samples")
    parser.add_argument("--verified-only", action="store_true",
                        help="Export only human-verified feedback rows (verdict='correct' or non-empty corrected_task)")
    parser.add_argument("--split-dir", default=None,
                        help="Optional directory in which to write stratified train.jsonl, eval.jsonl, and split_summary.json")
    parser.add_argument("--eval-ratio", type=float, default=0.15,
                        help="Evaluation split fraction per label (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    args = parser.parse_args()
    summary = export(
        args.output,
        args.min_samples,
        verified_only=bool(args.verified_only),
        split_dir=args.split_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
