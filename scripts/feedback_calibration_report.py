#!/usr/bin/env python3
"""Generate a compact calibration report from routing feedback.

Read-only utility for nightly checks:
- feedback volume
- wrong-rate by task
- model preference signal by task/model
- optional per-source breakdown

Does NOT mutate router state.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".local/state/nexus-router/data/router.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration report from routing feedback")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to router.sqlite")
    parser.add_argument(
        "--source-type",
        default="raw-user",
        help="Optional filter to routing_decisions.source_type (default: raw-user)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional filter to routing_decisions.mode (for example: route or shadow)",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max joined feedback rows to inspect")
    return parser.parse_args()


def _new_task_stats() -> defaultdict[str, dict[str, int]]:
    return defaultdict(lambda: {"total": 0, "wrong": 0, "correct": 0, "unlabelled": 0})


def _new_model_task() -> defaultdict[str, dict[str, object]]:
    return defaultdict(lambda: {"samples": 0, "score_sum": 0.0, "last_feedback_at": ""})


def _ingest_row(r: sqlite3.Row, task_stats, model_task) -> None:
    task = r["task_type"] or "unknown"
    model = r["model_id"] or "unknown"
    verdict = (r["verdict"] or "").strip().lower()
    model_verdict = (r["model_verdict"] or "").strip().lower()

    task_stats[task]["total"] += 1
    if verdict == "wrong":
        task_stats[task]["wrong"] += 1
    elif verdict == "correct":
        task_stats[task]["correct"] += 1

    # Feedback rows without explicit model_verdict are treated as unlabelled for
    # model-task signal purposes, matching the previous report semantics.
    is_unlabelled = not model_verdict
    if is_unlabelled:
        task_stats[task]["unlabelled"] += 1

    if model_verdict == "good":
        score = 1.0
    elif model_verdict == "neutral":
        score = 0.4
    elif model_verdict in {"bad", "too_cheap", "too_powerful"}:
        score = 0.0
    else:
        score = 0.75 if verdict == "correct" else 0.0

    if not is_unlabelled:
        key = f"{task}::{model}"
        model_task[key]["samples"] += 1
        model_task[key]["score_sum"] += score
        created_at = str(r["created_at"] or "")
        if created_at and created_at > str(model_task[key].get("last_feedback_at") or ""):
            model_task[key]["last_feedback_at"] = created_at


def _finalize_summary(task_stats, model_task, *, feedback_total: int) -> dict:
    by_task = []
    total_unlabelled = 0
    for task, s in sorted(task_stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
        total = max(1, s["total"])
        total_unlabelled += s["unlabelled"]
        by_task.append(
            {
                "task": task,
                "samples": s["total"],
                "wrong_rate": round(s["wrong"] / total, 4),
                "correct_rate": round(s["correct"] / total, 4),
                "unlabelled": s["unlabelled"],
            }
        )

    by_model_task = []
    for key, s in sorted(model_task.items(), key=lambda kv: kv[1]["samples"], reverse=True):
        task, model = key.split("::", 1)
        samples = max(1, s["samples"])
        mean_score = s["score_sum"] / samples
        centered = (mean_score - 0.5) * 2.0
        by_model_task.append(
            {
                "task": task,
                "model": model,
                "samples": s["samples"],
                "mean_score": round(mean_score, 4),
                "centered_signal": round(centered, 4),
                "last_feedback_at": s.get("last_feedback_at") or None,
            }
        )

    return {
        "feedback_total": int(feedback_total or 0),
        "unlabelled_feedback_count": total_unlabelled,
        "task_summary": by_task,
        "model_task_signals_top": by_model_task[:50],
    }


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(json.dumps({"ok": False, "error": f"db_not_found:{db_path}"}))
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    overall_task_stats = _new_task_stats()
    overall_model_task = _new_model_task()
    source_buckets: dict[str, dict[str, object]] = {}

    decision_columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
    mode_expr = (
        "COALESCE(NULLIF(rd.mode, ''), NULLIF(rd.provenance_mode, ''), CASE WHEN COALESCE(rd.shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "mode" in decision_columns and "provenance_mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(rd.mode, ''), NULLIF(rd.provenance_mode, ''), 'route')"
        if "mode" in decision_columns and "provenance_mode" in decision_columns
        else "COALESCE(NULLIF(rd.mode, ''), CASE WHEN COALESCE(rd.shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(rd.mode, ''), 'route')"
        if "mode" in decision_columns
        else "COALESCE(NULLIF(rd.provenance_mode, ''), CASE WHEN COALESCE(rd.shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "provenance_mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(rd.provenance_mode, ''), 'route')"
        if "provenance_mode" in decision_columns
        else "CASE WHEN COALESCE(rd.shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END"
    )

    filters = []
    params: list[object] = []
    if args.source_type:
        filters.append("COALESCE(NULLIF(rd.source_type, ''), 'standalone') = ?")
        params.append(args.source_type)
    if args.mode:
        filters.append(f"{mode_expr} = ?")
        params.append(args.mode)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    total_feedback = conn.execute(
        f"""
        SELECT COUNT(*) AS c
        FROM route_feedback rf
        JOIN routing_decisions rd ON rd.id = rf.decision_id
        {where_sql}
        """,
        params,
    ).fetchone()["c"]

    rows = conn.execute(
        f"""
        SELECT
          COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type, 'unknown') AS task_type,
          COALESCE(rf.preferred_model, rd.selected_model, 'unknown') AS model_id,
          rf.verdict,
          rf.model_verdict,
          rf.created_at,
          {mode_expr} AS mode,
          COALESCE(NULLIF(rd.source_type, ''), 'standalone') AS source_type
        FROM route_feedback rf
        JOIN routing_decisions rd ON rd.id = rf.decision_id
        {where_sql}
        ORDER BY rf.created_at DESC
        LIMIT ?
        """,
        [*params, args.limit],
    ).fetchall()

    for r in rows:
        _ingest_row(r, overall_task_stats, overall_model_task)

        source_type = str(r["source_type"] or "standalone")
        bucket = source_buckets.setdefault(
            source_type,
            {
                "task_stats": _new_task_stats(),
                "model_task": _new_model_task(),
                "feedback_total": 0,
                "latest_feedback_at": None,
            },
        )
        bucket["feedback_total"] += 1
        created_at = str(r["created_at"] or "") or None
        if created_at and (not bucket["latest_feedback_at"] or created_at > bucket["latest_feedback_at"]):
            bucket["latest_feedback_at"] = created_at
        _ingest_row(r, bucket["task_stats"], bucket["model_task"])

    overall = _finalize_summary(overall_task_stats, overall_model_task, feedback_total=int(total_feedback or 0))

    source_summary = []
    by_source = {}
    for source_type, bucket in sorted(source_buckets.items(), key=lambda kv: kv[1]["feedback_total"], reverse=True):
        finalized = _finalize_summary(
            bucket["task_stats"],
            bucket["model_task"],
            feedback_total=int(bucket["feedback_total"] or 0),
        )
        finalized["latest_feedback_at"] = bucket["latest_feedback_at"]
        by_source[source_type] = finalized
        source_summary.append(
            {
                "source_type": source_type,
                "feedback_total": finalized["feedback_total"],
                "latest_feedback_at": bucket["latest_feedback_at"],
            }
        )

    report = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path.resolve()),
        "filters": {
            "source_type": args.source_type,
            "mode": args.mode,
            "limit": args.limit,
        },
        "feedback_samples": len(rows),
        "feedback_total": overall["feedback_total"],
        "orphan_feedback_excluded": int((total_feedback or 0) - len(rows)),
        "unlabelled_feedback_count": overall["unlabelled_feedback_count"],
        "task_summary": overall["task_summary"],
        "model_task_signals_top": overall["model_task_signals_top"],
        "source_summary": source_summary,
        "by_source": by_source,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
