#!/usr/bin/env python3
"""Generate a compact calibration report from routing feedback.

Read-only utility for nightly checks:
- feedback volume
- wrong-rate by task
- model preference signal by task/model

Does NOT mutate router state.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path.home() / ".local/state/nexus-router/data/router.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration report from routing feedback")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to router.sqlite")
    parser.add_argument(
        "--source-type",
        default="raw-user",
        help="Filter to routing_decisions.source_type (default: raw-user)",
    )
    parser.add_argument(
        "--provenance-mode",
        default=None,
        help="Optional filter to routing_decisions.provenance_mode (for example: route or shadow)",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max joined feedback rows to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(json.dumps({"ok": False, "error": f"db_not_found:{db_path}"}))
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    task_stats = defaultdict(lambda: {"total": 0, "wrong": 0, "correct": 0})
    model_task = defaultdict(lambda: {"samples": 0, "score_sum": 0.0, "last_feedback_at": ""})

    decision_columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
    provenance_expr = (
        "COALESCE(rd.provenance_mode, CASE WHEN COALESCE(rd.shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "shadow_mode" in decision_columns
        else "COALESCE(rd.provenance_mode, 'route')"
    )

    filters = []
    params: list[object] = []
    if args.source_type:
        filters.append("COALESCE(rd.source_type, 'unknown') = ?")
        params.append(args.source_type)
    if args.provenance_mode:
        filters.append(f"{provenance_expr} = ?")
        params.append(args.provenance_mode)

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
          {provenance_expr} AS provenance_mode,
          COALESCE(rd.source_type, 'unknown') AS source_type
        FROM route_feedback rf
        JOIN routing_decisions rd ON rd.id = rf.decision_id
        {where_sql}
        ORDER BY rf.created_at DESC
        LIMIT ?
        """,
        [*params, args.limit],
    ).fetchall()

    for r in rows:
        task = r["task_type"] or "unknown"
        model = r["model_id"] or "unknown"
        verdict = (r["verdict"] or "").strip().lower()
        model_verdict = (r["model_verdict"] or "").strip().lower()

        task_stats[task]["total"] += 1
        if verdict == "wrong":
            task_stats[task]["wrong"] += 1
        elif verdict == "correct":
            task_stats[task]["correct"] += 1

        # Keep scoring aligned with db.py aggregation semantics.
        if model_verdict == "good":
            score = 1.0
        elif model_verdict == "neutral":
            score = 0.4
        elif model_verdict == "bad":
            score = 0.0
        else:
            score = 0.75 if verdict == "correct" else 0.0

        key = f"{task}::{model}"
        model_task[key]["samples"] += 1
        model_task[key]["score_sum"] += score
        created_at = str(r["created_at"] or "")
        if created_at and created_at > str(model_task[key].get("last_feedback_at") or ""):
            model_task[key]["last_feedback_at"] = created_at

    by_task = []
    for task, s in sorted(task_stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
        total = max(1, s["total"])
        by_task.append(
            {
                "task": task,
                "samples": s["total"],
                "wrong_rate": round(s["wrong"] / total, 4),
                "correct_rate": round(s["correct"] / total, 4),
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

    report = {
        "ok": True,
        "db": str(db_path),
        "filters": {
            "source_type": args.source_type,
            "provenance_mode": args.provenance_mode,
            "limit": args.limit,
        },
        "feedback_samples": len(rows),
        "feedback_total": int(total_feedback or 0),
        "orphan_feedback_excluded": int((total_feedback or 0) - len(rows)),
        "task_summary": by_task,
        "model_task_signals_top": by_model_task[:50],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
