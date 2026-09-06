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
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import tempfile

DB_PATH = Path.home() / ".local/state/nexus-router/data/router.sqlite"
ROOT = Path(__file__).resolve().parents[1]

try:
    from src.provider_freshness import evaluate_freshness_transitions
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    from provider_freshness import evaluate_freshness_transitions  # type: ignore


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
    by_task.sort(key=lambda item: (item["wrong_rate"], item["samples"]), reverse=True)

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


def _feedback_fingerprint(conn: sqlite3.Connection, where_sql: str, params: list[object]) -> dict[str, object]:
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS feedback_total,
          MAX(rf.created_at) AS latest_feedback_at,
          COUNT(DISTINCT COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type, 'unknown')) AS distinct_labels,
          COUNT(DISTINCT COALESCE(rf.preferred_model, rd.selected_model, 'unknown')) AS distinct_models,
          SUM(
            LENGTH(COALESCE(rf.verdict, '')) +
            LENGTH(COALESCE(rf.model_verdict, '')) +
            LENGTH(COALESCE(rf.corrected_task, '')) +
            LENGTH(COALESCE(rf.preferred_model, '')) +
            LENGTH(COALESCE(rf.reason_tag, ''))
          ) AS field_length_sum
        FROM route_feedback rf
        JOIN routing_decisions rd ON rd.id = rf.decision_id
        {where_sql}
        """,
        params,
    ).fetchone()
    raw = "|".join(
        [
            str(row["feedback_total"] or 0),
            str(row["latest_feedback_at"] or ""),
            str(row["distinct_labels"] or 0),
            str(row["distinct_models"] or 0),
            str(row["field_length_sum"] or 0),
        ]
    )
    return {
        "feedback_total": int(row["feedback_total"] or 0),
        "latest_feedback_at": row["latest_feedback_at"],
        "distinct_labels": int(row["distinct_labels"] or 0),
        "distinct_models": int(row["distinct_models"] or 0),
        "field_length_sum": int(row["field_length_sum"] or 0),
        "signature": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _load_previous_report(report_path: Path) -> dict[str, object] | None:
    try:
        if report_path.exists():
            return json.loads(report_path.read_text())
    except Exception:
        return None
    return None


def _should_skip_training(conn: sqlite3.Connection, where_sql: str, params: list[object]) -> tuple[bool, dict[str, object], dict[str, object] | None]:
    previous_path = Path.home() / ".local" / "state" / "nexus-router" / "reports" / f"calibration-{datetime.now(timezone.utc).date().isoformat()}.json"
    previous_report = _load_previous_report(previous_path)
    fingerprint = _feedback_fingerprint(conn, where_sql, params)
    if not previous_report:
        return False, fingerprint, None
    previous_feedback_total = int(previous_report.get("feedback_total") or 0)
    previous_latest_feedback_at = ((previous_report.get("feedback_fingerprint") or {}).get("latest_feedback_at") or previous_report.get("latest_feedback_at"))
    previous_signature = ((previous_report.get("feedback_fingerprint") or {}).get("signature"))
    skip = (
        previous_feedback_total == fingerprint["feedback_total"]
        and previous_latest_feedback_at == fingerprint["latest_feedback_at"]
        and previous_signature == fingerprint["signature"]
    )
    return skip, fingerprint, previous_report


def _training_export_summary(conn: sqlite3.Connection, where_sql: str, params: list[object]) -> dict[str, object] | None:
    export_script = ROOT / "scripts" / "export_classifier_training_data.py"
    if not export_script.exists():
        return None

    skip_training, fingerprint, previous_report = _should_skip_training(conn, where_sql, params)
    if skip_training:
        previous_training = (previous_report or {}).get("training_export") or {}
        return {
            "export_output_path": previous_training.get("export_output_path"),
            "records_written": int(previous_training.get("records_written") or previous_report.get("training_rows_used") or 0),
            "rows_scanned": int(previous_training.get("rows_scanned") or previous_report.get("training_rows_scanned") or 0),
            "label_distribution": previous_training.get("label_distribution") or {},
            "rare_labels": previous_training.get("rare_labels") or [],
            "command_ok": True,
            "stdout": "",
            "stderr": "",
            "skipped": True,
            "skip_reason": "feedback fingerprint unchanged",
            "feedback_fingerprint": fingerprint,
        }

    with tempfile.TemporaryDirectory(prefix="router-calibration-") as tmpdir:
        export_path = Path(tmpdir) / "training.jsonl"
        payload = {
            "export_output_path": str(export_path),
            "records_written": 0,
            "rows_scanned": 0,
            "label_distribution": {},
            "rare_labels": [],
            "skipped": False,
            "feedback_fingerprint": fingerprint,
        }
        import subprocess
        result = subprocess.run(
            [sys.executable, str(export_script), "--output", str(export_path), "--min-samples", "1"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        payload["command_ok"] = result.returncode == 0
        payload["stdout"] = result.stdout
        payload["stderr"] = result.stderr
        if result.returncode != 0:
            return payload
        marker = "Summary JSON:"
        if marker in result.stdout:
            try:
                summary_text = result.stdout.split(marker, 1)[1].strip()
                parsed = json.loads(summary_text)
                payload.update(parsed)
            except Exception:
                pass
        return payload


def _weekly_spend_by_task(conn: sqlite3.Connection, where_sql: str, params: list[object]) -> dict[str, object]:
    rows = conn.execute(
        f"""
        SELECT
          COALESCE(rd.task_type, 'unknown') AS task_type,
          COUNT(*) AS decisions,
          SUM(COALESCE(rd.estimated_cost_usd, 0.0)) AS estimated_spend_usd,
          SUM(COALESCE(rd.total_tokens, 0)) AS total_tokens
        FROM routing_decisions rd
        {where_sql}
          {'AND' if where_sql else 'WHERE'} rd.created_at >= datetime('now', '-7 days')
          AND rd.outcome_success IS NOT NULL
        GROUP BY COALESCE(rd.task_type, 'unknown')
        ORDER BY estimated_spend_usd DESC, decisions DESC
        """,
        params,
    ).fetchall()

    by_task = [
        {
            "task": row["task_type"],
            "decisions": int(row["decisions"] or 0),
            "estimated_spend_usd": round(float(row["estimated_spend_usd"] or 0.0), 6),
            "total_tokens": int(row["total_tokens"] or 0),
        }
        for row in rows
    ]
    return {
        "window": "7d",
        "by_task": by_task,
        "total_estimated_spend_usd": round(sum(item["estimated_spend_usd"] for item in by_task), 6),
        "pipeline": {
            "cost_enabled": True,
            "co2_enabled": False,
            "co2_note": "estimated_co2e_grams reserved in schema; provider factors not wired yet",
        },
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

    provider_rows = {
        row["provider"]: dict(row)
        for row in conn.execute("SELECT * FROM providers").fetchall()
    }
    health_rows = {
        row["provider"]: dict(row)
        for row in conn.execute("SELECT * FROM provider_health_state").fetchall()
    }
    provider_freshness = evaluate_freshness_transitions(
        provider_rows=provider_rows,
        health_rows=health_rows,
        stale_after_hours=24,
    )
    if provider_freshness["newly_stale"]:
        print(
            "[provider-freshness] healthy->stale transitions: "
            + ", ".join(provider_freshness["newly_stale"]),
            file=sys.stderr,
        )

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

    training_export = _training_export_summary(conn, where_sql, params)
    spend_summary = _weekly_spend_by_task(conn, where_sql, params)

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
        "training_export": training_export,
        "training_rows_used": int((training_export or {}).get("records_written") or 0),
        "training_rows_scanned": int((training_export or {}).get("rows_scanned") or 0),
        "feedback_fingerprint": (training_export or {}).get("feedback_fingerprint"),
        "provider_freshness": provider_freshness,
        "weekly_spend_summary": spend_summary,
        "task_summary": overall["task_summary"],
        "model_task_signals_top": overall["model_task_signals_top"],
        "source_summary": source_summary,
        "by_source": by_source,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
