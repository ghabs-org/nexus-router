from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_update_outcome_records_tokens_and_estimated_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "router.sqlite"))

    import src.paths as paths
    import src.db as db

    importlib.reload(paths)
    importlib.reload(db)

    db.ensure_schema()
    conn = db._connect()
    try:
        conn.execute(
            """
            INSERT INTO routing_decisions (
              id, created_at, task_type, selected_model, selected_provider,
              routing_score, selected_cost_score
            ) VALUES (?,?,?,?,?,?,?)
            """,
            ("dec-cost-1", db._now_iso(), "coding", "lab/quality-coder", "lab", 0.92, 0.30),
        )
        conn.commit()
    finally:
        conn.close()

    db.update_outcome(
        decision_id="dec-cost-1",
        success=True,
        latency_ms=1200,
        input_tokens=1500,
        output_tokens=300,
        total_tokens=1800,
    )

    conn = db._connect()
    try:
        row = conn.execute(
            "SELECT input_tokens, output_tokens, total_tokens, estimated_cost_usd FROM routing_decisions WHERE id=?",
            ("dec-cost-1",),
        ).fetchone()
    finally:
        conn.close()

    assert row["input_tokens"] == 1500
    assert row["output_tokens"] == 300
    assert row["total_tokens"] == 1800
    assert row["estimated_cost_usd"] is not None
    assert float(row["estimated_cost_usd"]) > 0


def test_weekly_spend_summary_aggregates_by_task(tmp_path):
    db_path = tmp_path / "router.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript((ROOT / "src" / "schema.sql").read_text())
    conn.execute(
        "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, estimated_cost_usd, total_tokens, outcome_success) VALUES (?,?,?,?,?,?,?)",
        ("a", "2099-01-01T00:00:00+00:00", "coding", "m/a", 0.12, 10000, 1),
    )
    conn.execute(
        "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, estimated_cost_usd, total_tokens, outcome_success) VALUES (?,?,?,?,?,?,?)",
        ("b", "2000-01-01T00:00:00+00:00", "coding", "m/a", 9.99, 90000, 1),
    )
    conn.commit()

    from scripts.feedback_calibration_report import _weekly_spend_by_task

    summary = _weekly_spend_by_task(conn, "", [])
    conn.close()

    assert summary["window"] == "7d"
    assert summary["pipeline"]["cost_enabled"] is True
    assert summary["pipeline"]["co2_enabled"] is False
    # sqlite datetime('now') means future test row always included, ancient row excluded
    coding = next(item for item in summary["by_task"] if item["task"] == "coding")
    assert coding["decisions"] == 1
    assert coding["estimated_spend_usd"] == 0.12
