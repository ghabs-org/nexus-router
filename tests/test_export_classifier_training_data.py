from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_export_includes_only_completed_outcomes(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "router.sqlite"))

    import src.paths as paths
    import src.db as db
    import scripts.export_classifier_training_data as exporter

    importlib.reload(paths)
    importlib.reload(db)
    importlib.reload(exporter)

    db.ensure_schema()
    conn = db._connect()
    try:
        now = db._now_iso()
        conn.execute(
            """
            INSERT INTO routing_decisions (
              id, created_at, task_type, selected_model, selected_provider,
              message_text, classifier_source, classifier_confidence, outcome_success
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("done-1", now, "coding", "p/a", "p", "Implement API endpoint", "local", 0.91, 1),
        )
        conn.execute(
            """
            INSERT INTO routing_decisions (
              id, created_at, task_type, selected_model, selected_provider,
              message_text, classifier_source, classifier_confidence, outcome_success
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("pending-1", now, "coding", "p/a", "p", "Refactor this module", "local", 0.88, None),
        )
        conn.commit()
    finally:
        conn.close()

    out_path = tmp_path / "training.jsonl"
    summary = exporter.export(str(out_path), min_samples=1)

    lines = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert summary["rows_scanned"] == 1
    assert summary["records_written"] == 1
    assert len(lines) == 1
    assert lines[0]["text"] == "Implement API endpoint"
