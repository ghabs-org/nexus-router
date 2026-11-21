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


def test_export_verified_only_recovers_corrections_and_dedupes(tmp_path, monkeypatch):
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
        rows = [
            # fallback + corrected_task should be recovered
            ("d-fb", "2026-11-01T00:00:00Z", "coding", "p/a", "p", "Design multi-region failover strategy", "fallback", 0.4, 1),
            # jev + null outcome + corrected_task should be recovered
            ("d-jev", "2026-11-02T00:00:00Z", "general_chat", "p/a", "p", "Review this PR diff for SQL injection", "jev", 0.7, None),
            # llm + verdict='correct' should be recovered
            ("d-llm", "2026-11-03T00:00:00Z", "fast_utility", "p/a", "p", "Translate this release changelog to Spanish", "llm", 0.9, 1),
            # wrong verdict with NULL corrected_task must be excluded
            ("d-wrong-null", "2026-11-04T00:00:00Z", "coding", "p/a", "p", "Compare Kubernetes vs Nomad", "local", 0.85, 1),
            # unverified local row: excluded in verified_only=True
            ("d-unverified", "2026-11-05T00:00:00Z", "coding", "p/a", "p", "Write pytest fixtures for auth cache", "local", 0.9, 1),
            # duplicate prompt text: newer unverified + older verified -> verified label must win, emitted once
            ("d-dup-old-ver", "2026-11-01T00:00:00Z", "coding", "p/a", "p", "Analyze root cause of deadlock", "local", 0.8, 1),
            ("d-dup-new-unv", "2026-11-06T00:00:00Z", "coding", "p/a", "p", "  Analyze   root cause of deadlock ", "local", 0.88, 1),
            # extra coding & reasoning rows to exercise stratified split
            ("d-c1", "2026-11-07T00:00:00Z", "coding", "p/a", "p", "Fix off-by-one bug in tokenizer", "local", 0.9, 1),
            ("d-c2", "2026-11-08T00:00:00Z", "coding", "p/a", "p", "Implement retry backoff helper", "local", 0.9, 1),
            ("d-r1", "2026-11-09T00:00:00Z", "reasoning", "p/a", "p", "Evaluate tradeoffs between CRDTs and OT", "local", 0.9, 1),
        ]
        for r in rows:
            conn.execute(
                """
                INSERT INTO routing_decisions (
                  id, created_at, task_type, selected_model, selected_provider,
                  message_text, classifier_source, classifier_confidence, outcome_success
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                r,
            )
        feedback = [
            ("f1", "2026-11-01T01:00:00Z", "d-fb", "wrong", "reasoning"),
            ("f2", "2026-11-02T01:00:00Z", "d-jev", "wrong", "code_review"),
            ("f3", "2026-11-03T01:00:00Z", "d-llm", "correct", None),
            ("f4", "2026-11-04T01:00:00Z", "d-wrong-null", "wrong", None),
            # older feedback superseded by newer correction
            ("f5a", "2026-11-01T01:00:00Z", "d-dup-old-ver", "wrong", "fast_utility"),
            ("f5b", "2026-11-01T02:00:00Z", "d-dup-old-ver", "wrong", "reasoning"),
            ("f6", "2026-11-07T01:00:00Z", "d-c1", "correct", None),
            ("f7", "2026-11-08T01:00:00Z", "d-c2", "correct", None),
            ("f8", "2026-11-09T01:00:00Z", "d-r1", "correct", None),
        ]
        for fb in feedback:
            conn.execute(
                "INSERT INTO route_feedback (id, created_at, decision_id, verdict, corrected_task) VALUES (?,?,?,?,?)",
                fb,
            )
        conn.commit()
    finally:
        conn.close()

    out_path = tmp_path / "verified.jsonl"
    split_dir = tmp_path / "split"
    summary = exporter.export(
        str(out_path),
        min_samples=1,
        verified_only=True,
        split_dir=split_dir,
        eval_ratio=0.3,
        seed=42,
    )

    records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    by_id = {r["decision_id"]: r for r in records}

    assert set(by_id) == {"d-fb", "d-jev", "d-llm", "d-dup-old-ver", "d-c1", "d-c2", "d-r1"}
    assert by_id["d-fb"]["label"] == "reasoning"
    assert by_id["d-jev"]["label"] == "code_review"
    assert by_id["d-llm"]["label"] == "fast_utility"
    assert by_id["d-dup-old-ver"]["label"] == "reasoning"
    assert summary["records_written"] == 7  # no 2x reasoning duplication
    assert summary["verified_records"] == 7

    train_rows = [json.loads(l) for l in (split_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in (split_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    train_norm = {exporter._normalize_text(r["text"]) for r in train_rows}
    eval_norm = {exporter._normalize_text(r["text"]) for r in eval_rows}
    assert train_norm.isdisjoint(eval_norm)
    assert len(train_rows) + len(eval_rows) == 7
    # Rare 1-sample classes (code_review, fast_utility) must stay in train, not eval.
    eval_labels = {r["label"] for r in eval_rows}
    assert "code_review" not in eval_labels
    assert "fast_utility" not in eval_labels
