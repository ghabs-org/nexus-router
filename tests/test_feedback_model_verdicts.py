"""The feedback cards' model verdicts (ok / too_cheap / too_powerful) score as meant."""

import pytest

from src import db
from src.feedback_taxonomy import normalize_reason_tag


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    db.ensure_schema()


def feedback(decision_id, model, routed_task, corrected_task, model_verdict):
    with db._connect() as conn:
        conn.execute(
            "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, selected_provider) "
            "VALUES (?,?,?,?,?)",
            (decision_id, db._now_iso(), routed_task, model, model.split("/")[0]),
        )
    db.record_feedback(decision_id, "wrong", corrected_task=corrected_task, model_verdict=model_verdict,
                       source_surface="telegram_feedback_card")


def test_model_ok_counts_for_the_model_on_the_corrected_task(state):
    for i in range(3):
        feedback(f"ok-{i}", "openai/gpt-6-astra", "general_chat", "code_review", "ok")
    for i in range(3):
        feedback(f"slow-{i}", "nvidia/nemotron-3-ultra", "fast_utility", "general_chat", "too_powerful")
    stats = db.load_model_stats()
    assert stats["openai/gpt-6-astra"]["feedback_preference"]["code_review"]["score"] == 0.75
    assert stats["nvidia/nemotron-3-ultra"]["feedback_preference"]["general_chat"]["score"] == 0.0


def test_model_ok_is_a_task_mismatch_not_a_model_one():
    tag = lambda mv: normalize_reason_tag(reason_tag=None, verdict="wrong", corrected_task="code_review",
                                          model_verdict=mv, preferred_model=None)
    assert tag("ok") == "task_mismatch"
    assert tag("too_cheap") == "task_and_model_mismatch"
