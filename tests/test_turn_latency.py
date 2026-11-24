"""Recorded turn times per model and task feed the latency objective."""

from datetime import datetime, timedelta, timezone

import uuid

import pytest

from src import db, scorer
from src.types import ClassifierOutput


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    db.ensure_schema()


def turns(model, task, *seconds, ok=1, fallback=0, days_ago=1):
    at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with db._connect() as conn:
        for i, s in enumerate(seconds):
            conn.execute(
                "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, selected_provider, "
                "outcome_success, fallback_used, latency_ms) VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), at, task, model, model.split("/")[0],
                 ok, fallback, int(s * 1000)),
            )


def test_medians_skip_failed_fallback_broken_and_old_turns(state):
    turns("nvidia/slow", "fast_utility", 58, 61, 69)
    turns("openai/quick", "fast_utility", 6, 8, 7, 9)
    turns("openai/quick", "fast_utility", 300, ok=0)
    turns("openai/quick", "fast_utility", 300, fallback=1)
    turns("openai/quick", "fast_utility", 0.2)
    turns("openai/quick", "fast_utility", 300, days_ago=90)
    stats = db.load_model_stats()
    quick = stats["openai/quick"]["turn_latency"]["fast_utility"]
    assert quick["samples"] == 4 and quick["median_ms"] == 7500
    assert stats["nvidia/slow"]["turn_latency"]["fast_utility"]["median_ms"] == 61000
    assert quick["task_median_ms"] == 9000  # over all 7 valid fast_utility turns


def test_slow_model_loses_to_quick_one_on_the_same_task(state):
    turns("nvidia/slow", "fast_utility", 58, 61, 69)
    turns("openai/quick", "fast_utility", 6, 8, 7, 9)
    stats = db.load_model_stats()
    slow = scorer._turn_latency_score("nvidia/slow", "fast_utility", stats)
    quick = scorer._turn_latency_score("openai/quick", "fast_utility", stats)
    assert quick[0] > 0.5 > slow[0]
    # Too few turns on a task, or none, leave the provider probe in charge.
    assert scorer._turn_latency_score("nvidia/slow", "coding", stats) is None
    turns("openai/rare", "fast_utility", 5, 5)
    assert scorer._turn_latency_score("openai/rare", "fast_utility", db.load_model_stats()) is None


def test_scores_use_and_show_recorded_turn_times(state):
    models = [
        {"id": "nvidia/slow", "provider": "nvidia", "availability": {"authed": True}, "scores": {"fast": 0.7}},
        {"id": "openai/quick", "provider": "openai", "availability": {"authed": True}, "scores": {"fast": 0.7}},
    ]
    classifier = ClassifierOutput(task_type="fast_utility", confidence=0.9)
    before = {s.model_id: s for s in scorer.score_models(classifier, models, {}, {})}
    turns("nvidia/slow", "fast_utility", 58, 61, 69)
    turns("openai/quick", "fast_utility", 6, 8, 7, 9)
    after = {s.model_id: s for s in scorer.score_models(classifier, models, {}, db.load_model_stats())}
    assert after["openai/quick"].total_score - after["nvidia/slow"].total_score > \
        before["openai/quick"].total_score - before["nvidia/slow"].total_score
    assert after["nvidia/slow"].component_scores["turn_latency_s"] == 61.0
    assert after["nvidia/slow"].component_scores["turn_latency_samples"] == 3
