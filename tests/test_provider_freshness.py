from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provider_freshness import build_provider_freshness_snapshot, evaluate_freshness_transitions


def _iso(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_stale_counts_exempt_disabled_and_dead_providers(tmp_path):
    providers = {
        "healthy": {"provider": "healthy", "enabled": 1, "status": "enabled"},
        "disabled": {"provider": "disabled", "enabled": 0, "status": "disabled"},
        "dead_auth": {"provider": "dead_auth", "enabled": 1, "status": "enabled"},
        "dead_quota": {"provider": "dead_quota", "enabled": 1, "status": "enabled"},
    }
    health = {
        "healthy": {"provider": "healthy", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(30)},
        "disabled": {"provider": "disabled", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(30)},
        "dead_auth": {"provider": "dead_auth", "auth": "expired", "quota": "healthy", "latency_updated_at": _iso(30)},
        "dead_quota": {"provider": "dead_quota", "auth": "ok", "quota": "exhausted", "latency_updated_at": _iso(30)},
    }

    snapshot = build_provider_freshness_snapshot(providers, health, stale_after_hours=24)
    assert snapshot["stale_candidate_count"] == 1
    assert snapshot["stale_candidates"] == ["healthy"]
    assert snapshot["exempt_reasons"]["provider_disabled"] == 1
    assert snapshot["exempt_reasons"]["auth_blocked"] == 1
    assert snapshot["exempt_reasons"]["quota_exhausted"] == 1


def test_transition_alert_only_on_healthy_to_stale(tmp_path):
    state_path = tmp_path / "provider-freshness-state.json"

    providers = {
        "a": {"provider": "a", "enabled": 1, "status": "enabled"},
        "b": {"provider": "b", "enabled": 1, "status": "enabled"},
    }
    first_health = {
        "a": {"provider": "a", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(2)},
        "b": {"provider": "b", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(2)},
    }
    first = evaluate_freshness_transitions(providers, first_health, stale_after_hours=24, state_path=state_path)
    assert first["newly_stale"] == []

    second_health = {
        "a": {"provider": "a", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(30)},
        "b": {"provider": "b", "auth": "ok", "quota": "healthy", "latency_updated_at": _iso(2)},
    }
    second = evaluate_freshness_transitions(providers, second_health, stale_after_hours=24, state_path=state_path)
    assert second["newly_stale"] == ["a"]

    third = evaluate_freshness_transitions(providers, second_health, stale_after_hours=24, state_path=state_path)
    assert third["newly_stale"] == []

    stored = json.loads(state_path.read_text())
    assert "a" in stored["stale_candidates"]
