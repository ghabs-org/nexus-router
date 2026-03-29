"""Tests for the generic quota-sync job."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import quota_sync


def test_derive_quota_state_from_ratio():
    assert quota_sync._derive_quota_state({"quota_remaining_ratio": 0.90}) == "healthy"
    assert quota_sync._derive_quota_state({"quota_remaining_ratio": 0.25}) == "low"
    assert quota_sync._derive_quota_state({"quota_remaining_ratio": 0.02}) == "exhausted"


def test_normalize_payload_accepts_object_list_and_wrapper():
    single = quota_sync._normalize_payload({"provider": "openai-codex"})
    wrapped = quota_sync._normalize_payload({"providers": [{"provider": "openai-codex"}, {"provider": "github-copilot"}]})
    listed = quota_sync._normalize_payload([{"provider": "openai-codex"}])

    assert len(single) == 1 and single[0]["provider"] == "openai-codex"
    assert [x["provider"] for x in wrapped] == ["openai-codex", "github-copilot"]
    assert listed[0]["provider"] == "openai-codex"


def test_sync_provider_snapshot_preserves_quota_ratio_and_logs(monkeypatch):
    calls = {"record": None, "log": None}

    def fake_record_observation(**kwargs):
        calls["record"] = kwargs

    def fake_log_provider_observation(**kwargs):
        calls["log"] = kwargs

    monkeypatch.setattr(quota_sync, "record_observation", fake_record_observation)
    monkeypatch.setattr(quota_sync, "log_provider_observation", fake_log_provider_observation)

    result = quota_sync.sync_provider_snapshot({
        "provider": "openai-codex",
        "auth": "ok",
        "quota_remaining_ratio": 0.24,
        "source": "collector:test",
    })

    assert result["quota"] == "low"
    assert result["quota_remaining_ratio"] == pytest.approx(0.24)
    assert calls["record"]["quota_remaining_ratio"] == pytest.approx(0.24)
    assert calls["record"]["quota_state"] == "low"
    assert calls["log"]["quota_state"] == "low"
    assert calls["log"]["note"] == "collector:test"
