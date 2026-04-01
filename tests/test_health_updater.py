"""Tests for provider health probe handling."""

from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import health
from src import health_updater


class _Result:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_provider_auth_marks_capacity_failure_as_exhausted_quota(monkeypatch):
    calls = {"record": None, "log": None}

    def fake_run(*args, **kwargs):
        return _Result(1, stdout="", stderr="HTTP 429 Too Many Requests: No capacity available; quota exhausted for now")

    def fake_record_observation(**kwargs):
        calls["record"] = kwargs

    def fake_log_provider_observation(**kwargs):
        calls["log"] = kwargs

    monkeypatch.setattr(health_updater.subprocess, "run", fake_run)
    monkeypatch.setattr(health_updater, "record_observation", fake_record_observation)
    monkeypatch.setattr(health_updater, "log_provider_observation", fake_log_provider_observation)

    result = health_updater.probe_provider_auth("google-gemini-cli")

    assert result["auth"] == "ok"
    assert result["quota"] == "exhausted"
    assert calls["record"]["provider"] == "google-gemini-cli"
    assert calls["record"]["quota_state"] == "exhausted"
    assert calls["record"]["http_status"] == 429
    assert calls["log"]["quota_state"] == "exhausted"
    assert calls["log"]["http_status"] == 429


def test_observe_turn_outcome_escalates_repeated_429s_to_exhausted(monkeypatch):
    calls = {"record": [], "log": []}

    def fake_record_observation(**kwargs):
        calls["record"].append(kwargs)

    def fake_log_provider_observation(**kwargs):
        calls["log"].append(kwargs)

    monkeypatch.setattr(health_updater, "record_observation", fake_record_observation)
    monkeypatch.setattr(health_updater, "log_provider_observation", fake_log_provider_observation)

    health_updater.observe_turn_outcome("google-gemini-cli", 429, None, "capacity", None, None)

    assert calls["record"][-1]["quota_state"] == "exhausted"
    assert calls["record"][-1]["http_status"] == 429
    assert calls["log"][-1]["quota_state"] == "exhausted"


def test_record_observation_soft_bans_provider_after_rate_limit(monkeypatch):
    state = {}

    monkeypatch.setattr(health, "load_provider_health_state", lambda providers=None: state)
    monkeypatch.setattr(health, "upsert_provider_health_state", lambda provider, payload: state.__setitem__(provider, {"provider": provider, **payload}))

    health.record_observation(
        provider="google-gemini-cli",
        auth_status="unknown",
        quota_state="low",
        error_type="rate_limit",
        http_status=429,
    )

    providers = health.load_provider_health(["google-gemini-cli"])
    provider = providers["google-gemini-cli"]

    assert provider.consecutive_rate_limits == 1
    assert provider.rate_limit_cooldown_until is not None
    assert provider.health_score == 0.0


def test_record_observation_escalates_soft_ban_window_with_repeat_429s(monkeypatch):
    state = {}

    monkeypatch.setattr(health, "load_provider_health_state", lambda providers=None: state)
    monkeypatch.setattr(health, "upsert_provider_health_state", lambda provider, payload: state.__setitem__(provider, {"provider": provider, **payload}))

    base = datetime(2026, 3, 31, 13, 0, tzinfo=timezone.utc)
    moments = [
        base,
        base,
        (base + timedelta(minutes=1)),
        (base + timedelta(minutes=1)),
    ]

    monkeypatch.setattr(health, "_now_dt", lambda: moments.pop(0))

    health.record_observation(
        provider="google-gemini-cli",
        auth_status="unknown",
        quota_state="low",
        error_type="rate_limit",
        http_status=429,
    )
    health.record_observation(
        provider="google-gemini-cli",
        auth_status="unknown",
        quota_state="low",
        error_type="rate_limit",
        http_status=429,
    )

    providers = health.load_provider_health(["google-gemini-cli"])
    provider = providers["google-gemini-cli"]
    cooldown_until = datetime.fromisoformat(provider.rate_limit_cooldown_until)

    assert provider.consecutive_rate_limits == 2
    assert cooldown_until == base + timedelta(minutes=15)


def test_record_observation_clears_rate_limit_soft_ban_after_success(monkeypatch):
    state = {}

    monkeypatch.setattr(health, "load_provider_health_state", lambda providers=None: state)
    monkeypatch.setattr(health, "upsert_provider_health_state", lambda provider, payload: state.__setitem__(provider, {"provider": provider, **payload}))

    health.record_observation(
        provider="google-gemini-cli",
        auth_status="unknown",
        quota_state="low",
        error_type="rate_limit",
        http_status=429,
    )
    health.record_observation(
        provider="google-gemini-cli",
        auth_status="ok",
        quota_state="healthy",
        error_type=None,
        http_status=200,
    )

    providers = health.load_provider_health(["google-gemini-cli"])
    provider = providers["google-gemini-cli"]

    assert provider.consecutive_rate_limits == 0
    assert provider.rate_limit_cooldown_until is None
    assert provider.health_score > 0.0
