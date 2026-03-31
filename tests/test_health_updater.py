"""Tests for provider health probe handling."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

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
