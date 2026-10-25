import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NEXUS_ROUTER_STATE_ROOT", str(Path(__file__).parent.parent / ".test-state"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.refresh_catalog as refresh_catalog
from scripts.refresh_catalog import check


def _models(provider: str, count: int, configured: bool = True) -> list[dict]:
    tags = ["configured"] if configured else []
    return [{"key": f"{provider}/m{i}", "available": True, "tags": tags} for i in range(count)]


def _catalog(*groups: list[dict], outcomes: dict[str, str] | None = None) -> dict:
    catalog = {"models": [m for group in groups for m in group]}
    if outcomes is not None:
        catalog["providerOutcomes"] = [{"provider": p, "status": s} for p, s in outcomes.items()]
    return catalog


def test_check_accepts_first_fetch():
    assert check(_catalog(_models("nvidia", 5)), None) == []


def test_check_refuses_empty_catalog():
    assert check({"models": []}, _catalog(_models("nvidia", 5))) == ["catalog has no models"]


def test_check_refuses_catalog_without_configured_models():
    problems = check(_catalog(_models("nvidia", 5, configured=False)), None)
    assert problems == ["no model is tagged 'configured'"]


def test_check_allows_provider_disabled_in_openclaw():
    # A disabled provider plugin vanishes with no discovery outcome (github-copilot, Oct 2026).
    old = _catalog(_models("github-copilot", 6), _models("nvidia", 10))
    new = _catalog(_models("nvidia", 10), outcomes={"nvidia": "ready"})
    assert check(new, old) == []


def test_check_refuses_provider_lost_to_failed_discovery():
    old = _catalog(_models("opencode", 10), _models("nvidia", 10))
    new = _catalog(_models("opencode", 2), _models("nvidia", 10), outcomes={"opencode": "auth-rejected", "nvidia": "ready"})
    assert check(new, old) == ["opencode dropped 10 -> 2 while discovery reported auth-rejected"]


def test_check_allows_failed_discovery_that_kept_its_models():
    old = _catalog(_models("opencode", 10))
    new = _catalog(_models("opencode", 12), outcomes={"opencode": "auth-rejected"})
    assert check(new, old) == []


def test_check_refuses_large_total_drop():
    old = _catalog(_models("nvidia", 10), _models("openrouter", 10))
    new = _catalog(_models("nvidia", 9))
    assert check(new, old) == ["total models dropped 20 -> 9"]



def test_fetch_allowlist_reads_openclaw_config(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='warning line\n["nvidia/*", "anthropic/claude-opus-5-5"]\n', stderr="")

    monkeypatch.setattr(refresh_catalog.subprocess, "run", fake_run)

    assert refresh_catalog.fetch_allowlist(10) == ["nvidia/*", "anthropic/claude-opus-5-5"]
    assert calls[0][1:] == ["config", "get", "agents.defaults.modelPolicy.allow", "--json"]


def test_fetch_allowlist_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(
        refresh_catalog.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Config path not found"),
    )

    assert refresh_catalog.fetch_allowlist(10) is None
