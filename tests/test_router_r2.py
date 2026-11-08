"""Round 2: token auth, dry run, caller catalog, /observe, /decisions, pricing, /metrics, hot reload."""

import threading
import time
from http.server import ThreadingHTTPServer

import httpx
import pytest

from src import db, pricing, server
from src.health import load_provider_health
from src.router import Router
from src.types import ClassifierOutput, PreSignals

CODING = ClassifierOutput(task_type="coding", complexity="medium", confidence=0.9, needs_tools=True)


@pytest.fixture
def router(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    return Router(persist=True)


@pytest.fixture
def http(router, monkeypatch):
    monkeypatch.setattr(server, "_router", router)
    monkeypatch.setattr(server, "ROUTER_TOKEN", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.RouterHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=10, trust_env=False) as client:
        yield client
    httpd.shutdown()
    httpd.server_close()


def route(client, **extra):
    body = {"message": "fix the failing parser test", "classifier": {"task_type": "coding", "confidence": 0.9}, **extra}
    return client.post("/route", json=body, headers=extra.pop("headers", None) if "headers" in extra else None)


def decision_count():
    with db._connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM routing_decisions").fetchone()[0]


def test_token_protects_everything_but_health(http, monkeypatch):
    monkeypatch.setattr(server, "ROUTER_TOKEN", "s3cret-token")
    assert http.get("/health").status_code == 200
    for method, path in (("GET", "/diagnostics"), ("GET", "/stats"), ("GET", "/metrics"), ("POST", "/probe"),
                         ("POST", "/feedback"), ("POST", "/route-mode"), ("POST", "/route"), ("GET", "/v1/models")):
        response = http.request(method, path, json={} if method == "POST" else None)
        assert response.status_code == 401, path
        assert response.json() == {"error": "unauthorized"}
    assert http.get("/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert http.get("/stats", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
    # CLIs attach with the token as their API key.
    assert http.get("/v1/models", headers={"x-api-key": "s3cret-token"}).status_code == 200
    assert http.get("/stats", headers={"x-api-key": "s3cret-token"}).status_code == 401  # only /v1/*
    status = http.get("/v1/nexus/cli-status", headers={"Authorization": "Bearer s3cret-token"}).json()
    assert "export ANTHROPIC_API_KEY=$NEXUS_ROUTER_TOKEN" in status["exports"]
    assert not any("s3cret-token" in line for line in status["exports"])


def test_dry_run_scores_without_writing(http):
    response = route(http, dry_run=True)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["decision_id"] is None and data["selected_model"]
    assert "dry_run" in data["explain"]["rules_triggered"]
    assert decision_count() == 0
    assert route(http, dry_run="yes").status_code == 400


def test_dry_run_never_explores(router):
    router._routing = {**router._routing, "scoring": {**(router._routing.get("scoring") or {}),
                                                      "exploration": {"enabled": True, "epsilon": 1.0}}}
    picks = {router.route(CODING, PreSignals(), dry_run=True).selected_model for _ in range(15)}
    assert len(picks) == 1


def test_caller_catalog_routes_to_models_newer_than_the_registry(router):
    assert not any(m["id"] == "anthropic/claude-opus-9-9" for m in router._registry)
    decision = router.route(CODING, PreSignals(), dry_run=True, available_models=[
        "anthropic/claude-opus-9-9", {"id": "openai-codex/gpt-9-sol", "context_window": 400000, "vision": True},
    ])
    assert decision.selected_model in {"anthropic/claude-opus-9-9", "openai-codex/gpt-9-sol"}
    candidates = {m["model"] for m in decision.explain["model_scores"]}
    assert candidates <= {"anthropic/claude-opus-9-9", "openai-codex/gpt-9-sol"}
    assert "caller_catalog" in decision.explain["rules_triggered"]
    # Known registry ids are reused but forced available.
    known = next(m["id"] for m in router._registry if m["provider"] == "anthropic")
    decision = router.route(CODING, PreSignals(), dry_run=True, available_models=[known])
    assert decision.selected_model == known


def test_caller_catalog_validation(http):
    assert route(http, dry_run=True, available_models=["no-slash"]).status_code == 400
    assert route(http, dry_run=True, available_models=[{"id": "a/b", "extra": 1}]).status_code == 400
    assert route(http, dry_run=True, available_models=["a/b"] * 201).status_code == 400
    assert route(http, dry_run=True, available_models=[{"id": "anthropic/claude-opus-9-9", "vision": True}]).status_code == 200


def test_observe_feeds_provider_health(http):
    assert http.post("/observe", json={"provider": "github-copilot", "http_status": 429, "error_type": "rate_limit",
                                       "quota_hint": "low"}).status_code == 200
    health = load_provider_health(["github-copilot"])["github-copilot"]
    assert health.rate_limit_cooldown_until
    assert http.post("/observe", json={"provider": ""}).status_code == 400
    assert http.post("/observe", json={"provider": "x", "http_status": 99}).status_code == 400
    assert http.post("/observe", json={"provider": "x", "error_type": "weird"}).status_code == 400


def test_decisions_list_links_without_message_text(http):
    first = route(http, nexus_context={"nexus_project": "ws", "nexus_workflow_id": "run-1", "nexus_step_id": "draft"},
                  source_type="studio", source_tag="studio:run-1").json()
    route(http, source_type="telegram")
    items = http.get("/decisions", params={"source_type": "studio"}).json()["items"]
    assert [i["id"] for i in items] == [first["decision_id"]]
    assert items[0]["nexus_workflow_id"] == "run-1" and items[0]["source_tag"] == "studio:run-1"
    assert "message_text" not in items[0]
    info = http.get("/route-info", params={"id": first["decision_id"]}).json()
    assert info["nexus_step_id"] == "draft"
    assert len(http.get("/decisions", params={"limit": "1"}).json()["items"]) == 1


def test_pricing_tariff_subscription_and_proxy(http, tmp_path, monkeypatch):
    table = tmp_path / "pricing.yaml"
    table.write_text('models:\n  "openai/*": {input_per_mtok: 2.0, output_per_mtok: 8.0}\n'
                     '  "openai/gpt-x-mini": {input_per_mtok: 0.5, output_per_mtok: 1.0}\n'
                     'subscription_providers: [openai-codex]\n')
    monkeypatch.setattr(pricing, "PRICING_FILE", table)
    monkeypatch.setitem(pricing._cache, "mtime", None)
    assert pricing.estimate("openai/gpt-x", "openai", 1_000_000, 500_000) == (6.0, "tariff")
    assert pricing.estimate("openai/gpt-x-mini", "openai", 1_000_000, 0) == (0.5, "tariff")
    assert pricing.estimate("openai-codex/gpt-x", "openai-codex", 10, 10) == (0.0, "subscription")
    assert pricing.estimate("anthropic/x", "anthropic", 10, 10) == (None, None)
    assert http.get("/pricing").json()["subscription_providers"] == ["openai-codex"]
    data = route(http).json()
    http.post("/outcome", json={"decision_id": data["decision_id"], "success": True, "input_tokens": 100, "output_tokens": 50})
    info = http.get("/route-info", params={"id": data["decision_id"]}).json()
    assert info["cost_source"] in {"tariff", "subscription", "proxy"}
    stats = http.get("/stats").json()["models"]
    assert any("estimated_cost_usd_24h" in row for row in stats)


def test_metrics_are_prometheus_text(http):
    data = route(http).json()
    http.post("/outcome", json={"decision_id": data["decision_id"], "success": True})
    response = http.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for name in ("nexus_router_decisions_total", "nexus_router_outcomes_total", "nexus_router_provider_health",
                 "nexus_router_model_cooldowns_active", "nexus_router_cli_turns_total"):
        assert f"# TYPE {name} " in text
    assert 'nexus_router_outcomes_total{model="%s",success="true"} 1' % data["selected_model"] in text


def test_routing_policy_hot_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    routing = tmp_path / "routing.yaml"
    routing.write_text("scoring:\n  confidence_gate:\n    min_confidence: 0.65\n")
    reloadable = Router(routing_path=routing, persist=False)
    low = ClassifierOutput(task_type="coding", confidence=0.80)
    assert not reloadable.route(low, PreSignals()).confidence_gate_triggered
    time.sleep(0.01)
    routing.write_text("scoring:\n  confidence_gate:\n    min_confidence: 0.95\n")
    import os
    os.utime(routing, (time.time() + 5, time.time() + 5))
    assert reloadable.route(low, PreSignals()).confidence_gate_triggered
    routing.write_text("scoring: [broken")
    os.utime(routing, (time.time() + 10, time.time() + 10))
    assert reloadable.route(low, PreSignals()).confidence_gate_triggered  # last good policy kept
