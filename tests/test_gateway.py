"""Real serialized HTTP through the gateway and Vertex-shaped upstream stand-ins.

Use NEXUS_TEST_SOCKETPAIR=1 in restricted sandboxes; ordinary runs use TCP.
No upstream HTTP client is stubbed. These do not establish live Vertex support.
"""

import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from src import db, gateway
from src.cli_proxy import UsageTap
from src.gateway_cli import main
from src.gateway_policy import GatewayPolicy, PolicyError


class VertexUpstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.server.respond(self, body)


USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 30,
         "cache_creation_input_tokens": 20,
         "cache_creation": {"ephemeral_5m_input_tokens": 12, "ephemeral_1h_input_tokens": 8}}


def answer(handler, body):
    streaming = handler.path.endswith(":streamRawPredict")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
    handler.end_headers()
    if streaming:
        first = {"type": "message_start", "message": {"id": "msg-1", "usage": {k: v for k, v in USAGE.items() if k != "output_tokens"}}}
        handler.wfile.write(f"event: message_start\ndata: {json.dumps(first)}\n\n".encode())
        handler.wfile.flush()
        handler.server.first_chunk.set()
        handler.server.release.wait(5)
        handler.wfile.write(b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":5}}\n\n'
                            b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    else:
        handler.wfile.write(json.dumps({"id": "msg-1", "type": "message", "role": "assistant",
                                      "content": [{"type": "text", "text": "Done"}], "usage": USAGE}).encode())


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    monkeypatch.setenv("NEXUS_GATEWAY_TOKEN", "client-secret")
    monkeypatch.setenv("VERTEX_TEST_TOKEN", "google-secret")
    monkeypatch.setenv("SECOND_CLIENT_TOKEN", "other-secret")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    server_type = ThreadingHTTPServer
    if os.getenv("NEXUS_TEST_SOCKETPAIR") == "1":
        from cli_socketpair import install
        server_type = install(monkeypatch)
    monkeypatch.setattr(gateway, "ThreadingHTTPServer", server_type)
    upstream = server_type(("127.0.0.1", 0), VertexUpstream)
    upstream.requests, upstream.respond = [], answer
    upstream.first_chunk, upstream.release = threading.Event(), threading.Event()
    upstream.release.set()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    config = gateway.load_config(Path(__file__).parents[1] / "src/gateway.example.yaml")
    config["providers"]["vertex"].update(base_url=f"http://127.0.0.1:{upstream.server_port}", access_token_env="VERTEX_TEST_TOKEN")
    config["clients"]["second"] = {"token_env": "SECOND_CLIENT_TOKEN"}
    config.update(connect_timeout_seconds=.3, read_timeout_seconds=2)
    for model in config["models"].values():
        model["pricing"] = {"input": 3, "output": 15, "cache_read": .3, "cache_write_5m": 3.75, "cache_write_1h": 6}
    server = gateway.create_server(config, port=0)
    db.upsert_provider_health_state("vertex", {"auth": "ok", "quota": "healthy", "health_score": 1})
    with db._connect() as conn:
        conn.execute("UPDATE providers SET enabled=1,status='enabled' WHERE provider='vertex'")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", headers={"Authorization": "Bearer client-secret"},
                      timeout=5, trust_env=False) as client:
        yield client, server, upstream, config
    upstream.release.set()
    server.shutdown()
    server.server_close()
    upstream.shutdown()
    upstream.server_close()


def payload(text="Summarize this quarterly report"):
    return {"model": "nexus-auto", "max_tokens": 4096,
            "system": [{"type": "text", "text": "You are Cowork", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": text}]}


@pytest.mark.parametrize("text,model,effort", [
    ("Rewrite this email", "claude-haiku-4-5@20251001", "model-default"),
    ("Summarize this report", "claude-sonnet-4-6", "medium"),
    ("Analyze conflicting evidence", "claude-opus-4-6", "high"),
    ("Traduci questa email", "claude-haiku-4-5@20251001", "model-default"),
])
def test_native_vertex_routes_and_preserves_cache(harness, text, model, effort):
    client, server, upstream, config = harness
    body = payload(text)
    original = copy.deepcopy(body)
    response = client.post("/v1/messages", json=body, headers={"anthropic-beta": "example-beta", "Cookie": "secret"})
    assert response.status_code == 200, response.text
    path, headers, forwarded = upstream.requests[-1]
    assert path.endswith(f"/models/{model.replace('@', '%40')}:rawPredict")
    assert "/projects/test-project/locations/europe-west1/" in path
    assert headers["authorization"] == "Bearer google-secret"
    assert headers["anthropic-beta"] == "example-beta"
    assert "Cookie" not in headers and "x-api-key" not in headers
    assert "model" not in forwarded and "stream" not in forwarded
    assert forwarded["anthropic_version"] == "vertex-2023-10-16"
    assert forwarded["system"] == original["system"] and forwarded["tools"] == original["tools"]
    assert body == original
    assert response.headers["x-nexus-effort"] == effort
    status = client.get("/v1/nexus/status").json()
    assert status["usage_24h"]["input_tokens"] == 60
    assert status["usage_24h"]["cache_read_input_tokens"] == 30
    assert status["usage_24h"]["estimated_cost_usd"] == pytest.approx(.000207)


def test_model_effort_pin_survives_restart_and_user_followups(harness):
    client, server, upstream, config = harness
    headers = {"x-session-id": "task-123"}
    body = payload("Analyze conflicting evidence")
    assert client.post("/v1/messages", json=body, headers=headers).status_code == 200
    body["messages"].append({"role": "assistant", "content": "Plan"})
    body["messages"].append({"role": "user", "content": "Rewrite the email"})
    body["thinking"] = {"type": "disabled"}
    body["output_config"] = {"effort": "low"}
    server.gateway = gateway.GatewayProxy(config)
    response = client.post("/v1/messages", json=body, headers=headers)
    assert response.status_code == 200
    assert response.headers["x-nexus-cache-pinned"] == "true"
    assert upstream.requests[-1][2]["thinking"] == {"type": "adaptive"}
    assert upstream.requests[-1][2]["output_config"]["effort"] == "high"
    assert "claude-opus-4-6" in upstream.requests[-1][0]


def test_explicit_override_is_validated_and_task_change_rejected(harness):
    client, server, upstream, config = harness
    body = payload()
    body.update(model="claude-sonnet-4-6", output_config={"effort": "low"})
    headers = {"x-session-id": "explicit"}
    assert client.post("/v1/messages", json=body, headers=headers).status_code == 200
    body["output_config"]["effort"] = "high"
    assert client.post("/v1/messages", json=body, headers=headers).status_code == 409
    body["model"] = "unknown-model"
    assert client.post("/v1/messages", json=body).status_code == 400
    body.update(model="claude-haiku-4-5@20251001", output_config={"effort": "high"})
    assert client.post("/v1/messages", json=body).status_code == 400
    assert len(upstream.requests) == 1


def test_tenants_cannot_share_pins_or_usage(harness):
    client, server, upstream, config = harness
    headers = {"x-session-id": "shared-id"}
    assert client.post("/v1/messages", json=payload("Analyze architecture"), headers=headers).status_code == 200
    headers["Authorization"] = "Bearer other-secret"
    response = client.post("/v1/messages", json=payload("Rewrite this email"), headers=headers)
    assert response.status_code == 200 and response.headers["x-nexus-profile"] == "utility"
    assert client.get("/v1/nexus/status", headers=headers).json()["usage_24h"]["turns"] == 1
    assert client.get("/v1/nexus/status").json()["usage_24h"]["turns"] == 1


def test_auth_disabled_provider_preview_and_count_tokens(harness):
    client, server, upstream, config = harness
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer bad"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "", "x-api-key": "client-secret"}).status_code == 200
    preview = client.post("/v1/nexus/route-preview", json=payload()).json()
    assert preview["profile"] == "standard" and not preview["writes_pin"]
    assert client.post("/v1/messages/count_tokens", json=payload()).status_code == 404
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM gateway_task_pins").fetchone()[0] == 0
        conn.execute("UPDATE providers SET enabled=0 WHERE provider='vertex'")
    assert client.post("/v1/messages", json=payload()).status_code == 503
    assert not upstream.requests


def test_streaming_flush_and_usage_accounting(harness):
    client, server, upstream, config = harness
    upstream.release.clear()
    body = payload()
    body["stream"] = True
    with client.stream("POST", "/v1/messages", json=body) as response:
        assert response.status_code == 200
        lines = response.iter_lines()
        assert next(lines) == "event: message_start"
        assert not upstream.release.is_set()
        upstream.release.set()
        remaining = list(lines)
        assert any("message_stop" in line for line in remaining)
    assert client.get("/v1/nexus/status").json()["usage_24h"]["input_tokens"] == 60


def test_unknown_model_fallback_and_signed_thinking_guard(harness):
    client, server, upstream, config = harness
    def fail_haiku(handler, body):
        if "haiku" in handler.path:
            handler.send_response(404)
            handler.end_headers()
        else:
            answer(handler, body)
    upstream.respond = fail_haiku
    headers = {"x-session-id": "thinking-task"}
    body = payload("Rewrite this email")
    response = client.post("/v1/messages", json=body, headers=headers)
    assert response.status_code == 200 and "sonnet" in response.headers["x-nexus-routed-model"]
    body["messages"].append({"role": "assistant", "content": [{"type": "thinking", "thinking": "private", "signature": "signed"}]})
    before = len(upstream.requests)
    response = client.post("/v1/messages", json=body, headers={"x-session-id": "new-task"})
    assert response.status_code == 409 and len(upstream.requests) == before
    response = client.post("/v1/messages", json=body, headers=headers)
    assert response.status_code == 200 and "sonnet" in upstream.requests[-1][0]
    assert upstream.requests[-1][2]["messages"][-1] == body["messages"][-1]


def test_token_admission_gate_and_output_cap(harness):
    client, server, upstream, config = harness
    server.gateway_auth.clients[0][2].update(max_tokens_24h=60)
    body = payload()
    body["max_tokens"] = 999999
    assert client.post("/v1/messages", json=body).status_code == 200
    assert upstream.requests[-1][2]["max_tokens"] == 16384
    response = client.post("/v1/messages", json=payload("Another task"))
    assert response.status_code == 402 and response.json()["error"]["type"] == "billing_error"
    assert len(upstream.requests) == 1


def test_legacy_vertex_beta_translation(harness):
    client, server, upstream, config = harness
    server.gateway.providers["vertex"]["beta_transport"] = "body"
    assert client.post("/v1/messages", json=payload(), headers={"anthropic-beta": "a,b"}).status_code == 200
    assert upstream.requests[-1][2]["anthropic_beta"] == ["a", "b"]
    assert "anthropic-beta" not in upstream.requests[-1][1]


def test_truncated_stream_is_not_replayed_and_blocks_budgeted_calls(harness):
    client, server, upstream, config = harness
    server.gateway_auth.clients[0][2].update(max_tokens_24h=100000)
    def truncated(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n')
    upstream.respond = truncated
    with client.stream("POST", "/v1/messages", json={**payload(), "stream": True}) as response:
        assert response.status_code == 200
        assert b"message_stop" not in b"".join(response.iter_bytes())
    assert len(upstream.requests) == 1
    assert client.get("/v1/nexus/status").json()["recent_turns"][0]["usage_complete"] == 0
    assert client.post("/v1/messages", json=payload("New task")).status_code == 402
    assert len(upstream.requests) == 1


def test_provider_rate_limit_does_not_retry_another_model_on_same_provider(harness):
    client, server, upstream, config = harness
    def limited(handler, body):
        handler.send_response(429)
        handler.end_headers()
    upstream.respond = limited
    assert client.post("/v1/messages", json=payload()).status_code == 503
    assert len(upstream.requests) == 1


def test_background_worker_does_not_change_main_task_profile(harness):
    client, server, upstream, config = harness
    headers = {"x-session-id": "main-and-worker"}
    assert client.post("/v1/messages", json=payload("Analyze architecture"), headers=headers).status_code == 200
    worker = payload("Extract these fields")
    worker["system"] = "You are a delegated worker"
    response = client.post("/v1/messages", json=worker, headers=headers)
    assert response.status_code == 200 and response.headers["x-nexus-profile"] == "utility"
    response = client.post("/v1/messages", json=payload("Thanks"), headers=headers)
    assert response.status_code == 200 and response.headers["x-nexus-profile"] == "deep"
    assert "opus" in response.headers["x-nexus-routed-model"]


def test_concurrency_admission_runs_before_a_second_upstream_request(harness):
    client, server, upstream, config = harness
    server.gateway_auth.clients[0][2].update(max_concurrent=1)
    upstream.release.clear()
    first_headers = {"x-session-id": "first-task"}
    principal, _ = server.gateway_auth.authenticate({"Authorization": "Bearer client-secret"})
    first_key = server.gateway.policy.identity(payload(), first_headers, principal)
    first_stripe = int(first_key[:8], 16) % len(server.gateway.session_locks)
    second_headers = None
    for index in range(10):
        candidate = {"x-session-id": f"other-task-{index}"}
        key = server.gateway.policy.identity(payload(), candidate, principal)
        if int(key[:8], 16) % len(server.gateway.session_locks) != first_stripe:
            second_headers = candidate
            break
    assert second_headers is not None
    with client.stream("POST", "/v1/messages", json={**payload(), "stream": True}, headers=first_headers) as response:
        lines = response.iter_lines()
        assert next(lines) == "event: message_start"
        assert client.post("/v1/messages", json=payload(), headers=second_headers).status_code == 429
        upstream.release.set()
        list(lines)
    assert len(upstream.requests) == 1


def test_database_does_not_store_prompts_or_credentials(harness):
    client, server, upstream, config = harness
    prompt = "Confidential example for privacy verification"
    assert client.post("/v1/messages", json=payload(prompt)).status_code == 200
    data = db.DB_PATH.read_bytes()
    assert all(value.encode() not in data for value in (prompt, "client-secret", "google-secret"))


def test_usage_tap_keeps_cache_categories_separate():
    tap = UsageTap(False, detailed=True)
    tap.feed(json.dumps({"usage": USAGE}).encode())
    usage = tap.finish()
    assert usage["input_tokens"] == 60 and usage["uncached_input_tokens"] == 10
    assert usage["ephemeral_1h_input_tokens"] == 8


def test_cli_init_check_and_cowork_configuration(tmp_path, monkeypatch, capsys):
    config, state, output = tmp_path / "gateway.yaml", tmp_path / "state", tmp_path / "cowork.json"
    common = ["--config", str(config), "--state-dir", str(state)]
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "original.sqlite")
    monkeypatch.setenv("NEXUS_ROUTER_STATE_ROOT", str(state))
    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(state / "data/router.sqlite"))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    monkeypatch.delenv("NEXUS_GATEWAY_TOKEN", raising=False)
    assert main(["init", *common]) == 0
    assert config.stat().st_mode & 0o777 == 0o600
    assert main(["init", *common]) == 1
    assert main(["check", *common]) == 0
    assert main(["enable-provider", "vertex", *common]) == 0
    assert main(["cowork-config", *common, "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["inferenceModels"][0]["name"] == "nexus-auto"
    assert result["inferenceModels"][0]["behavesAs"] == "claude-sonnet-4-6"
    assert result["inferenceGatewayApiKey"] not in capsys.readouterr().out
    assert output.stat().st_mode & 0o777 == 0o600
    assert main(["cowork-config", *common, "--output", str(tmp_path / "bad.json"), "--url", "http://shared.example"]) == 1
