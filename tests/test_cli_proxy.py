"""Real HTTP end-to-end checks: actual proxy + threaded upstream stand-ins + SQLite.

No upstream HTTP client is stubbed. The upstream request log proves the route ran.
"""

import copy
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from src import db, server
from src.cli_proxy import CliProxy
from src.health import load_provider_health


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.server.respond(self, body)


def answer(handler, body):
    streaming = body.get("stream")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
    handler.send_header("X-Upstream-Cache", "preserved")
    handler.end_headers()
    if streaming:
        handler.wfile.write(b'data: {"part":1}\n\n')
        handler.wfile.flush()
        handler.server.first_chunk.set()
        handler.server.release.wait(3)
        handler.wfile.write(b'data: {"part":2}\n\ndata: [DONE]\n\n')
    else:
        handler.wfile.write(json.dumps({"id": "reply1", "model": body["model"], "ok": True}).encode())


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    db.ensure_schema()
    server_type = ThreadingHTTPServer
    if os.getenv("NEXUS_TEST_SOCKETPAIR") == "1":
        from cli_socketpair import install
        server_type = install(monkeypatch)
    upstreams = []
    for name in ("primary", "fallback", "cheap"):
        upstream = server_type(("127.0.0.1", 0), Upstream)
        upstream.requests = []
        upstream.respond = answer
        upstream.first_chunk = threading.Event()
        upstream.release = threading.Event()
        upstream.release.set()
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        upstreams.append(upstream)
        db.upsert_provider_health_state(name, {"auth": "ok", "quota": "healthy", "health_score": 1})
        with db._connect() as conn:
            conn.execute("UPDATE providers SET enabled=1, status='enabled' WHERE provider=?", (name,))
    config = {"providers": {name: {"base_url": f"http://127.0.0.1:{up.server_port}/v1", "protocols": ["messages", "chat", "responses"]}
                            for name, up in zip(("primary", "fallback", "cheap"), upstreams)},
              "routes": {"MAIN_EXECUTION": ["primary:main", "fallback:backup"],
                         "MAIN_PLANNING": ["fallback:planner", "primary:planner"],
                         "STATELESS_HOUSEKEEPING": ["cheap:small"], "SUBAGENT_WORKER": ["cheap:small"]},
              "connect_timeout_seconds": .2, "read_timeout_seconds": 2}
    proxy = CliProxy(config)
    monkeypatch.setattr(server, "_cli_proxy", proxy)
    httpd = server_type(("127.0.0.1", 0), server.RouterHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=5, trust_env=False) as client:
        yield client, proxy, upstreams
    httpd.shutdown()
    httpd.server_close()
    for upstream in upstreams:
        upstream.release.set()
        upstream.shutdown()
        upstream.server_close()


def payload():
    return {"model": "cli-model", "system": [{"type": "text", "text": "You are Claude Code", "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "Task", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Implement a parser", "cache_control": {"type": "ephemeral"}}]}], "max_tokens": 4096}


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions", "/v1/responses"])
def test_native_http_endpoints_preserve_payload_headers_and_pin(harness, path):
    client, proxy, upstreams = harness
    body = payload() if path == "/v1/messages" else {"model": "cli", "input": "Build a parser"} if path.endswith("responses") else {"model": "cli", "messages": [{"role": "user", "content": "Build a parser"}]}
    original = copy.deepcopy(body)
    headers = {"anthropic-beta": "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14", "x-session-id": path, "Authorization": "Bearer CLI-secret"}
    first = client.post(path, json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert first.headers["X-Nexus-Cache-Pinned"] == "false"
    assert first.headers["X-Upstream-Cache"] == "preserved"
    assert first.headers["X-Nexus-Call-Category"] == "MAIN_EXECUTION"
    second = client.post(path, json=body, headers=headers)
    assert second.headers["X-Nexus-Cache-Pinned"] == "true"
    route, received_headers, received_body = upstreams[0].requests[0]
    assert route == path
    assert received_headers["anthropic-beta"] == headers["anthropic-beta"]
    assert "authorization" not in {k.lower() for k in received_headers}
    original["model"] = "main"
    assert received_body == original
    status = client.get("/v1/nexus/cli-status").json()
    assert len(status["active_session_pins"]) == 1
    assert len(status["recent_decisions"]) == 2
    assert "export OPENAI_BASE_URL=" in status["exports"][1]


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/chat/completions", "/v1/responses"])
def test_sse_first_chunk_arrives_before_upstream_finishes(harness, path):
    client, proxy, upstreams = harness
    upstreams[0].release.clear()
    body = payload()
    body["stream"] = True
    try:
        with client.stream("POST", path, json=body) as response:
            assert response.status_code == 200
            chunks = response.iter_raw()
            first = next(chunks)
            assert first == b'data: {"part":1}\n\n'
            assert upstreams[0].first_chunk.is_set()
            assert not upstreams[0].release.is_set()
            upstreams[0].release.set()
            rest = b"".join(chunks)
            assert rest == b'data: {"part":2}\n\ndata: [DONE]\n\n'
    finally:
        upstreams[0].release.set()


def test_pin_survives_policy_reorder_housekeeping_bypasses_and_phase_changes(harness):
    client, proxy, upstreams = harness
    headers = {"x-session-id": "shared"}
    assert client.post("/v1/messages", json=payload(), headers=headers).json()["model"] == "main"
    proxy.routes[next(c for c in proxy.routes if c.value == "MAIN_EXECUTION")].reverse()
    second = client.post("/v1/messages", json=payload(), headers=headers)
    assert second.json()["model"] == "main"
    assert second.headers["X-Nexus-Cache-Pinned"] == "true"
    house = {"model": "cli", "system": "Summarize the conversation", "max_tokens": 500}
    assert client.post("/v1/messages", json=house, headers=headers).json()["model"] == "small"
    assert client.post("/v1/messages", json=payload(), headers=headers).json()["model"] == "main"
    plan = {**payload(), "thinking": {"type": "enabled"}}
    planning = client.post("/v1/messages", json=plan, headers=headers)
    assert planning.json()["model"] == "planner"
    assert planning.headers["X-Nexus-Cache-Pinned"] == "false"


@pytest.mark.parametrize("code", [401, 403, 429, 500, 529, 503])
def test_http_failure_updates_real_db_and_fails_over(harness, code):
    client, proxy, upstreams = harness
    def fail(handler, body):
        handler.send_response(code)
        handler.end_headers()
    upstreams[0].respond = fail
    response = client.post("/v1/messages", json=payload())
    assert response.status_code == 200, response.text
    assert response.headers["X-Nexus-Routed-Model"] == "fallback:backup"
    health = load_provider_health(["primary"])["primary"]
    assert health.health_score == 0
    assert health.auth == "expired" if code in {401, 403} else health.rate_limit_cooldown_until
    again = client.post("/v1/messages", json=payload())
    assert again.headers["X-Nexus-Cache-Pinned"] == "true"
    assert len(upstreams[0].requests) == 1


def test_db_disabled_pinned_provider_and_degraded_provider_are_skipped(harness):
    client, proxy, upstreams = harness
    assert client.post("/v1/messages", json=payload()).status_code == 200
    with db._connect() as conn:
        conn.execute("UPDATE providers SET enabled=0 WHERE provider='primary'")
    response = client.post("/v1/messages", json=payload())
    assert response.headers["X-Nexus-Routed-Model"] == "fallback:backup"
    db.upsert_provider_health_state("fallback", {"auth": "ok", "quota": "low", "quota_remaining_ratio": .1, "health_score": .45})
    assert client.post("/v1/messages", json=payload()).status_code == 503


def test_connect_failure_and_response_timeout_trip_circuit(harness):
    client, proxy, upstreams = harness
    upstreams[0].respond = lambda handler, body: time.sleep(.3)
    proxy.config["read_timeout_seconds"] = .05
    response = client.post("/v1/messages", json=payload())
    assert response.status_code == 200, response.text
    assert load_provider_health(["primary"])["primary"].health_score == 0
    proxy.providers["fallback"]["base_url"] = "http://127.0.0.1:1"
    assert client.post("/v1/messages", json=payload()).status_code == 503
    assert load_provider_health(["fallback"])["fallback"].health_score == 0


def test_responses_continuation_fails_closed_when_pin_unhealthy(harness):
    client, proxy, upstreams = harness
    body = {"model": "cli", "input": "hi"}
    headers = {"x-session-id": "codex"}
    assert client.post("/v1/responses", json=body, headers=headers).status_code == 200
    body["previous_response_id"] = "reply1"
    assert client.post("/v1/responses", json=body, headers=headers).status_code == 200
    assert client.post("/v1/responses", json=body).status_code == 409
    with db._connect() as conn:
        conn.execute("UPDATE providers SET enabled=0 WHERE provider='primary'")
    assert client.post("/v1/responses", json=body, headers=headers).status_code == 503
    assert not upstreams[1].requests


def test_native_fallback_preferred_to_cross_protocol_for_main(harness):
    client, proxy, upstreams = harness
    proxy.providers["primary"]["protocols"] = ["chat"]
    response = client.post("/v1/messages", json=payload())
    assert response.headers["X-Nexus-Routed-Model"] == "fallback:backup"
    assert not upstreams[0].requests


def test_cross_protocol_housekeeping_real_http(harness):
    client, proxy, upstreams = harness
    proxy.providers["cheap"]["protocols"] = ["chat"]
    def chat_answer(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"id": "chat1", "choices": [{"message": {"content": "short summary"}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 23, "completion_tokens": 4}}).encode())
    upstreams[2].respond = chat_answer
    body = {"model": "cli", "system": "Summarize the conversation", "messages": [{"role": "user", "content": "transcript"}], "max_tokens": 100}
    response = client.post("/v1/messages", json=body, headers={"anthropic-beta": "prompt-caching"})
    assert response.status_code == 200, response.text
    assert response.json()["content"] == [{"type": "text", "text": "short summary"}]
    assert response.json()["usage"]["input_tokens"] == 23
    route, headers, received = upstreams[2].requests[0]
    assert route == "/v1/chat/completions"
    assert "anthropic-beta" not in headers
    assert received["messages"][0]["role"] == "system"
    assert not proxy.pins


@pytest.mark.parametrize("body", [[], {"messages": "bad"}, {"tools": [2]}, {"stream": "true"}])
def test_invalid_body_returns_400(harness, body):
    client, _, _ = harness
    assert client.post("/v1/messages", json=body).status_code == 400


@pytest.mark.parametrize("source,target", [("messages", "chat"), ("chat", "messages")])
def test_cross_protocol_sse_streaming_over_http(harness, source, target):
    client, proxy, upstreams = harness
    proxy.providers["cheap"]["protocols"] = [target]
    def stream(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        if target == "chat":
            events = [{"choices": [{"delta": {"content": "summary"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        else:
            events = [{"type": "message_start", "message": {"id": "m1", "usage": {"input_tokens": 10}}},
                      {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                      {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "summary"}},
                      {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
                      {"type": "message_stop"}]
        handler.wfile.write(("data: " + json.dumps(events[0]) + "\n\n").encode())
        handler.wfile.flush()
        handler.server.release.wait(3)
        for event in events[1:]:
            handler.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        if target == "chat":
            handler.wfile.write(b"data: [DONE]\n\n")
    upstreams[2].respond = stream
    upstreams[2].release.clear()
    body = {"model": "cli", "stream": True, "max_tokens": 4, "messages": [{"role": "user", "content": "ping"}]}
    path = "/v1/messages" if source == "messages" else "/v1/chat/completions"
    try:
        with client.stream("POST", path, json=body) as response:
            assert response.status_code == 200
            chunks = response.iter_bytes()
            first = next(chunks)
            assert b"message_start" in first if source == "messages" else b"chat.completion.chunk" in first
            assert not upstreams[2].release.is_set()
            upstreams[2].release.set()
            result = first + b"".join(chunks)
        assert b"summary" in result
        assert b"message_stop" in result if source == "messages" else b"[DONE]" in result
    finally:
        upstreams[2].release.set()


def test_bounded_pins_expire_and_background_does_not_extend_them(harness):
    client, proxy, _ = harness
    proxy.max_pins = 2
    for session in ["a", "b", "c"]:
        assert client.post("/v1/messages", json=payload(), headers={"x-session-id": session}).status_code == 200
    assert len(proxy.pins) == 2
    for pin in proxy.pins.values():
        pin.touched -= proxy.pin_ttl + 1
    assert not client.get("/v1/nexus/cli-status").json()["active_session_pins"]


def test_expired_db_cooldown_allows_recovery(harness):
    client, _, _ = harness
    db.upsert_provider_health_state("primary", {"auth": "unknown", "quota": "low", "health_score": 0,
        "recent_error_rate": .8, "rate_limit_cooldown_until": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
    response = client.post("/v1/messages", json=payload())
    assert response.headers["X-Nexus-Routed-Model"] == "primary:main"


def test_stream_disconnect_does_not_replay_request_on_fallback(harness):
    client, _, upstreams = harness
    def truncated(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", "1000")
        handler.end_headers()
        handler.wfile.write(b"data: partial\n\n")
    upstreams[0].respond = truncated
    with client.stream("POST", "/v1/messages", json={**payload(), "stream": True}) as response:
        assert b"data: partial" in b"".join(response.iter_bytes())
    assert not upstreams[1].requests
    assert load_provider_health(["primary"])["primary"].health_score == 0


def test_worker_routes_cheap_and_uses_only_configured_credentials(harness, monkeypatch):
    client, proxy, upstreams = harness
    monkeypatch.setenv("TEST_CHEAP_API_KEY", "cheap-secret")
    proxy.providers["cheap"]["api_key_env"] = "TEST_CHEAP_API_KEY"
    body = {"model": "claude-3-haiku-latest", "system": "You are a search subagent", "messages": []}
    response = client.post("/v1/messages", json=body, headers={"x-api-key": "client-secret", "cookie": "secret"})
    assert response.headers["X-Nexus-Call-Category"] == "SUBAGENT_WORKER"
    assert response.headers["X-Nexus-Routed-Model"] == "cheap:small"
    headers = {k.lower(): v for k, v in upstreams[2].requests[0][1].items()}
    assert headers["x-api-key"] == "cheap-secret"
    assert "cookie" not in headers
    assert not proxy.pins


def test_background_call_does_not_wait_for_main_stream(harness):
    client, _, upstreams = harness
    upstreams[0].release.clear()
    headers = {"x-session-id": "shared"}
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(client.post("/v1/messages", json={**payload(), "stream": True}, headers=headers)), daemon=True)
    thread.start()
    try:
        assert upstreams[0].first_chunk.wait(2)
        response = client.post("/v1/messages", json={"model": "cli", "max_tokens": 1}, headers=headers)
        assert response.status_code == 200
        assert response.headers["X-Nexus-Routed-Model"] == "cheap:small"
        assert not upstreams[0].release.is_set()
    finally:
        upstreams[0].release.set()
        thread.join(3)
    assert outcome[0].status_code == 200
