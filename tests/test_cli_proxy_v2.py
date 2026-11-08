"""Local-mode upgrades: persisted pins/turns, usage capture, /v1/models, count_tokens,
keyless subscription backends, per-model cooldowns and learned ordering.

Reuses the real-HTTP harness from test_cli_proxy (real upstreams, real SQLite).
"""

import json
import time

import pytest

from src import cooldowns, db, server
from src.cli_classifier import CallCategory
from src.cli_proxy import CliProxy, Target, UsageTap
from test_cli_proxy import harness, payload  # noqa: F401  (fixture re-export)

MAIN = CallCategory.MAIN_EXECUTION


def turns():
    with db._connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM cli_proxy_turns ORDER BY id")]


def test_pins_and_turns_survive_restart(harness, monkeypatch):
    client, proxy, _ = harness
    headers = {"x-session-id": "persist-me"}
    assert client.post("/v1/messages", json=payload(), headers=headers).status_code == 200
    restarted = CliProxy(proxy.config)
    assert [p.target.label for p in restarted.pins.values()] == ["primary:main"]
    monkeypatch.setattr(server, "_cli_proxy", restarted)
    again = client.post("/v1/messages", json=payload(), headers=headers)
    assert again.headers["X-Nexus-Cache-Pinned"] == "true"
    status = client.get("/v1/nexus/cli-status").json()
    assert [d["target"] for d in status["recent_decisions"]] == ["primary:main", "primary:main"]
    assert status["recent_decisions"][0]["cache_pinned"] is True
    assert status["usage_24h"][0]["turns"] == 2


def test_restart_drops_pins_for_targets_no_longer_routed(harness):
    client, proxy, _ = harness
    assert client.post("/v1/messages", json=payload(), headers={"x-session-id": "s"}).status_code == 200
    config = {**proxy.config, "routes": {**proxy.config["routes"], "MAIN_EXECUTION": ["fallback:backup"]}}
    assert not CliProxy(config).pins


def test_expired_persisted_pins_are_not_loaded(harness):
    client, proxy, _ = harness
    assert client.post("/v1/messages", json=payload(), headers={"x-session-id": "old"}).status_code == 200
    with db._connect() as conn:
        conn.execute("UPDATE cli_proxy_pins SET touched_at = touched_at - ?", (proxy.pin_ttl + 5,))
    assert not CliProxy(proxy.config).pins


def test_streamed_usage_is_recorded_across_split_lines(harness):
    client, _, upstreams = harness
    def stream(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        start = 'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":1}}}\n\n'
        handler.wfile.write(start[:40].encode())  # split mid-line
        handler.wfile.flush()
        time.sleep(.05)
        handler.wfile.write(start[40:].encode())
        handler.wfile.write(b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\ndata: {"type":"message_stop"}\n\n')
    upstreams[0].respond = stream
    with client.stream("POST", "/v1/messages", json={**payload(), "stream": True}) as response:
        assert b"message_stop" in b"".join(response.iter_bytes())
    row = turns()[-1]
    assert (row["input_tokens"], row["output_tokens"], row["http_status"]) == (15, 7, 200)


def test_non_streamed_usage_is_recorded(harness):
    client, _, upstreams = harness
    def answer(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"id": "r", "usage": {"input_tokens": 3, "output_tokens": 2}}).encode())
    upstreams[0].respond = answer
    assert client.post("/v1/messages", json=payload()).status_code == 200
    assert (turns()[-1]["input_tokens"], turns()[-1]["output_tokens"]) == (3, 2)


def test_usage_tap_reads_chat_and_responses_formats():
    chat = UsageTap(True)
    chat.feed(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":8,')
    chat.feed(b'"completion_tokens":4}}\n\ndata: [DONE]\n\n')
    assert chat.finish() == {"input_tokens": 8, "output_tokens": 4}
    responses = UsageTap(True)
    responses.feed(b'event: response.completed\ndata: {"type":"response.completed","response":{"usage":{"input_tokens":9,"output_tokens":1}}}')
    assert responses.finish() == {"input_tokens": 9, "output_tokens": 1}  # last line without trailing newline
    big = UsageTap(False)
    big.feed(b"x" * (UsageTap.MAX_BODY + 1))
    assert big.finish() == {}


def test_models_lists_configured_route_models(harness):
    client, _, _ = harness
    data = client.get("/v1/models").json()
    assert data["object"] == "list" and data["has_more"] is False
    assert [m["id"] for m in data["data"]] == ["small", "planner", "main", "backup"] or \
        {m["id"] for m in data["data"]} == {"main", "backup", "planner", "small"}
    assert all(m["type"] == "model" and m["object"] == "model" for m in data["data"])


def test_count_tokens_forwards_to_messages_native_target(harness):
    client, proxy, upstreams = harness
    def counted(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"input_tokens": 42}).encode())
    upstreams[0].respond = counted
    response = client.post("/v1/messages/count_tokens", json=payload(), headers={"Authorization": "Bearer client"})
    assert response.status_code == 200 and response.json() == {"input_tokens": 42}
    assert response.headers["X-Nexus-Routed-Model"] == "primary:main"
    path, headers, body = upstreams[0].requests[-1]
    assert path == "/v1/messages/count_tokens" and body["model"] == "main"
    assert "authorization" not in {k.lower() for k in headers}
    assert not turns()  # token counting is not a turn
    for name in proxy.providers:
        proxy.providers[name]["protocols"] = ["chat"]
    assert client.post("/v1/messages/count_tokens", json=payload()).status_code == 404


def test_keyless_chat_provider_serves_as_subscription_backend(harness):
    client, proxy, upstreams = harness
    config = {**proxy.config,
              "providers": {**proxy.config["providers"],
                            "subscription": {"base_url": proxy.config["providers"]["cheap"]["base_url"], "protocols": ["chat"]}},
              "routes": {**proxy.config["routes"], "STATELESS_HOUSEKEEPING": ["subscription:cc:sonnet"]}}
    sub = CliProxy(config)
    assert sub.routes[CallCategory.STATELESS_HOUSEKEEPING] == [Target("subscription", "cc:sonnet")]
    db.upsert_provider_health_state("subscription", {"auth": "ok", "quota": "healthy", "health_score": 1})
    with db._connect() as conn:
        conn.execute("INSERT INTO providers (provider, enabled, status) VALUES ('subscription', 1, 'enabled') "
                     "ON CONFLICT(provider) DO UPDATE SET enabled=1, status='enabled'")
    def chat(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"id": "c", "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode())
    upstreams[2].respond = chat
    server._cli_proxy = sub
    body = {"model": "cli", "system": "Summarize the conversation", "messages": [{"role": "user", "content": "t"}], "max_tokens": 50}
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["X-Nexus-Routed-Model"] == "subscription:cc:sonnet"
    path, headers, received = upstreams[2].requests[-1]
    assert path == "/v1/chat/completions" and received["model"] == "cc:sonnet"
    assert not {"authorization", "x-api-key"} & {k.lower() for k in headers}
    assert turns()[-1]["converted"] == 1


def test_cooling_target_is_skipped_and_5xx_counts_against_the_model(harness):
    client, proxy, upstreams = harness
    for _ in range(int(cooldowns.config()["failures"])):
        cooldowns.record_failure("primary:main", MAIN.value)
    response = client.post("/v1/messages", json=payload())
    assert response.headers["X-Nexus-Routed-Model"] == "fallback:backup"
    status = client.get("/v1/nexus/cli-status").json()
    assert [c["model"] for c in status["model_cooldowns"]] == ["primary:main"]
    def fail(handler, body):
        handler.send_response(500)
        handler.end_headers()
    upstreams[1].respond = fail
    client.post("/v1/messages", json=payload(), headers={"x-session-id": "other"})
    with db._connect() as conn:
        row = conn.execute("SELECT failures FROM model_cooldowns WHERE model='fallback:backup'").fetchone()
    assert row["failures"] == 1


def record(target, status, n):
    with db._connect() as conn:
        for _ in range(n):
            conn.execute("INSERT INTO cli_proxy_turns (created_at, session, category, target, http_status, latency_ms) "
                         "VALUES (?, 's', 'MAIN_EXECUTION', ?, ?, 100)", (time.time(), target, status))


def test_learned_ordering_needs_the_margin_and_keeps_pins_first(harness):
    client, proxy, _ = harness
    assert client.post("/v1/messages", json=payload(), headers={"x-session-id": "pinned"}).status_code == 200
    record("primary:main", 500, 8)
    record("fallback:backup", 200, 8)
    assert [t.label for t in proxy._ordered(MAIN)[0]] == ["primary:main", "fallback:backup"]  # static default
    proxy.config["ordering"] = "learned"
    order, scores = proxy._ordered(MAIN)
    assert [t.label for t in order] == ["fallback:backup", "primary:main"]
    assert scores[Target("fallback", "backup")] > scores[Target("primary", "main")]
    proxy.config["learned_margin"] = 1.0
    assert [t.label for t in proxy._ordered(MAIN)[0]] == ["primary:main", "fallback:backup"]
    proxy.config["learned_margin"] = 0.15
    fresh = client.post("/v1/messages", json=payload(), headers={"x-session-id": "new"})
    assert fresh.headers["X-Nexus-Routed-Model"] == "fallback:backup"
    pinned = client.post("/v1/messages", json=payload(), headers={"x-session-id": "pinned"})
    assert pinned.headers["X-Nexus-Routed-Model"] == "primary:main"
    status = client.get("/v1/nexus/cli-status").json()
    assert status["ordering"] == "learned"
    assert status["applied_order"]["MAIN_EXECUTION"][0]["target"] == "fallback:backup"


def test_client_hangups_do_not_count_against_a_target(harness):
    _, proxy, _ = harness
    record("primary:main", 499, 20)
    record("fallback:backup", 200, 1)
    scores = proxy._target_scores(proxy.routes[MAIN])
    assert scores[Target("primary", "main")] == 0.5  # same as a target with no history


def test_invalid_ordering_is_rejected(harness):
    _, proxy, _ = harness
    with pytest.raises(ValueError, match="ordering"):
        CliProxy({**proxy.config, "ordering": "random"})
