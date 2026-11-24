"""OpenClaw plugin observations and the /diagnostics alerts built from them."""

import threading
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import httpx
import pytest

from src import db, diagnostics, server
from src.router import Router


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    db.ensure_schema()


def events(*items):
    for kind, ok, *detail in items:
        db.record_plugin_event(kind, ok, "telegram", detail[0] if detail else None)


def plugin(now=None):
    return diagnostics._openclaw_plugin(now or datetime.now(timezone.utc))


def test_two_uncaptured_messages_in_a_row_raise_a_restart_alert(state):
    events(("message", True), ("message", False, "uncaptured"))
    assert plugin()["alerts"] == []
    events(("message", False, "uncaptured"))
    summary = plugin()
    [alert] = summary["alerts"]
    assert alert["code"] == "message" and alert["count"] == 2
    assert "systemctl --user restart openclaw-gateway" in alert["message"]
    assert summary["message"]["failure_streak"] == 2 and summary["message"]["ok_24h"] == 1
    assert alert["since"] == summary["message"]["failing_since"]
    # One captured message ends the incident.
    events(("message", True))
    assert plugin()["alerts"] == []


def test_card_and_route_failures_need_three_in_a_row(state):
    events(("card", False, "bridge_http_502"), ("card", False, "bridge_http_502"),
           ("route", False, "timeout"), ("route", False, "timeout"))
    assert plugin()["alerts"] == []
    events(("card", False, "bridge_http_502"), ("route", False, "timeout"))
    codes = {a["code"]: a for a in plugin()["alerts"]}
    assert set(codes) == {"card", "route"}
    assert "bridge_http_502" in codes["card"]["message"] and "timeout" in codes["route"]["message"]


def test_only_the_last_day_counts_and_old_rows_are_pruned(state):
    events(("message", False, "uncaptured"), ("message", False, "uncaptured"))
    assert plugin(datetime.now(timezone.utc) + timedelta(days=2))["alerts"] == []
    with db._connect() as conn:
        conn.execute("UPDATE plugin_events SET created_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(days=15)).isoformat(),))
    events(("message", True))
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM plugin_events").fetchone()[0] == 1


@pytest.fixture
def http(state, monkeypatch):
    monkeypatch.setattr(server, "_router", Router(persist=True))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.RouterHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=10, trust_env=False) as client:
        yield client
    httpd.shutdown()
    httpd.server_close()


def test_http_plugin_events_validate_and_surface_in_diagnostics(http):
    for bad in ({"kind": "message"}, {"kind": "other", "ok": True}, {"kind": "message", "ok": "yes"},
                {"kind": "card", "ok": False, "detail": "Bridge said: no"}, {"kind": "card", "ok": False, "channel": 5}):
        assert http.post("/plugin-events", json=bad).status_code == 400, bad
    for _ in range(2):
        assert http.post("/plugin-events", json={"kind": "message", "ok": False, "channel": "telegram",
                                                 "detail": "uncaptured"}).json() == {"ok": True}
    data = http.get("/diagnostics").json()
    assert data["openclaw_plugin"]["alerts"][0]["code"] == "message"
    assert data["openclaw_plugin"]["alerts"][0]["message"] in data["warnings"]
