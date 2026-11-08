"""Explainability, provider filter, A/B experiments, per-model cooldowns, diagnostics."""

import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from src import cooldowns, db, server
from src.router import Router, _experiment_routing
from src.types import ClassifierOutput, PreSignals

CODING = ClassifierOutput(task_type="coding", complexity="medium", confidence=0.9, needs_tools=True)


@pytest.fixture
def router(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    return Router(persist=True)


def test_allowed_providers_restricts_selection_and_fallbacks(router):
    decision = router.route(CODING, PreSignals(), allowed_providers=["nvidia"])
    assert decision.selected_provider == "nvidia"
    assert all(model.startswith("nvidia/") for model in decision.fallbacks)
    assert "provider_filter" in decision.explain["rules_triggered"]
    assert decision.explain["allowed_providers"] == ["nvidia"]


def test_allowed_providers_without_models_raises_no_eligible(router):
    with pytest.raises(RuntimeError, match="No eligible models"):
        router.route(CODING, PreSignals(), allowed_providers=["no-such-provider"])


def test_explain_is_persisted_and_readable_without_message_text(router):
    decision = router.route(CODING, PreSignals(), message_text="secret prompt", classifier_source="explicit")
    explain = decision.explain
    assert explain["classifier"]["source"] == "explicit"
    assert explain["classifier"]["effective_task_type"] == decision.task_type
    assert explain["model_scores"][0]["model"] == decision.selected_model
    assert len(explain["model_scores"]) <= 5
    info = db.load_decision_info(decision.decision_id)
    assert info["explain"] == explain
    assert "message_text" not in info and info["feedback"] == []
    assert db.load_decision_info("missing") is None


def test_experiment_assignment_is_stable_and_merges_scoring():
    routing = {"scoring": {"exploration": {"enabled": True, "epsilon": 0.03}},
               "experiment": {"id": "exp1", "split": 0.5,
                              "variants": {"control": {}, "candidate": {"exploration": {"enabled": False}}}}}
    first = _experiment_routing(routing, "conversation-1")
    assert first[0] == _experiment_routing(routing, "conversation-1")[0]
    assert _experiment_routing({**routing, "experiment": {**routing["experiment"], "split": 0}}, "k")[0] == "exp1:control"
    policy, merged = _experiment_routing({**routing, "experiment": {**routing["experiment"], "split": 1}}, "k")
    assert policy == "exp1:candidate"
    assert merged["scoring"]["exploration"] == {"enabled": False, "epsilon": 0.03}
    assert routing["scoring"]["exploration"]["enabled"] is True  # base config untouched
    assert _experiment_routing({"scoring": {}}, "k") == (None, {"scoring": {}})
    keys = [f"c{i}" for i in range(400)]
    share = sum(_experiment_routing(routing, k)[0] == "exp1:candidate" for k in keys) / len(keys)
    assert 0.4 < share < 0.6


def test_experiment_policy_id_is_persisted(router):
    router._routing = {**router._routing, "experiment": {"id": "exp2", "split": 1,
                                                          "variants": {"control": {}, "candidate": {}}}}
    decision = router.route(CODING, PreSignals(), experiment_key="conv")
    assert decision.policy_id == "exp2:candidate"
    assert db.load_decision_info(decision.decision_id)["policy_id"] == "exp2:candidate"


def test_cooldowns_open_after_repeated_failures_and_reset_on_success(router):
    cfg = cooldowns.config()
    for _ in range(int(cfg["failures"]) - 1):
        assert cooldowns.record_failure("p/m", "coding") is False
    assert cooldowns.record_failure("p/m", "coding") is True
    assert cooldowns.cooling_models("coding") == {"p/m"}
    assert cooldowns.cooling_models("reasoning") == set()
    cooldowns.record_success("p/m", "coding")
    assert cooldowns.cooling_models("coding") == set()


def test_cooling_model_is_excluded_from_scoring(router):
    first = router.route(CODING, PreSignals())
    for _ in range(int(cooldowns.config()["failures"])):
        cooldowns.record_failure(first.selected_model, first.task_type)
    second = router.route(CODING, PreSignals())
    assert second.selected_model != first.selected_model
    assert second.explain["exclusions"].get("model_cooldown") == 1


@pytest.fixture
def http(router, monkeypatch):
    monkeypatch.setattr(server, "_router", router)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.RouterHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=10, trust_env=False) as client:
        yield client
    httpd.shutdown()
    httpd.server_close()


def explicit_route(client, **extra):
    body = {"message": "fix the failing parser test", "classifier": {"task_type": "coding", "confidence": 0.9}, **extra}
    return client.post("/route", json=body)


def test_http_route_explain_route_info_and_outcome_cooldown(http):
    response = explicit_route(http, allowed_providers=["openai-codex"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["selected_provider"] == "openai-codex" and data["explain"]["allowed_providers"] == ["openai-codex"]
    info = http.get("/route-info", params={"id": data["decision_id"]}).json()
    assert info["selected_model"] == data["selected_model"] and info["outcome_success"] is None
    for _ in range(int(cooldowns.config()["failures"])):
        assert http.post("/outcome", json={"decision_id": data["decision_id"], "success": False}).status_code == 200
    assert data["selected_model"] in cooldowns.cooling_models("coding")
    assert http.get("/route-info", params={"id": data["decision_id"]}).json()["outcome_success"] == 0
    assert http.get("/route-info", params={"id": "nope"}).status_code == 404
    assert http.get("/route-info").status_code == 400


def test_http_route_rejects_bad_inputs_and_reports_no_eligible(http):
    assert explicit_route(http, allowed_providers="nvidia").status_code == 400
    assert explicit_route(http, experiment_key=5).status_code == 400
    conflict = explicit_route(http, allowed_providers=["no-such-provider"])
    assert conflict.status_code == 409 and "trace" not in conflict.json()


def test_http_diagnostics_snapshot(http):
    cooldowns.record_failure("x/y", "coding")
    data = http.get("/diagnostics").json()
    for key in ("warnings", "registry", "classifier", "database", "policies", "providers", "model_cooldowns", "cli_proxy"):
        assert key in data
    assert data["registry"]["models"] > 0
    assert data["policies"]["routing.yaml"] == "ok"
    assert data["database"]["schema_version"] and "error" not in data["database"]
    assert "MAIN_EXECUTION" in data["cli_proxy"]["routes"]


def test_policy_experiment_report_compares_variants(router):
    import sqlite3

    from scripts.policy_experiment_report import report

    router._routing = {**router._routing, "experiment": {"id": "rep", "split": 0.5,
                                                          "variants": {"control": {}, "candidate": {}}}}
    ids = {}
    for key in ("a", "b", "c", "d", "e", "f", "g", "h"):
        decision = router.route(CODING, PreSignals(), experiment_key=key)
        ids.setdefault(decision.policy_id, decision.decision_id)
        db.update_outcome(decision_id=decision.decision_id, success=decision.policy_id.endswith("candidate"), latency_ms=100)
    assert set(ids) == {"rep:control", "rep:candidate"}
    db.record_feedback(decision_id=ids["rep:candidate"], verdict="correct")
    with sqlite3.connect(db.DB_PATH) as conn:
        rows = {r["policy_id"]: r for r in report(conn, "rep", 1)}
    assert rows["rep:candidate"]["success_rate"] == 1.0 and rows["rep:control"]["success_rate"] == 0.0
    assert rows["rep:candidate"]["feedback_correct_rate"] == 1.0
    assert rows["rep:control"]["feedback"] == 0
    with sqlite3.connect(db.DB_PATH) as conn:
        assert report(conn, "other", 1) == []
