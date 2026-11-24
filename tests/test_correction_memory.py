"""Correction memory: tasks given through feedback apply to similar messages at once."""

import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from src import correction_memory, db, server
from src.router import Router
from src.types import ClassifierOutput, PreSignals


@pytest.fixture
def router(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "router.sqlite")
    monkeypatch.setenv("NEXUS_ROUTER_CORRECTION_MEMORY", "on")
    db.ensure_schema()
    correction_memory.invalidate()
    return Router(persist=True)


def feedback(router, message, routed_task, verdict, corrected_task=None, surface="telegram_feedback_card"):
    decision = router.route(
        ClassifierOutput(task_type=routed_task, confidence=0.8), PreSignals(),
        message_text=message, source_type="raw-user", classifier_source="local",
    )
    feedback_id = db.record_feedback(decision.decision_id, verdict, corrected_task=corrected_task,
                                     source_surface=surface)
    correction_memory.invalidate()
    return feedback_id


def test_eligible_skips_short_and_agreeing_replies():
    assert not correction_memory.eligible("Hi")
    assert not correction_memory.eligible("fix it")
    assert not correction_memory.eligible("Yes go ahead")
    assert not correction_memory.eligible("Ok do that please")
    assert correction_memory.eligible("Please fix the failure")
    assert correction_memory.eligible("Yes, go ahead with the checks and the implementation please")


def test_a_task_the_user_picked_applies_to_similar_messages(router):
    picked = feedback(router, "Please fix this and the backup failure", "fast_utility", "wrong", "code_review")
    match = correction_memory.lookup("Please fix the backup failure")
    assert match.task_type == "code_review"
    assert match.support == 1 and match.share == 1.0 and match.similarity >= correction_memory.SIMILARITY
    assert match.feedback_ids == (picked,)
    assert correction_memory.lookup("Summarise the quarterly sales report for the board") is None


def test_confirmations_alone_never_change_the_task(router):
    feedback(router, "Please fix the backup failure tonight", "code_review", "correct")
    assert correction_memory.lookup("Please fix the backup failure") is None


def test_split_feedback_changes_nothing(router):
    feedback(router, "Please check the backup failure", "general_chat", "wrong", "code_review")
    feedback(router, "Please check the backup failures", "general_chat", "wrong", "reasoning")
    assert correction_memory.lookup("Please check the backup failure now") is None


def test_wrong_without_a_task_and_test_surfaces_are_ignored(router):
    feedback(router, "Please fix the backup failure", "general_chat", "wrong")
    feedback(router, "Please fix the backup failures", "general_chat", "wrong", "code_review", surface="test")
    assert db.load_labeled_feedback() == []
    assert correction_memory.lookup("Please fix the backup failure") is None


def test_memory_can_be_switched_off(router, monkeypatch):
    feedback(router, "Please fix this and the backup failure", "fast_utility", "wrong", "code_review")
    monkeypatch.setenv("NEXUS_ROUTER_CORRECTION_MEMORY", "off")
    assert correction_memory.lookup("Please fix the backup failure") is None
    assert correction_memory.summary() == {"enabled": False, "examples": 0, "user_picked": 0}


@pytest.fixture
def http(router, monkeypatch):
    # Every message classifies as low-confidence general chat, so only the memory can change it.
    monkeypatch.setattr(server, "_router", router)
    monkeypatch.setattr(server, "classify_with_local_model", lambda *a, **k: None)
    monkeypatch.setattr(server, "classify_with_jev", lambda *a, **k: None)
    monkeypatch.setattr(server, "heuristic_classify",
                        lambda *a, **k: ClassifierOutput(task_type="general_chat", confidence=0.4))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.RouterHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://127.0.0.1:{httpd.server_port}", timeout=10, trust_env=False) as client:
        yield client
    httpd.shutdown()
    httpd.server_close()


def test_route_applies_feedback_at_once_and_records_why(http, router):
    message = "Please fix the backup failure"
    before = http.post("/route", json={"message": message}).json()
    assert before["task_type"] == "general_chat"
    response = http.post("/feedback", json={"decision_id": before["decision_id"], "verdict": "wrong",
                                            "corrected_task": "code_review"})
    assert response.status_code == 200

    after = http.post("/route", json={"message": "Please fix the backup failures"}).json()
    assert after["task_type"] == "code_review"
    assert after["classifier_source"] == "memory"
    # Above the confidence gate, which would otherwise route it as general_chat again.
    assert after["confidence"] >= correction_memory.CONFIDENCE
    memory = after["explain"]["classifier"]["correction_memory"]
    assert memory["from_task"] == "general_chat" and memory["from_source"] == "heuristic"
    assert "correction_memory" in after["explain"]["rules_triggered"]
    assert any(r.startswith("correction memory: general_chat -> code_review") for r in after["reason"])
    stored = http.get("/route-info", params={"id": after["decision_id"]}).json()
    assert stored["classifier_source"] == "memory" and stored["task_type"] == "code_review"

    # Caller-given tasks, forced route modes and attachments keep their own task.
    explicit = http.post("/route", json={"message": "Please fix the backup failures",
                                         "classifier": {"task_type": "coding", "confidence": 0.9}}).json()
    assert explicit["task_type"] == "coding" and explicit["classifier_source"] == "explicit"
    forced = http.post("/route", json={"message": "Please fix the backup failures", "route_mode": "fast"}).json()
    assert forced["classifier_source"] == "heuristic"
    image = http.post("/route", json={"message": "Please fix the backup failures", "has_image": True}).json()
    assert image["classifier_source"] == "heuristic"

    diagnostics = http.get("/diagnostics").json()["correction_memory"]
    assert diagnostics == {"enabled": True, "examples": 1, "user_picked": 1, "applied_7d": 1}


def test_a_broken_memory_never_breaks_routing(http, monkeypatch):
    def broken(message):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(correction_memory, "lookup", broken)
    response = http.post("/route", json={"message": "Please fix the backup failures"})
    assert response.status_code == 200
    assert response.json()["classifier_source"] == "heuristic"
