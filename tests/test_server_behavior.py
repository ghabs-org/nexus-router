from src.server import RouterHandler, _is_tiny_prompt, should_reclassify_with_llm
from src.types import ClassifierOutput


def test_is_tiny_prompt_flags_short_messages():
    assert _is_tiny_prompt("hi") is True
    assert _is_tiny_prompt("hello there") is True
    assert _is_tiny_prompt("Did you read this?") is True
    assert _is_tiny_prompt("please summarize this long article for me") is False


def test_should_not_reclassify_tiny_prompt_with_llm():
    classifier = ClassifierOutput(task_type="general_chat", confidence=0.6)
    assert should_reclassify_with_llm(classifier, True, "prior context", "hi") is False


def test_should_not_reclassify_short_followup_with_llm():
    classifier = ClassifierOutput(task_type="general_chat", confidence=0.6)
    assert should_reclassify_with_llm(
        classifier,
        True,
        "prior context about router timeouts",
        "Did you read this?",
    ) is False


def test_should_reclassify_non_tiny_general_chat_when_enabled():
    classifier = ClassifierOutput(task_type="general_chat", confidence=0.6)
    assert should_reclassify_with_llm(
        classifier,
        True,
        "prior context",
        "Can you help me compare these two approaches?",
    ) is True


def test_handle_outcome_forwards_quota_signals(monkeypatch):
    observed = {}

    def fake_update_outcome(**kwargs):
        observed["update"] = kwargs

    def fake_observe_turn_outcome(**kwargs):
        observed["observe"] = kwargs

    class _Handler:
        def __init__(self):
            self.responses = []

    handler = _Handler()

    monkeypatch.setattr("src.server.update_outcome", fake_update_outcome)
    monkeypatch.setattr("src.server.observe_turn_outcome", fake_observe_turn_outcome)
    monkeypatch.setattr("src.server._json_response", lambda _handler, status, data: observed.setdefault("response", (status, data)))

    RouterHandler._handle_outcome(handler, {
        "decision_id": "dec-123",
        "success": False,
        "latency_ms": 3210,
        "provider": "google-gemini-cli",
        "http_status": 429,
        "error_type": "rate_limit",
        "quota_hint": "exhausted",
        "quota_remaining_ratio": 0,
    })

    assert observed["update"]["decision_id"] == "dec-123"
    assert observed["observe"]["provider"] == "google-gemini-cli"
    assert observed["observe"]["http_status"] == 429
    assert observed["observe"]["error_type"] == "rate_limit"
    assert observed["observe"]["quota_hint"] == "exhausted"
    assert observed["observe"]["quota_remaining_ratio"] == 0
    assert observed["response"] == (200, {"ok": True})



def test_handle_feedback_records_structured_feedback(monkeypatch):
    observed = {}

    def fake_record_feedback(**kwargs):
        observed["feedback"] = kwargs
        return "fb-123"

    class _Handler:
        pass

    handler = _Handler()
    monkeypatch.setattr("src.server.record_feedback", fake_record_feedback)
    monkeypatch.setattr("src.server._json_response", lambda _handler, status, data: observed.setdefault("response", (status, data)))

    RouterHandler._handle_feedback(handler, {
        "decision_id": "dec-123",
        "verdict": "wrong",
        "corrected_task": "reasoning",
        "model_verdict": "good",
        "preferred_model": "github-copilot/claude-sonnet-4.6",
        "reason_tag": "quality",
        "source_surface": "telegram",
        "source_channel": "direct",
        "metadata": {"raw": "wrong -> reasoning"},
    })

    assert observed["feedback"]["decision_id"] == "dec-123"
    assert observed["feedback"]["preferred_model"] == "github-copilot/claude-sonnet-4.6"
    assert observed["feedback"]["source_surface"] == "telegram"
    assert observed["response"] == (200, {"ok": True, "feedback_id": "fb-123"})


def test_handle_feedback_rejects_invalid_verdict(monkeypatch):
    observed = {}
    class _Handler:
        pass
    handler = _Handler()
    monkeypatch.setattr("src.server._json_response", lambda _handler, status, data: observed.setdefault("response", (status, data)))
    RouterHandler._handle_feedback(handler, {"decision_id": "dec-123", "verdict": "maybe"})
    assert observed["response"] == (400, {"error": "verdict must be correct|wrong"})
