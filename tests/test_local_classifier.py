import json
from pathlib import Path

import pytest

from src.local_classifier import LocalRouteClassifier, classify_with_local_model
from src.server import RouterHandler
from src.types import PreSignals


class _MockWFile:
    def __init__(self):
        import io
        self._buf = io.BytesIO()

    def write(self, b: bytes):
        self._buf.write(b)


class _MockHandler:
    def __init__(self):
        self.wfile = _MockWFile()
        self.status = None

    def send_response(self, code: int):
        self.status = code

    def send_header(self, key: str, value: str):
        pass

    def end_headers(self):
        pass

    def call_route(self, body: dict):
        RouterHandler._handle_route(self, body)
        return self.status, json.loads(self.wfile._buf.getvalue())


@pytest.fixture(autouse=True)
def reset_local_classifier_singleton(monkeypatch):
    monkeypatch.setattr("src.local_classifier._LOCAL_CLASSIFIER", None)


def test_local_classifier_unavailable_without_artifact(tmp_path):
    classifier = LocalRouteClassifier(tmp_path / "missing")
    assert classifier.available is False
    assert "missing ONNX model" in (classifier.load_error or "")


def test_classify_with_local_model_respects_confidence_and_margin(monkeypatch):
    class FakeClassifier:
        load_error = None
        available = True

        def classify(self, message, pre_signals):
            from src.local_classifier import LocalClassifierResult
            from src.types import ClassifierOutput
            return LocalClassifierResult(
                classifier=ClassifierOutput(
                    task_type="reasoning",
                    complexity="medium",
                    confidence=0.84,
                    classifier_provider="local",
                    classifier_model="onnx:test",
                ),
                top_label="reasoning",
                confidence=0.84,
                margin=0.21,
                top2=[("reasoning", 0.84), ("general_chat", 0.63)],
                artifact_dir="/tmp/router-classifier",
            )

    monkeypatch.setattr("src.local_classifier.get_local_classifier", lambda: FakeClassifier())
    result = classify_with_local_model("compare these designs", PreSignals(message_length=24))
    assert result is not None
    assert result.classifier.task_type == "reasoning"
    assert result.classifier.classifier_provider == "local"


def test_classify_with_local_model_rejects_ambiguous_result(monkeypatch):
    class FakeClassifier:
        load_error = None
        available = True

        def classify(self, message, pre_signals):
            from src.local_classifier import LocalClassifierResult
            from src.types import ClassifierOutput
            return LocalClassifierResult(
                classifier=ClassifierOutput(task_type="general_chat", confidence=0.69),
                top_label="general_chat",
                confidence=0.69,
                margin=0.03,
                top2=[("general_chat", 0.69), ("reasoning", 0.66)],
                artifact_dir="/tmp/router-classifier",
            )

    monkeypatch.setattr("src.local_classifier.get_local_classifier", lambda: FakeClassifier())
    assert classify_with_local_model("maybe this maybe that", PreSignals(message_length=20)) is None


def test_classify_with_local_model_allows_rare_labels_with_per_class_threshold(monkeypatch):
    class FakeClassifier:
        load_error = None
        available = True

        def classify(self, message, pre_signals):
            from src.local_classifier import LocalClassifierResult
            from src.types import ClassifierOutput
            return LocalClassifierResult(
                classifier=ClassifierOutput(task_type="code_review", confidence=0.58),
                top_label="code_review",
                confidence=0.58,
                margin=0.03,
                top2=[("code_review", 0.58), ("coding", 0.55)],
                artifact_dir="/tmp/router-classifier",
            )

    monkeypatch.setattr("src.local_classifier.get_local_classifier", lambda: FakeClassifier())
    result = classify_with_local_model("Can you review this patch and list regressions?", PreSignals(message_length=42))
    assert result is not None
    assert result.classifier.task_type == "code_review"


def test_route_prefers_local_classifier_when_available(monkeypatch):
    from src.types import ClassifierOutput

    monkeypatch.setattr(
        "src.server.classify_with_local_model",
        lambda *_args, **_kwargs: type(
            "LocalResult",
            (),
            {
                "classifier": ClassifierOutput(
                    task_type="reasoning",
                    complexity="medium",
                    confidence=0.91,
                    classifier_provider="local",
                    classifier_model="onnx:/app/artifacts/router-classifier/onnx/model.onnx",
                ),
                "confidence": 0.91,
                "margin": 0.31,
                "top2": [("reasoning", 0.91), ("general_chat", 0.60)],
                "artifact_dir": "/app/artifacts/router-classifier/onnx",
            },
        )(),
    )
    monkeypatch.setattr("src.server.heuristic_classify", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.server.classify_with_model", lambda **_kwargs: None)

    status, body = _MockHandler().call_route({
        "message": "Compare these two architectures for throughput and operational risk.",
        "use_llm_classifier": True,
        "conversation_context": "We were discussing event-driven systems.",
    })
    assert status == 200
    assert body["classifier_source"] == "local"
    assert body["classifier_provider"] == "local"
    assert body["classifier_debug"]["local_confidence"] == 0.91


def test_route_falls_back_to_llm_when_local_unavailable(monkeypatch):
    from src.types import ClassifierOutput

    class FakeLocal:
        load_error = "missing ONNX model: /app/artifacts/router-classifier/onnx/model.onnx"

    monkeypatch.setattr("src.server.classify_with_local_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.server.get_local_classifier", lambda: FakeLocal())
    monkeypatch.setattr("src.server.heuristic_classify", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "src.server.classify_with_model",
        lambda **_kwargs: ClassifierOutput(task_type="reasoning", complexity="medium", confidence=0.82),
    )

    status, body = _MockHandler().call_route({
        "message": "Compare these two architectures for throughput and operational risk.",
        "use_llm_classifier": True,
        "conversation_context": "We were discussing event-driven systems.",
    })
    assert status == 200
    assert body["classifier_source"] == "llm"
    assert "local_unavailable" in body["classifier_debug"]
