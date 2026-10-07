"""Jev classifier backend: typed answers in, fall-through on doubt."""

import io
import json
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

from src.jev_classifier import classify_with_jev
from src.types import PreSignals


def _signals():
    return PreSignals(has_image=False)


def _fake_urlopen(payload):
    response = MagicMock()
    response.read = MagicMock(return_value=json.dumps(payload).encode())
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


def _answers(choice="coding", confidence=0.9, complexity="medium", tools=0.8):
    return {
        "answers": {
            "task_type": {
                "choice": choice,
                "probabilities": {choice: confidence, "general_chat": 1 - confidence},
                "confidence": confidence,
            },
            "complexity": {"choice": complexity, "probabilities": {}, "confidence": 0.7},
            "needs_tools": {"noul": tools},
        }
    }


def test_confident_choice_maps_to_classifier(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("JEV_ENABLED", raising=False)
    with patch("urllib.request.urlopen", return_value=_fake_urlopen(_answers())):
        result = classify_with_jev("fix this bug", _signals())
    assert result is not None
    assert result.classifier.task_type == "coding"
    assert result.classifier.complexity == "medium"
    assert result.classifier.needs_tools is True
    assert result.classifier.classifier_provider == "jev"
    assert result.confidence == 0.9


def test_low_confidence_falls_through(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch(
        "urllib.request.urlopen",
        return_value=_fake_urlopen(_answers(confidence=0.2)),
    ):
        assert classify_with_jev("hmm", _signals()) is None


def test_missing_key_and_disabled_skip(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert classify_with_jev("hi", _signals()) is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("JEV_ENABLED", "false")
    assert classify_with_jev("hi", _signals()) is None


def test_backend_selects_free_opencode(monkeypatch):
    import sys

    sys.path.insert(0, ".")
    from src.jev_classifier import _jev_backend

    monkeypatch.delenv("JEV_BACKEND", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-test")
    assert _jev_backend() == (
        "https://opencode.ai/zen/v1/systemone",
        "oc-test",
        "jev-1.13-free",
    )


def test_posts_to_systemone_endpoint(monkeypatch):
    import sys

    sys.path.insert(0, ".")
    from src import jev_classifier

    monkeypatch.setenv("OPENCODE_API_KEY", "oc-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = {}

    class _Resp:
        def read(self):
            import json as _json

            return _json.dumps({"answers": {}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["model"] = __import__("json").loads(req.data)["model"]
        seen["ua"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    jev_classifier.classify_with_jev("hi", _signals())
    assert seen["url"] == "https://opencode.ai/zen/v1/systemone"
    assert seen["model"] == "jev-1.13-free"
    assert seen["ua"] and "Python-urllib" not in seen["ua"]
