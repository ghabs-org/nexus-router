"""
Tests for nexus-router: scorer and router.
Run with: python -m pytest tests/ -v
"""

import json
import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import ClassifierOutput, PreSignals, ProviderHealth
from src.scorer import score_models, _preference_score, _learned_score
from src.router import Router
from src.classifier import extract_pre_signals, heuristic_classify, parse_classifier_response


# ── Fixtures ──────────────────────────────────────────────────────────────────

REGISTRY_FILE = Path(__file__).parent.parent / "catalog/normalized/models.json"


@pytest.fixture(autouse=True, scope="session")
def ensure_full_registry():
    """Regenerate registry with all configured providers before running tests."""
    import subprocess
    subprocess.run(
        [sys.executable, "src/generate_registry.py", "--only-configured"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
    )


@pytest.fixture
def registry():
    if not REGISTRY_FILE.exists():
        pytest.skip("Registry not generated. Run: python src/generate_registry.py")
    with open(REGISTRY_FILE) as f:
        data = json.load(f)
    return data["models"]


@pytest.fixture
def authed_models(registry):
    return [m for m in registry if m["availability"]["authed"]]


@pytest.fixture
def provider_health_ok():
    return {
        "openai-codex": ProviderHealth(provider="openai-codex", auth="ok", quota="healthy", health_score=0.95),
        "github-copilot": ProviderHealth(provider="github-copilot", auth="ok", quota="healthy", health_score=0.95),
        "google-gemini-cli": ProviderHealth(provider="google-gemini-cli", auth="ok", quota="healthy", health_score=0.95),
    }


@pytest.fixture
def router():
    return Router(persist=False)


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorer:

    def test_coding_task_prefers_codex(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(task_type="coding", complexity="high", confidence=0.90)
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible, "Should have at least one eligible model"
        assert "openai-codex" in eligible[0].provider or "codex" in eligible[0].model_id.lower()

    def test_code_review_prefers_claude(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(task_type="code_review", complexity="medium", confidence=0.85)
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible
        # Top model should be a Claude variant
        assert "claude" in eligible[0].model_id.lower(), \
            f"Expected Claude for code review, got {eligible[0].model_id}"

    def test_vision_excludes_text_only_models(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(task_type="vision", needs_vision=True, confidence=0.90)
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        # grok-code-fast-1 is text-only — should be excluded
        grok = next((s for s in scored if "grok" in s.model_id), None)
        if grok:
            assert grok.excluded, "Text-only model should be excluded from vision task"

    def test_long_context_excludes_small_context_models(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(
            task_type="long_context", needs_long_context=True, confidence=0.85
        )
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        # Models with small context windows should be excluded
        excluded = [s for s in scored if s.excluded and "context_too_small" in (s.exclusion_reason or "")]
        assert excluded, "Should exclude models with insufficient context window"

    def test_degraded_provider_excluded(self, authed_models):
        degraded_health = {
            "openai-codex": ProviderHealth(provider="openai-codex", auth="expired", health_score=0.0),
            "github-copilot": ProviderHealth(provider="github-copilot", auth="ok", quota="healthy", health_score=0.95),
            "google-gemini-cli": ProviderHealth(provider="google-gemini-cli", auth="ok", quota="healthy", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, authed_models, degraded_health, {})
        eligible = [s for s in scored if not s.excluded]
        codex_eligible = [s for s in eligible if s.provider == "openai-codex"]
        assert not codex_eligible, "Degraded provider should have no eligible models"

    def test_fast_utility_cheap_profile(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(
            task_type="fast_utility", cost_profile="cheap", complexity="low", confidence=0.80
        )
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible
        # Top model should score high on speed/cost
        top = eligible[0]
        assert top.speed > 0.70 or top.cost > 0.75, \
            f"Expected fast/cheap model at top for fast_utility+cheap, got {top.model_id}"

    def test_all_authed_models_scored(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(task_type="general_chat", confidence=0.75)
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        # All input models should appear in scored (eligible or excluded)
        assert len(scored) == len(authed_models)

    def test_scores_in_range(self, authed_models, provider_health_ok):
        classifier = ClassifierOutput(task_type="reasoning", confidence=0.80)
        scored = score_models(classifier, authed_models, provider_health_ok, {})
        for s in scored:
            if not s.excluded:
                assert 0.0 <= s.total_score <= 1.0, f"Score out of range for {s.model_id}: {s.total_score}"

    def test_preference_score_ordering(self):
        order = ["model-a", "model-b", "model-c"]
        assert _preference_score("model-a", order) > _preference_score("model-b", order)
        assert _preference_score("model-b", order) > _preference_score("model-c", order)
        assert _preference_score("unknown-model", order) == 0.50

    def test_learned_score_no_data(self):
        score = _learned_score("openai-codex/gpt-5.4", "coding", {})
        assert score == 0.75  # neutral prior

    def test_learned_score_high_success(self):
        stats = {"openai-codex/gpt-5.4": {
            "total_selected": 100, "total_success": 95,
            "success_rate": 0.95, "total_override": 2,
        }}
        score = _learned_score("openai-codex/gpt-5.4", "coding", stats)
        assert score > 0.85

    def test_learned_score_penalises_overrides(self):
        low_override = {"m": {"total_selected": 100, "total_success": 80, "success_rate": 0.80, "total_override": 2}}
        high_override = {"m": {"total_selected": 100, "total_success": 80, "success_rate": 0.80, "total_override": 40}}
        assert _learned_score("m", "coding", low_override) > _learned_score("m", "coding", high_override)


# ── Router tests ──────────────────────────────────────────────────────────────

class TestRouter:

    def test_route_coding_returns_decision(self, router):
        classifier = ClassifierOutput(task_type="coding", complexity="high", confidence=0.90)
        decision = router.route(classifier)
        assert decision.selected_model
        assert decision.selected_provider
        assert isinstance(decision.fallbacks, list)
        assert len(decision.fallbacks) >= 1

    def test_route_returns_reason(self, router):
        classifier = ClassifierOutput(task_type="code_review", confidence=0.85)
        decision = router.route(classifier)
        assert decision.reason
        assert len(decision.reason) >= 1

    def test_route_all_task_types(self, router):
        task_types = [
            "coding", "code_review", "reasoning", "summarization",
            "fast_utility", "long_context", "vision", "general_chat",
        ]
        for task_type in task_types:
            classifier = ClassifierOutput(
                task_type=task_type,
                needs_vision=(task_type == "vision"),
                needs_long_context=(task_type == "long_context"),
                confidence=0.80,
            )
            decision = router.route(classifier)
            assert decision.selected_model, f"No model selected for task_type={task_type}"

    def test_route_coding_prefers_codex_over_gemini(self, router):
        classifier = ClassifierOutput(task_type="coding", complexity="high", confidence=0.90)
        decision = router.route(classifier)
        # For a high-complexity coding task, should not select a Gemini model as primary
        assert "gemini" not in decision.selected_model.lower() or "copilot" in decision.selected_model.lower(), \
            f"Unexpected primary for coding task: {decision.selected_model}"

    def test_route_long_context_selects_large_window(self, router):
        classifier = ClassifierOutput(task_type="long_context", needs_long_context=True, confidence=0.85)
        decision = router.route(classifier)
        # Should select a model with a large context window (>= 200k)
        # Acceptable: gemini (1M), claude opus (1M), claude sonnet 4.6 (977k), gpt-5.2-codex (400k)
        large_ctx_hints = ["gemini", "opus", "sonnet-4.6", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.4"]
        model = decision.selected_model.lower()
        assert any(h in model for h in large_ctx_hints), \
            f"Expected large-context model, got {decision.selected_model}"

    def test_route_score_between_0_and_1(self, router):
        classifier = ClassifierOutput(task_type="general_chat", confidence=0.75)
        decision = router.route(classifier)
        assert 0.0 <= decision.score <= 1.0

    def test_explain_output(self, router):
        classifier = ClassifierOutput(task_type="coding", confidence=0.88)
        decision = router.route(classifier)
        explanation = router.explain(decision)
        assert "Task:" in explanation
        assert "Model:" in explanation
        assert "Reason:" in explanation

    def test_pre_signals_affect_reason(self, router):
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        pre = PreSignals(has_code=True, has_diff=True, estimated_tokens=500)
        decision = router.route(classifier, pre_signals=pre)
        reasons_text = " ".join(decision.reason)
        assert "code" in reasons_text.lower() or "diff" in reasons_text.lower()


# ── Classifier tests ──────────────────────────────────────────────────────────

class TestClassifier:

    def test_extract_presignals_code_block(self):
        msg = "Here is the code:\n```python\ndef hello(): pass\n```"
        signals = extract_pre_signals(msg)
        assert signals.has_code

    def test_extract_presignals_diff(self):
        msg = "@@ -1,3 +1,4 @@\n-old line\n+new line"
        signals = extract_pre_signals(msg)
        assert signals.has_diff
        assert signals.has_code  # diff implies code

    def test_extract_presignals_stack_trace(self):
        msg = "I'm getting this error:\nTraceback (most recent call last):\n  File 'app.py', line 5\nValueError: bad input"
        signals = extract_pre_signals(msg)
        assert signals.has_logs

    def test_extract_presignals_image(self):
        signals = extract_pre_signals("What's in this image?", has_image_attachment=True)
        assert signals.has_image

    def test_heuristic_classifies_diff_as_review(self):
        msg = "@@ -1,3 +1,4 @@\n-old\n+new"
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        assert result is not None
        assert result.task_type == "code_review"

    def test_heuristic_classifies_image_as_vision(self):
        msg = "What does this look like?"
        signals = extract_pre_signals(msg, has_image_attachment=True)
        result = heuristic_classify(msg, signals)
        assert result is not None
        assert result.task_type == "vision"

    def test_heuristic_short_message_fast_utility(self):
        msg = "What time is it?"
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        assert result is not None
        assert result.task_type == "fast_utility"

    def test_heuristic_returns_none_for_ambiguous(self):
        msg = "Let's think about the best architecture for our new product. We need to balance performance and cost."
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        # Should be None or reasoning — ambiguous, classifier should fire
        if result is not None:
            assert result.task_type in ("reasoning", "general_chat", "fast_utility")

    def test_parse_classifier_response_valid(self):
        raw = '{"task_type": "coding", "complexity": "high", "needs_tools": true, "needs_vision": false, "needs_long_context": false, "cost_profile": "balanced", "confidence": 0.88, "detected_language": "en"}'
        result = parse_classifier_response(raw)
        assert result is not None
        assert result.task_type == "coding"
        assert result.complexity == "high"
        assert result.confidence == 0.88

    def test_parse_classifier_response_with_fences(self):
        raw = '```json\n{"task_type": "summarization", "confidence": 0.75, "complexity": "medium", "needs_tools": false, "needs_vision": false, "needs_long_context": false, "cost_profile": "cheap"}\n```'
        result = parse_classifier_response(raw)
        assert result is not None
        assert result.task_type == "summarization"

    def test_parse_classifier_response_invalid_task_type(self):
        raw = '{"task_type": "magic_task", "confidence": 0.90, "complexity": "low", "needs_tools": false, "needs_vision": false, "needs_long_context": false}'
        result = parse_classifier_response(raw)
        assert result is not None
        assert result.task_type == "general_chat"  # fallback for unknown

    def test_parse_classifier_response_malformed(self):
        result = parse_classifier_response("This is not JSON at all")
        assert result is None


# ── Integration: end-to-end ───────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_coding(self, router):
        """Full pipeline: extract signals → heuristic classify → route"""
        message = "Fix this bug:\n```python\ndef divide(a, b):\n    return a / b\n```\nIt crashes when b=0."
        signals = extract_pre_signals(message)
        classifier = heuristic_classify(message, signals) or ClassifierOutput(
            task_type="coding", subtype="debugging", confidence=0.80
        )
        decision = router.route(classifier, pre_signals=signals)
        assert decision.selected_model
        assert decision.task_type == "coding"

    def test_full_pipeline_vision(self, router):
        message = "What's in this screenshot?"
        signals = extract_pre_signals(message, has_image_attachment=True)
        classifier = heuristic_classify(message, signals) or ClassifierOutput(
            task_type="vision", needs_vision=True, confidence=0.90
        )
        decision = router.route(classifier, pre_signals=signals)
        # Should select a vision-capable model (gemini, claude, gpt-4o, etc.)
        vision_hints = ["gemini", "claude", "gpt-4o", "gpt-5"]
        model = decision.selected_model.lower()
        assert any(h in model for h in vision_hints), \
            f"Expected vision-capable model, got {decision.selected_model}"

    def test_full_pipeline_large_context(self, router):
        message = "A" * 900_000  # very long message
        signals = extract_pre_signals(message)
        classifier = heuristic_classify(message, signals) or ClassifierOutput(
            task_type="long_context", needs_long_context=True, confidence=0.85
        )
        decision = router.route(classifier, pre_signals=signals)
        assert decision.selected_model
        # Should pick a model with large context window
        assert signals.estimated_tokens > 200_000
