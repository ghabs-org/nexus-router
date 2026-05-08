"""
Tests for nexus-router: scorer and router.
Run with: python -m pytest tests/ -v
"""

import json
import os
import pytest
from pathlib import Path

os.environ.setdefault("NEXUS_ROUTER_STATE_ROOT", str(Path(__file__).parent.parent / ".test-state"))

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import ClassifierOutput, PreSignals, ProviderHealth, RoutingDecision
from src.scorer import score_models, _preference_score, _learned_score, _fast_mode_correction, _reasoning_mode_total, _build_preference_order
from src.router import Router, _adapt_classifier_for_light_chat
from src.classifier import extract_pre_signals, heuristic_classify, parse_classifier_response, classify_with_model, select_classifier_models, _call_direct_provider_classifier
from src.server import should_reclassify_with_llm, RouterHandler


# ── Fixtures ──────────────────────────────────────────────────────────────────

from src.paths import REGISTRY_FILE


@pytest.fixture(autouse=True, scope="session")
def ensure_full_registry():
    """Regenerate registry with all configured providers before running tests."""
    import subprocess
    repo_root = Path(__file__).parent.parent
    state_root = repo_root / ".test-state"
    generated_root = state_root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    src_catalog = repo_root / "catalog" / "raw" / "openclaw-models.json"
    if src_catalog.exists():
        (generated_root / "openclaw-models.json").write_text(src_catalog.read_text())

        env = dict(**__import__("os").environ)
        env.setdefault("NEXUS_ROUTER_STATE_ROOT", str(state_root))
        subprocess.run(
            [sys.executable, "src/generate_registry.py", "--only-configured"],
            cwd=str(repo_root),
            capture_output=True,
            env=env,
        )
    else:
        # catalog/raw is intentionally gitignored; fresh checkouts use the
        # committed generated registry for tests that don't exercise fetching.
        committed_registry = repo_root / "generated" / "models.json"
        if committed_registry.exists():
            (generated_root / "models.json").write_text(committed_registry.read_text())


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
        "openrouter": ProviderHealth(provider="openrouter", auth="ok", quota="healthy", health_score=0.95),
        "nvidia": ProviderHealth(provider="nvidia", auth="ok", quota="healthy", health_score=0.95),
    }


@pytest.fixture
def router():
    return Router(persist=False)


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorer:

    def test_new_router_catalog_models_are_available(self, authed_models):
        models = {m["id"]: m for m in authed_models}
        expected = {
            "openai-codex/gpt-5.4": None,
            "openai-codex/gpt-5.3-codex": None,
            "nvidia/deepseek-ai/deepseek-v3.2": True,
            "nvidia/qwen/qwen3.5-122b-a10b": True,
            "openrouter/openai/gpt-oss-120b:free": True,
        }
        for model_id, is_free in expected.items():
            assert model_id in models, f"Expected {model_id} in authed models"
            if is_free is not None:
                assert models[model_id]["features"].get("is_free") is is_free

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
        text_only_grok = [
            s for s in scored
            if s.model_id in {
                "github-copilot/grok-code-fast-1",
                "openrouter/x-ai/grok-3",
                "openrouter/x-ai/grok-3-beta",
                "openrouter/x-ai/grok-3-mini",
                "openrouter/x-ai/grok-3-mini-beta",
                "openrouter/x-ai/grok-code-fast-1",
            }
        ]
        assert text_only_grok, "Expected text-only Grok variants in scored models"
        assert all(s.excluded for s in text_only_grok), "Text-only models should be excluded from vision task"

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

    def test_rate_limited_provider_is_excluded_during_soft_ban(self, authed_models):
        degraded_health = {
            "openai-codex": ProviderHealth(provider="openai-codex", auth="ok", quota="healthy", health_score=0.95),
            "github-copilot": ProviderHealth(provider="github-copilot", auth="ok", quota="healthy", health_score=0.95),
            "google-gemini-cli": ProviderHealth(
                provider="google-gemini-cli",
                auth="unknown",
                quota="low",
                consecutive_rate_limits=1,
                rate_limit_cooldown_until="2099-01-01T00:05:00+00:00",
                health_score=0.0,
            ),
        }
        classifier = ClassifierOutput(task_type="general_chat", confidence=0.85)
        scored = score_models(classifier, authed_models, degraded_health, {})
        eligible = [s for s in scored if not s.excluded]
        gemini_eligible = [s for s in eligible if s.provider == "google-gemini-cli"]
        assert not gemini_eligible, "Soft-banned provider should have no eligible models"

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


    def test_model_preference_bump_is_bounded(self):
        classifier = ClassifierOutput(task_type="reasoning", confidence=0.82)
        provider_health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
        models = [{
            "id": "p/one", "provider": "p",
            "scores": {"coding": 0.8, "review": 0.8, "reasoning": 0.9, "summarize": 0.8, "fast": 0.7, "cost": 0.7, "context": 0.8, "vision": 0.7},
            "features": {"contextWindow": 300000}, "availability": {"authed": True},
        }]
        learned = {
            "p/one": {
                "total_selected": 10,
                "success_rate": 0.8,
                "total_override": 0,
                "feedback_preference": {
                    "reasoning": {"samples": 12, "score": 1.0, "last_feedback_at": "2099-01-01T00:00:00+00:00", "top_reason_tag": "quality"}
                },
            }
        }
        policy = {"feedback": {"model_preference": {"enabled": True, "max_bump": 0.04, "min_samples": 3, "decay_window_days": 30}}}
        scored = score_models(classifier, models, provider_health, learned, routing_policy=policy)
        assert scored[0].model_preference_bump <= 0.04
        assert scored[0].model_preference_bump > 0

    def test_model_preference_bump_not_applied_below_min_samples(self):
        classifier = ClassifierOutput(task_type="reasoning", confidence=0.82)
        provider_health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
        models = [{
            "id": "p/one", "provider": "p",
            "scores": {"coding": 0.8, "review": 0.8, "reasoning": 0.9, "summarize": 0.8, "fast": 0.7, "cost": 0.7, "context": 0.8, "vision": 0.7},
            "features": {"contextWindow": 300000}, "availability": {"authed": True},
        }]
        learned = {
            "p/one": {
                "total_selected": 10,
                "success_rate": 0.8,
                "total_override": 0,
                "feedback_preference": {
                    "reasoning": {"samples": 2, "score": 1.0, "last_feedback_at": "2099-01-01T00:00:00+00:00", "top_reason_tag": "quality"}
                },
            }
        }
        policy = {"feedback": {"model_preference": {"enabled": True, "max_bump": 0.04, "min_samples": 3, "decay_window_days": 30}}}
        scored = score_models(classifier, models, provider_health, learned, routing_policy=policy)
        assert scored[0].model_preference_bump == 0

    def test_model_preference_bump_active_without_routing_policy(self):
        # Regression: feedback penalty must apply even when routing_policy=None
        # (routing.yaml disabled means routing_policy is always None in production)
        classifier = ClassifierOutput(task_type="fast_utility", confidence=0.90)
        provider_health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
        models = [{
            "id": "p/flash", "provider": "p",
            "scores": {"coding": 0.7, "review": 0.7, "reasoning": 0.7, "summarize": 0.7, "fast": 0.8, "cost": 0.8, "context": 0.7, "vision": 0.6},
            "features": {"contextWindow": 128000}, "availability": {"authed": True},
        }]
        learned = {
            "p/flash": {
                "total_selected": 50,
                "success_rate": 0.85,
                "total_override": 0,
                "feedback_preference": {
                    "fast_utility": {"samples": 27, "score": 0.0, "last_feedback_at": "2099-01-01T00:00:00+00:00"}
                },
            }
        }
        # No routing_policy — simulates disabled routing.yaml (the original bug scenario)
        scored = score_models(classifier, models, provider_health, learned, routing_policy=None)
        assert scored[0].model_preference_bump < -0.20, (
            "Expected strong penalty (< -0.20) for 27 negative feedback samples with no routing_policy; "
            f"got {scored[0].model_preference_bump}"
        )

    def test_model_preference_bump_never_overrides_hard_health_constraints(self):
        classifier = ClassifierOutput(task_type="reasoning", confidence=0.82)
        provider_health = {
            "bad": ProviderHealth(provider="bad", auth="ok", quota="healthy", health_score=0.1),
            "good": ProviderHealth(provider="good", auth="ok", quota="healthy", health_score=0.95),
        }
        models = [
            {
                "id": "bad/model", "provider": "bad",
                "scores": {"coding": 0.9, "review": 0.9, "reasoning": 0.95, "summarize": 0.8, "fast": 0.7, "cost": 0.7, "context": 0.8, "vision": 0.7},
                "features": {"contextWindow": 300000}, "availability": {"authed": True},
            },
            {
                "id": "good/model", "provider": "good",
                "scores": {"coding": 0.8, "review": 0.8, "reasoning": 0.8, "summarize": 0.8, "fast": 0.7, "cost": 0.7, "context": 0.8, "vision": 0.7},
                "features": {"contextWindow": 300000}, "availability": {"authed": True},
            },
        ]
        learned = {"bad/model": {"total_selected": 10, "success_rate": 0.9, "total_override": 0, "feedback_preference": {"reasoning": {"samples": 20, "score": 1.0, "last_feedback_at": "2099-01-01T00:00:00+00:00"}}}}
        policy = {"feedback": {"model_preference": {"enabled": True, "max_bump": 0.04, "min_samples": 3, "decay_window_days": 30}}}
        scored = score_models(classifier, models, provider_health, learned, routing_policy=policy)
        bad = next(s for s in scored if s.model_id == "bad/model")
        assert bad.excluded is True
        assert "health_too_low" in (bad.exclusion_reason or "")

    def test_hard_feedback_exclusion_removes_strongly_disliked_model_for_task(self):
        classifier = ClassifierOutput(task_type="general_chat", confidence=0.82, cost_profile="cheap")
        provider_health = {
            "p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95),
        }
        models = [
            {
                "id": "p/disliked-flash", "provider": "p",
                "scores": {"coding": 0.7, "review": 0.7, "reasoning": 0.7, "summarize": 0.8, "fast": 0.96, "cost": 0.97, "context": 0.7, "vision": 0.6},
                "features": {"contextWindow": 128000}, "availability": {"authed": True},
            },
            {
                "id": "p/next-best-flash", "provider": "p",
                "scores": {"coding": 0.7, "review": 0.7, "reasoning": 0.76, "summarize": 0.82, "fast": 0.90, "cost": 0.90, "context": 0.7, "vision": 0.6},
                "features": {"contextWindow": 128000}, "availability": {"authed": True},
            },
        ]
        learned = {
            "p/disliked-flash": {
                "total_selected": 120,
                "success_rate": 0.83,
                "total_override": 0,
                "feedback_preference": {
                    "general_chat": {
                        "samples": 49,
                        "score": 0.0,
                        "last_feedback_at": "2099-01-01T00:00:00+00:00",
                        "top_reason_tag": "too_cheap",
                    }
                },
            },
            "p/next-best-flash": {
                "total_selected": 20,
                "success_rate": 0.84,
                "total_override": 0,
            },
        }
        policy = {"feedback": {"model_preference": {
            "enabled": True,
            "max_bump": 0.04,
            "max_penalty": 0.30,
            "min_samples": 3,
            "decay_window_days": 30,
            "hard_exclude_enabled": True,
            "hard_exclude_min_samples": 10,
            "hard_exclude_centered_score_lte": -0.8,
        }}}
        scored = score_models(classifier, models, provider_health, learned, routing_policy=policy)
        winner = next(s for s in scored if not s.excluded)
        disliked = next(s for s in scored if s.model_id == "p/disliked-flash")
        assert disliked.excluded is True
        assert "feedback_hard_exclude:general_chat" in (disliked.exclusion_reason or "")
        assert winner.model_id == "p/next-best-flash"

    def test_fast_mode_correction_prefers_lower_cost_and_higher_speed(self):
        cheap_fast = _fast_mode_correction(task_fit=0.80, cost_score=0.90, speed_score=0.90)
        expensive_slow = _fast_mode_correction(task_fit=0.80, cost_score=0.55, speed_score=0.55)
        assert cheap_fast > expensive_slow

    def test_reasoning_mode_correction_rewards_top_models(self):
        strong = _reasoning_mode_total(task_fit=0.93, reasoning_score=0.95, health=0.95, learned=0.92, preference=0.5, speed_score=0.5, quota_penalty=0.0)
        weak = _reasoning_mode_total(task_fit=0.75, reasoning_score=0.80, health=0.80, learned=0.78, preference=0.5, speed_score=0.5, quota_penalty=0.0)
        assert strong > weak

    def test_fast_utility_preference_order_prefers_gemini_3_flash_preview_over_2_5_flash(self):
        import yaml
        policy_path = Path(__file__).parent.parent / "policies" / "routing.yaml"
        with open(policy_path) as f:
            routing_policy = yaml.safe_load(f)

        order = _build_preference_order("fast_utility", routing_policy)
        assert order.index("google-gemini-cli/gemini-3-flash-preview") < order.index("google-gemini-cli/gemini-2.5-flash")

    def test_fast_route_mode_changes_ranking_bias(self):
        classifier = ClassifierOutput(task_type="coding", complexity="medium", confidence=0.80)
        provider_health = {
            "p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95),
        }

        models = [
            {
                "id": "p/high-taskfit-expensive",
                "provider": "p",
                "scores": {
                    "coding": 0.92,
                    "review": 0.70,
                    "reasoning": 0.70,
                    "summarize": 0.70,
                    "fast": 0.60,
                    "cost": 0.55,
                    "context": 0.70,
                    "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
            {
                "id": "p/good-taskfit-cheap-fast",
                "provider": "p",
                "scores": {
                    "coding": 0.85,
                    "review": 0.70,
                    "reasoning": 0.70,
                    "summarize": 0.70,
                    "fast": 0.95,
                    "cost": 0.95,
                    "context": 0.70,
                    "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
        ]

        base_ranked = score_models(classifier, models, provider_health, {}, route_mode="balanced")
        fast_ranked = score_models(classifier, models, provider_health, {}, route_mode="fast")

        def by_id(rows, model_id):
            return next(r for r in rows if r.model_id == model_id)

        cheap_base = by_id(base_ranked, "p/good-taskfit-cheap-fast").total_score
        expensive_base = by_id(base_ranked, "p/high-taskfit-expensive").total_score
        cheap_fast = by_id(fast_ranked, "p/good-taskfit-cheap-fast").total_score
        expensive_fast = by_id(fast_ranked, "p/high-taskfit-expensive").total_score

        assert (cheap_fast - expensive_fast) > (cheap_base - expensive_base)
        assert fast_ranked[0].model_id == "p/good-taskfit-cheap-fast"

    def test_high_complexity_reasoning_excludes_lightweight_models(self):
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        provider_health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
        models = [
            {
                "id": "p/strong-pro",
                "provider": "p",
                "scores": {
                    "coding": 0.80, "review": 0.80, "reasoning": 0.92, "summarize": 0.74,
                    "fast": 0.60, "cost": 0.55, "context": 0.80, "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
            {
                "id": "p/cheap-mini",
                "provider": "p",
                "scores": {
                    "coding": 0.84, "review": 0.84, "reasoning": 0.90, "summarize": 0.78,
                    "fast": 0.96, "cost": 0.96, "context": 0.80, "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
        ]

        ranked = score_models(classifier, models, provider_health, {})
        assert ranked[0].model_id == "p/strong-pro"
        assert next(s for s in ranked if s.model_id == "p/cheap-mini").excluded is True

    def test_reasoning_route_mode_prefers_stronger_reasoning_models(self):
        classifier = ClassifierOutput(task_type="coding", complexity="high", confidence=0.85, cost_profile="balanced")
        provider_health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}

        models = [
            {
                "id": "p/strong-reasoning",
                "provider": "p",
                "scores": {
                    "coding": 0.85, "review": 0.80, "reasoning": 0.95, "summarize": 0.74,
                    "fast": 0.60, "cost": 0.62, "context": 0.80, "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
            {
                "id": "p/weaker-reasoning",
                "provider": "p",
                "scores": {
                    "coding": 0.90, "review": 0.70, "reasoning": 0.82, "summarize": 0.74,
                    "fast": 0.80, "cost": 0.90, "context": 0.80, "vision": 0.60,
                },
                "features": {"contextWindow": 200000},
                "availability": {"authed": True},
            },
        ]

        ranked = score_models(classifier, models, provider_health, {}, route_mode="reasoning")
        assert ranked[0].model_id == "p/strong-reasoning"


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

    def test_adapt_only_weak_tiny_low_complexity_general_chat_to_fast_utility(self):
        classifier = ClassifierOutput(task_type="general_chat", complexity="low", confidence=0.30)
        pre = PreSignals(message_length=20, estimated_tokens=6)
        adapted = _adapt_classifier_for_light_chat(classifier, pre)
        assert adapted.task_type == "fast_utility"
        assert adapted.cost_profile == "cheap"

    def test_keep_high_confidence_low_complexity_general_chat_as_general_chat(self):
        classifier = ClassifierOutput(task_type="general_chat", complexity="low", confidence=0.95)
        pre = PreSignals(message_length=80, estimated_tokens=20)
        adapted = _adapt_classifier_for_light_chat(classifier, pre)
        assert adapted.task_type == "general_chat"

    def test_adapt_low_confidence_short_general_chat_to_fast_utility(self):
        classifier = ClassifierOutput(task_type="general_chat", complexity="medium", confidence=0.25)
        pre = PreSignals(message_length=12, estimated_tokens=4)
        adapted = _adapt_classifier_for_light_chat(classifier, pre)
        assert adapted.task_type == "fast_utility"
        assert adapted.complexity == "low"

    def test_keep_rich_general_chat_as_general_chat(self):
        classifier = ClassifierOutput(task_type="general_chat", complexity="medium", confidence=0.60)
        pre = PreSignals(message_length=300, estimated_tokens=90, has_code=True)
        adapted = _adapt_classifier_for_light_chat(classifier, pre)
        assert adapted.task_type == "general_chat"

    def test_explain_output(self, router):
        classifier = ClassifierOutput(task_type="coding", confidence=0.88)
        decision = router.route(classifier)
        explanation = router.explain(decision)
        assert "Task:" in explanation
        assert "Model:" in explanation
        assert "Reason:" in explanation

    def test_pre_signals_affect_reason(self, router):
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        pre = PreSignals(has_code=True, has_diff=True, has_file_refs=True, estimated_tokens=500)
        decision = router.route(classifier, pre_signals=pre)
        reasons_text = " ".join(decision.reason).lower()
        assert "inline/fenced code detected" in reasons_text
        assert "diff/patch detected" in reasons_text
        assert "file path/reference detected" in reasons_text

    def test_route_mode_auto_forces_cheap_profile(self, router):
        classifier = ClassifierOutput(task_type="general_chat", cost_profile="balanced", confidence=0.70)
        decision = router.route(classifier, route_mode="auto")
        reasons_text = " ".join(decision.reason).lower()
        assert "route mode: auto" in reasons_text

    def test_route_mode_balanced_preserves_profile(self, router):
        classifier = ClassifierOutput(task_type="general_chat", cost_profile="balanced", confidence=0.70)
        decision = router.route(classifier, route_mode="balanced")
        reasons_text = " ".join(decision.reason).lower()
        assert "route mode: balanced" in reasons_text


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
        # Diff is a content signal — heuristic no longer classifies it; returns None.
        msg = "@@ -1,3 +1,4 @@\n-old\n+new"
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        assert result is None

    def test_heuristic_classifies_image_as_vision(self):
        # Image attachment — always fires regardless of message length.
        msg = "What does this look like?"
        signals = extract_pre_signals(msg, has_image_attachment=True)
        result = heuristic_classify(msg, signals)
        assert result is not None
        assert result.task_type == "vision"

    def test_heuristic_classifies_image_as_vision_long_message(self):
        # Image attachment with long text — should still fire (not gated on length).
        msg = "Describe this image in detail. " * 30  # >500 chars
        signals = extract_pre_signals(msg, has_image_attachment=True)
        result = heuristic_classify(msg, signals)
        assert result is not None
        assert result.task_type == "vision"

    def test_heuristic_short_message_fast_utility(self):
        # Short message without image or >200k token count — falls through to classifier.
        msg = "What time is it?"
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        assert result is None

    def test_heuristic_single_word_test_not_coding(self):
        # Single-word message — content heuristics removed; returns None.
        msg = "test"
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        assert result is None

    def test_heuristic_returns_none_for_ambiguous(self):
        msg = "Let's think about the best architecture for our new product. We need to balance performance and cost."
        signals = extract_pre_signals(msg)
        result = heuristic_classify(msg, signals)
        # Content-based heuristics removed — structural heuristic always returns None here.
        assert result is None

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

    def test_should_reclassify_with_llm_on_reply_context(self):
        classifier = ClassifierOutput(task_type="fast_utility", confidence=0.72)
        assert should_reclassify_with_llm(classifier, True, "Previous coding discussion")
        assert should_reclassify_with_llm(ClassifierOutput(task_type="general_chat", confidence=0.55), True, "Previous coding discussion")
        assert not should_reclassify_with_llm(classifier, False, "Previous coding discussion")
        assert not should_reclassify_with_llm(ClassifierOutput(task_type="coding", confidence=0.72), True, "Previous coding discussion")

    def test_select_classifier_models_prefers_efficient_healthy_model(self):
        registry = [
            {
                "id": "slow/strong",
                "provider": "slow",
                "availability": {"authed": True},
                "scores": {"cost": 0.55, "fast": 0.55, "reasoning": 0.95, "multilingual": 0.95, "tools": 0.80, "vision": 0.80},
            },
            {
                "id": "cheap/healthy",
                "provider": "cheap",
                "availability": {"authed": True},
                "scores": {"cost": 0.95, "fast": 0.92, "reasoning": 0.78, "multilingual": 0.80, "tools": 0.80, "vision": 0.80},
            },
            {
                "id": "cheap/bad-health",
                "provider": "cheap-low",
                "availability": {"authed": True},
                "scores": {"cost": 0.97, "fast": 0.94, "reasoning": 0.79, "multilingual": 0.78, "tools": 0.80, "vision": 0.80},
            },
        ]
        health = {
            "slow": ProviderHealth(provider="slow", auth="ok", quota="healthy", health_score=0.85),
            "cheap": ProviderHealth(provider="cheap", auth="ok", quota="healthy", health_score=0.95),
            "cheap-low": ProviderHealth(provider="cheap-low", auth="ok", quota="healthy", health_score=0.25),
        }
        selected = select_classifier_models(registry=registry, provider_health=health, preferred_model=None, limit=3)
        assert selected[0] == "cheap/healthy"
        assert "cheap/bad-health" not in selected

    def test_call_direct_provider_classifier_openai_codex(self, monkeypatch):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse({
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"task_type":"reasoning","complexity":"medium","needs_tools":false,"needs_vision":false,"needs_long_context":false,"cost_profile":"balanced","confidence":0.89}'
                            }
                        ]
                    }
                ]
            })

        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr("src.classifier.urllib_request.urlopen", fake_urlopen)

        raw = _call_direct_provider_classifier("classify this", "openai-codex/gpt-5.4-mini", timeout_seconds=7)
        assert raw is not None
        assert '"task_type":"reasoning"' in raw
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer test-openai-key"
        assert captured["body"]["model"] == "gpt-5.4-mini"
        assert captured["timeout"] == 7

    def test_classify_with_model_prefers_direct_provider_before_cli(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(
            "src.classifier._call_direct_provider_classifier",
            lambda *args, **kwargs: '{"task_type":"reasoning","subtype":"comparison","complexity":"medium","needs_tools":false,"needs_vision":false,"needs_long_context":false,"cost_profile":"balanced","confidence":0.91,"detected_language":"it"}',
        )
        monkeypatch.setattr("src.classifier.shutil.which", lambda _: None)

        pre = PreSignals(message_length=42)
        result = classify_with_model("Confronta queste due architetture.", pre, model="openai-codex/gpt-5.4-mini")
        assert result is not None
        assert result.task_type == "reasoning"
        assert result.subtype == "comparison"
        assert result.detected_language == "it"
        assert result.classifier_model is not None
        assert result.classifier_provider is not None

    def test_classify_with_model_falls_back_to_openclaw_cli(self, monkeypatch):
        class Result:
            def __init__(self, returncode=0, stdout=""):
                self.returncode = returncode
                self.stdout = stdout

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:4] == ["openclaw", "--profile", "nexus-router-classifier", "models"] and cmd[4] == "set":
                return Result(0, "")
            if cmd[:6] == ["openclaw", "--profile", "nexus-router-classifier", "agent", "--agent", "main"]:
                return Result(0, '{"task_type":"reasoning","subtype":"comparison","complexity":"medium","needs_tools":false,"needs_vision":false,"needs_long_context":false,"cost_profile":"balanced","confidence":0.91,"detected_language":"it"}')
            raise AssertionError(f"Unexpected command: {cmd}")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("src.classifier._call_direct_provider_classifier", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.classifier._run_codex_classifier_turn", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.classifier.shutil.which", lambda b: "/usr/bin/openclaw" if b == "openclaw" else None)
        monkeypatch.setattr("src.classifier.subprocess.run", fake_run)

        pre = PreSignals(message_length=42)
        result = classify_with_model("Confronta queste due architetture.", pre, model="openai-codex/gpt-5.4-mini")
        assert result is not None
        assert result.task_type == "reasoning"
        assert result.subtype == "comparison"
        assert result.detected_language == "it"
        assert result.classifier_model is not None
        assert result.classifier_provider is not None
        assert any(cmd[:4] == ["openclaw", "--profile", "nexus-router-classifier", "models"] and cmd[4] == "set" for cmd in calls)


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




# ── Provenance fields: /route endpoint ───────────────────────────────────────

import io
import src.server as server_module
from src.server import RouterHandler

_MINIMAL_REGISTRY = {
    "generatedAt": "2026-01-01T00:00:00Z",
    "totalModels": 1,
    "models": [
        {
            "id": "openai-codex/gpt-5.4",
            "provider": "openai-codex",
            "modelId": "gpt-5.4",
            "name": "GPT-5.4",
            "features": {
                "supportsVision": True,
                "supportsTools": True,
                "supportsReasoning": False,
                "contextWindow": 266000,
                "inputModalities": ["text", "image"],
            },
            "availability": {"authed": True, "available": True, "local": False},
            "scores": {
                "coding": 0.96,
                "review": 0.82,
                "reasoning": 0.85,
                "summarize": 0.74,
                "fast": 0.68,
                "cost": 0.55,
                "context": 0.80,
                "vision": 0.75,
                "tools": 0.92,
                "multilingual": 0.78,
            },
            "scoreSource": {},
        }
    ],
}


class _MockWFile:
    def __init__(self):
        self._buf = io.BytesIO()

    def write(self, b: bytes):
        self._buf.write(b)


class _MockHandler:
    """Minimal BaseHTTPRequestHandler stand-in for testing _handle_route directly."""

    def __init__(self):
        self.wfile = _MockWFile()
        self.status: int | None = None

    def send_response(self, code: int):
        self.status = code

    def send_header(self, key: str, value: str):
        pass

    def end_headers(self):
        pass

    def call_route(self, body: dict) -> tuple[int, dict]:
        RouterHandler._handle_route(self, body)
        return self.status, json.loads(self.wfile._buf.getvalue())


class TestRouteProvenance:

    @pytest.fixture(autouse=True)
    def _patch_router(self, tmp_path, monkeypatch):
        """Provide a minimal router so /route tests don't need a real registry."""
        reg_file = tmp_path / "models.json"
        reg_file.write_text(json.dumps(_MINIMAL_REGISTRY))
        minimal_router = Router(registry_path=reg_file, persist=False)

        def fake_route(*_args, **kwargs):
            classifier = kwargs.get("classifier")
            return RoutingDecision(
                task_type=classifier.task_type,
                confidence=classifier.confidence,
                selected_model="openai-codex/gpt-5.4",
                selected_provider="openai-codex",
                score=0.8,
                reason=[],
            )

        minimal_router.route = fake_route
        monkeypatch.setattr(server_module, "_router", minimal_router)
        monkeypatch.setattr(server_module, "classify_with_local_model", lambda *_args, **_kwargs: None)
        healthy_provider = lambda: {
            "openai-codex": ProviderHealth(
                provider="openai-codex",
                auth="ok",
                quota="healthy",
                health_score=0.95,
            )
        }
        monkeypatch.setattr(server_module, "load_provider_health", healthy_provider)
        monkeypatch.setattr("src.router.load_provider_health", healthy_provider)

    def test_explicit_classifier_source(self):
        """Providing a 'classifier' hint yields classifier_source='explicit'."""
        status, body = _MockHandler().call_route({
            "message": "summarise this",
            "classifier": {
                "task_type": "summarization",
                "complexity": "low",
                "confidence": 0.90,
            },
        })
        assert status == 200
        assert body["classifier_source"] == "explicit"
        assert body["reply_context_used"] is False

    def test_heuristic_classifier_source(self):
        """Image message → heuristic detects vision → classifier_source='heuristic'."""
        status, body = _MockHandler().call_route({
            "message": "What's in this screenshot?",
            "has_image_attachment": True,
        })
        assert status == 200
        assert body["classifier_source"] == "heuristic"
        assert body["reply_context_used"] is False

    def test_fallback_classifier_source(self):
        """Medium-length message with no special signals, no LLM → classifier_source='fallback'."""
        status, body = _MockHandler().call_route({
            "message": (
                "Please explain the difference between REST and GraphQL in detail,"
                " including performance trade-offs for various use cases."
            ),
            "use_llm_classifier": False,
        })
        assert status == 200
        assert body["classifier_source"] == "fallback"
        assert body["reply_context_used"] is False

    def test_llm_classifier_source_with_context(self, monkeypatch):
        """LLM path taken with context → classifier_source='llm', reply_context_used=True."""
        mock_result = ClassifierOutput(
            task_type="reasoning", complexity="medium", confidence=0.88
        )
        monkeypatch.setattr(server_module, "classify_with_model", lambda **_kw: mock_result)
        monkeypatch.setattr(server_module, "heuristic_classify", lambda *_a: None)

        status, body = _MockHandler().call_route({
            "message": "what do you think?",
            "use_llm_classifier": True,
            "conversation_context": "We were discussing distributed systems.",
        })
        assert status == 200
        assert body["classifier_source"] == "llm"
        assert body["reply_context_used"] is True

    def test_llm_classifier_source_without_context(self, monkeypatch):
        """LLM path taken but no context → classifier_source='llm', reply_context_used=False."""
        mock_result = ClassifierOutput(
            task_type="general_chat", complexity="low", confidence=0.80
        )
        monkeypatch.setattr(server_module, "classify_with_model", lambda **_kw: mock_result)
        monkeypatch.setattr(server_module, "heuristic_classify", lambda *_a: None)

        status, body = _MockHandler().call_route({
            "message": "Hello",
            "use_llm_classifier": True,
        })
        assert status == 200
        assert body["classifier_source"] == "llm"
        assert body["reply_context_used"] is False

    def test_heuristic_with_context_does_not_set_reply_context_used(self):
        """Heuristic classifies as vision (not reclassified by LLM) → reply_context_used=False."""
        status, body = _MockHandler().call_route({
            "message": "What's in this screenshot?",
            "has_image_attachment": True,
            "use_llm_classifier": True,
            "conversation_context": "We discussed Python earlier.",
        })
        assert status == 200
        # vision task_type ≠ fast_utility → no LLM reclassification triggered
        assert body["classifier_source"] == "heuristic"
        assert body["reply_context_used"] is False

    def test_invalid_conversation_context_list_returns_400(self):
        """Non-string conversation_context (list) → 400."""
        status, body = _MockHandler().call_route({
            "message": "Hello",
            "conversation_context": ["not", "a", "string"],
        })
        assert status == 400
        assert body.get("error") == "invalid_conversation_context_type"

    def test_invalid_conversation_context_dict_returns_400(self):
        """Non-string conversation_context (dict) → 400."""
        status, body = _MockHandler().call_route({
            "message": "Hello",
            "conversation_context": {"key": "value"},
        })
        assert status == 400
        assert body.get("error") == "invalid_conversation_context_type"



def test_load_model_stats_includes_feedback_preferences(tmp_path, monkeypatch):
    import importlib
    import os

    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "routing.sqlite"))
    import src.paths as paths
    import src.db as db
    importlib.reload(paths)
    importlib.reload(db)

    db.ensure_schema()
    conn = db._connect()
    try:
        conn.execute(
            "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, selected_provider) VALUES (?,?,?,?,?)",
            ("dec-feedback-pref-1", db._now_iso(), "reasoning", "model/a", "provider-a"),
        )
        conn.commit()
    finally:
        conn.close()

    db.record_feedback(
        decision_id="dec-feedback-pref-1",
        verdict="wrong",
        preferred_model="model/a",
        model_verdict="good",
        reason_tag="quality",
        source_surface="telegram",
    )
    stats = db.load_model_stats()
    assert stats["model/a"]["feedback_preference"]["reasoning"]["samples"] == 1


def test_load_model_stats_maps_too_cheap_feedback_to_selected_model_when_preferred_model_missing(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "routing.sqlite"))
    import src.db as db
    importlib.reload(db)

    db.ensure_schema()
    conn = db._connect()
    try:
        conn.execute(
            "INSERT INTO routing_decisions (id, created_at, task_type, selected_model, selected_provider) VALUES (?,?,?,?,?)",
            ("dec-too-cheap-1", db._now_iso(), "general_chat", "google-gemini-cli/gemini-2.5-flash", "google-gemini-cli"),
        )
        conn.commit()
    finally:
        conn.close()

    db.record_feedback(
        decision_id="dec-too-cheap-1",
        verdict="wrong",
        preferred_model=None,
        model_verdict="too_cheap",
        reason_tag="too_cheap",
        source_surface="telegram",
    )
    stats = db.load_model_stats()
    pref = stats["google-gemini-cli/gemini-2.5-flash"]["feedback_preference"]["general_chat"]
    assert pref["samples"] == 1
    assert pref["score"] == 0.0
    assert pref["top_reason_tag"] == "too_cheap"


def test_provider_table_seed_enables_only_primary_runtime_providers(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "routing.sqlite"))
    import src.paths as paths
    import src.db as db
    importlib.reload(paths)
    importlib.reload(db)

    db.ensure_schema()
    conn = db._connect()
    try:
        now = db._now_iso()
        conn.executemany(
            "INSERT INTO provider_health_state (provider, auth, quota, health_score, last_check_at) VALUES (?,?,?,?,?)",
            [
                ("openai-codex", "ok", "healthy", 1.0, now),
                ("google-gemini-cli", "ok", "healthy", 0.9, now),
                ("openrouter", "ok", "healthy", 0.8, now),
                ("nvidia", "ok", "healthy", 0.8, now),
                ("github-copilot", "ok", "unknown", 0.9, now),
            ],
        )
        conn.commit()
        db._ensure_providers_compat(conn)
        enabled = {row["provider"] for row in conn.execute("SELECT provider FROM providers WHERE enabled=1")}
        disabled = {row["provider"] for row in conn.execute("SELECT provider FROM providers WHERE enabled=0")}
    finally:
        conn.close()

    assert enabled == {"openai-codex", "google-gemini-cli", "openrouter", "nvidia"}
    assert "github-copilot" in disabled


def test_disabled_provider_control_returns_zero_health(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("NEXUS_ROUTER_DB_PATH", str(tmp_path / "routing.sqlite"))
    import src.paths as paths
    import src.db as db
    import src.health as health
    importlib.reload(paths)
    importlib.reload(db)
    importlib.reload(health)

    db.ensure_schema()
    conn = db._connect()
    try:
        now = db._now_iso()
        conn.execute(
            "INSERT INTO provider_health_state (provider, auth, quota, health_score, last_check_at) VALUES (?,?,?,?,?)",
            ("github-copilot", "ok", "healthy", 1.0, now),
        )
        conn.execute(
            "UPDATE providers SET enabled=0, status='disabled', updated_at=? WHERE provider='github-copilot'",
            (now,),
        )
        conn.commit()
    finally:
        conn.close()

    loaded = health.load_provider_health(["github-copilot"])
    assert loaded["github-copilot"].health_score == 0.0
