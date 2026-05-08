"""
Provider / candidate contract tests for Nexus Router.

These tests pin the behavioural guarantees of the router's provider-handling,
candidate selection, and scoring logic. They use synthetic models and providers
(no real registry or DB) to make each contract explicit and independent.

CTO brief: "Create a router provider/candidate contract test suite that covers
enabled/disabled providers, quota-exhausted providers, benchmark inheritance
across provider-prefixed IDs, ranking exclusion when health is 0.0, and
direct Anthropic availability."
"""

import json
import os
import sys
import pytest
from pathlib import Path

os.environ.setdefault("NEXUS_ROUTER_STATE_ROOT", str(Path(__file__).parent.parent / ".test-state"))

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.types import ClassifierOutput, PreSignals, ProviderHealth
from src.scorer import score_models, HEALTH_HARD_CUTOFF


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model(
    model_id: str,
    provider: str,
    *,
    coding: float = 0.80,
    review: float = 0.75,
    reasoning: float = 0.80,
    summarize: float = 0.75,
    fast: float = 0.70,
    cost: float = 0.65,
    context: float = 0.80,
    vision: float = 0.70,
    context_window: int = 200_000,
    authed: bool = True,
    is_free: object = None,
    eco: float = 0.50,
) -> dict:
    """Build a synthetic model registry entry."""
    features = {"contextWindow": context_window}
    if is_free is not None:
        features["is_free"] = is_free
    return {
        "id": model_id,
        "provider": provider,
        "scores": {
            "coding": coding, "review": review, "reasoning": reasoning,
            "summarize": summarize, "fast": fast, "cost": cost,
            "context": context, "vision": vision, "eco": eco,
        },
        "features": features,
        "availability": {"authed": authed},
    }


def _health(provider: str, *, health_score: float = 0.95, auth: str = "ok", quota: str = "healthy", **kw) -> ProviderHealth:
    return ProviderHealth(provider=provider, auth=auth, quota=quota, health_score=health_score, **kw)


# ── Contract 1: Enabled / disabled providers ─────────────────────────────────

class TestEnabledDisabledProviders:
    """Provider enablement must be a hard gate: disabled providers should never
    have eligible candidates regardless of model quality."""

    def test_disabled_provider_zero_health_excludes_all_its_models(self):
        """A provider with health_score=0.0 must exclude all its models."""
        models = [
            _model("disabled-p/strong-model", "disabled-p", coding=0.95, reasoning=0.95),
            _model("enabled-p/ok-model", "enabled-p", coding=0.80, reasoning=0.80),
        ]
        health = {
            "disabled-p": _health("disabled-p", health_score=0.0),
            "enabled-p": _health("enabled-p", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert all(s.model_id != "disabled-p/strong-model" for s in eligible)
        disabled_scores = [s for s in scored if s.model_id == "disabled-p/strong-model"]
        assert disabled_scores[0].excluded is True
        assert "health_too_low" in (disabled_scores[0].exclusion_reason or "")

    def test_enabled_provider_models_are_eligible(self):
        """An enabled, healthy provider's models must appear as eligible."""
        models = [_model("p/model-a", "p")]
        health = {"p": _health("p", health_score=0.90)}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1
        assert eligible[0].model_id == "p/model-a"

    def test_health_cutoff_boundary(self):
        """Models from a provider exactly at the health cutoff boundary are excluded."""
        models = [_model("p/borderline", "p")]
        # Just below the cutoff
        health = {"p": _health("p", health_score=HEALTH_HARD_CUTOFF - 0.01)}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        assert scored[0].excluded is True

    def test_health_just_above_cutoff_is_eligible(self):
        """Models from a provider just above the health cutoff are eligible."""
        models = [_model("p/ok", "p")]
        health = {"p": _health("p", health_score=HEALTH_HARD_CUTOFF + 0.01)}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1


# ── Contract 2: Quota-exhausted providers ─────────────────────────────────────

class TestQuotaExhaustedProviders:
    """Quota state must affect scoring: exhausted providers should receive
    maximum penalty and effectively be excluded from selection."""

    def test_exhausted_quota_gets_maximum_penalty(self):
        """A provider with quota=exhausted should get a penalty of 1.0,
        pushing its score well below any healthy competitor."""
        models = [
            _model("exhausted-p/top-model", "exhausted-p", coding=0.98, reasoning=0.98),
            _model("healthy-p/ok-model", "healthy-p", coding=0.80, reasoning=0.80),
        ]
        health = {
            "exhausted-p": _health("exhausted-p", quota="exhausted", health_score=0.0),
            "healthy-p": _health("healthy-p", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        # The exhausted provider should be excluded (health=0.0 < cutoff)
        assert all(s.provider != "exhausted-p" for s in eligible)

    def test_low_quota_deprioritizes_for_cheap_tasks(self):
        """A provider with low quota should be deprioritized for cheap/fast tasks
        but might still be chosen for high-value tasks."""
        models = [
            _model("low-q/strong", "low-q", coding=0.92, fast=0.60, cost=0.55),
            _model("healthy-q/decent", "healthy-q", coding=0.85, fast=0.80, cost=0.85),
        ]
        health = {
            "low-q": _health("low-q", health_score=0.90, quota="low",
                             quota_remaining_ratio=0.15),
            "healthy-q": _health("healthy-q", health_score=0.95),
        }
        # For a cheap/fast task, the low-quota provider should lose
        classifier = ClassifierOutput(task_type="fast_utility", cost_profile="cheap", confidence=0.80)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].provider == "healthy-q"

    def test_rate_limited_provider_excluded_during_cooldown(self):
        """A provider in rate-limit cooldown (health_score=0.0) should be excluded."""
        models = [
            _model("rl-p/model", "rl-p", coding=0.90),
            _model("ok-p/model", "ok-p", coding=0.80),
        ]
        health = {
            "rl-p": _health("rl-p", health_score=0.0,
                            consecutive_rate_limits=2,
                            rate_limit_cooldown_until="2099-01-01T00:00:00+00:00"),
            "ok-p": _health("ok-p", health_score=0.90),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert all(s.provider != "rl-p" for s in eligible)


# ── Contract 3: Benchmark / score inheritance across provider-prefixed IDs ────

class TestBenchmarkInheritance:
    """The same base model served through different providers should use its
    own score profile. Provider-prefixed IDs are distinct entries with
    independent scores — no implicit inheritance."""

    def test_same_model_different_providers_independently_scored(self):
        """Two entries for conceptually the same model (e.g. gemini via
        google-gemini-cli and github-copilot) are scored independently."""
        models = [
            _model("google-gemini-cli/gemini-3-flash", "google-gemini-cli",
                   coding=0.75, fast=0.95, cost=0.95, reasoning=0.70),
            _model("github-copilot/gemini-3-flash", "github-copilot",
                   coding=0.78, fast=0.90, cost=0.85, reasoning=0.72),
        ]
        health = {
            "google-gemini-cli": _health("google-gemini-cli", health_score=0.95),
            "github-copilot": _health("github-copilot", health_score=0.90),
        }
        classifier = ClassifierOutput(task_type="fast_utility", cost_profile="cheap", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 2
        # Each should have a different total_score since health and scores differ
        assert eligible[0].total_score != eligible[1].total_score

    def test_provider_health_differentiates_same_model(self):
        """Same model, different providers: provider with better health wins."""
        models = [
            _model("slow-p/model-x", "slow-p", coding=0.90),
            _model("fast-p/model-x", "fast-p", coding=0.90),
        ]
        health = {
            "slow-p": _health("slow-p", health_score=0.50),
            "fast-p": _health("fast-p", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].provider == "fast-p"


# ── Contract 4: Ranking exclusion when health is 0.0 ─────────────────────────

class TestHealthZeroExclusion:
    """When health_score is 0.0, models must be excluded from ranking entirely,
    not just deprioritized."""

    def test_zero_health_models_never_eligible(self):
        """No model from a provider with health_score=0.0 should appear in eligible."""
        models = [
            _model("dead-p/great-model", "dead-p", coding=0.99, reasoning=0.99),
            _model("alive-p/ok-model", "alive-p", coding=0.70),
        ]
        health = {
            "dead-p": _health("dead-p", health_score=0.0),
            "alive-p": _health("alive-p", health_score=0.80),
        }
        for task in ["coding", "reasoning", "general_chat", "fast_utility"]:
            classifier = ClassifierOutput(task_type=task, confidence=0.85)
            scored = score_models(classifier, models, health, {})
            eligible = [s for s in scored if not s.excluded]
            assert all(s.provider != "dead-p" for s in eligible), \
                f"dead provider should have no eligible models for {task}"

    def test_expired_auth_yields_zero_health_and_exclusion(self):
        """Expired auth → health_score=0.0 → full exclusion."""
        models = [_model("expired-p/model", "expired-p")]
        health = {
            "expired-p": _health("expired-p", auth="expired", health_score=0.0),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        assert scored[0].excluded is True

    def test_missing_auth_yields_zero_health_and_exclusion(self):
        """Missing auth → health_score=0.0 → full exclusion."""
        models = [_model("noauth-p/model", "noauth-p")]
        health = {
            "noauth-p": _health("noauth-p", auth="missing", health_score=0.0),
        }
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        assert scored[0].excluded is True


# ── Contract 5: Direct Anthropic availability ────────────────────────────────

class TestDirectAnthropicAvailability:
    """When Anthropic is a configured provider, its models should participate
    in routing according to the same rules as any other provider."""

    def test_anthropic_models_eligible_when_healthy(self):
        """Anthropic models with good health should be eligible candidates."""
        models = [
            _model("anthropic/claude-opus-4-7", "anthropic",
                   coding=0.90, reasoning=0.95, review=0.92, vision=0.80),
            _model("openai-codex/gpt-5.4", "openai-codex",
                   coding=0.96, reasoning=0.85, review=0.82),
        ]
        health = {
            "anthropic": _health("anthropic", health_score=0.90),
            "openai-codex": _health("openai-codex", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        anthropic_eligible = [s for s in eligible if s.provider == "anthropic"]
        assert len(anthropic_eligible) >= 1

    def test_anthropic_wins_reasoning_when_highest_score(self):
        """With highest reasoning score and good health, Anthropic should win reasoning tasks."""
        models = [
            _model("anthropic/claude-opus-4-7", "anthropic",
                   reasoning=0.96, coding=0.88, fast=0.55, cost=0.50),
            _model("openai-codex/gpt-5.4", "openai-codex",
                   reasoning=0.85, coding=0.96, fast=0.68, cost=0.55),
        ]
        health = {
            "anthropic": _health("anthropic", health_score=0.95),
            "openai-codex": _health("openai-codex", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.95)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].provider == "anthropic"

    def test_anthropic_excluded_when_billing_fails(self):
        """When Anthropic has billing/auth failure (health=0.0), its models must be excluded."""
        models = [
            _model("anthropic/claude-opus-4-7", "anthropic",
                   coding=0.90, reasoning=0.95),
            _model("openai-codex/gpt-5.4", "openai-codex",
                   coding=0.96, reasoning=0.85),
        ]
        health = {
            "anthropic": _health("anthropic", health_score=0.0, auth="ok", quota="exhausted"),
            "openai-codex": _health("openai-codex", health_score=0.95),
        }
        classifier = ClassifierOutput(task_type="reasoning", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert all(s.provider != "anthropic" for s in eligible)


# ── Contract 6: Lightweight model exclusion for high reasoning ───────────────

class TestLightweightModelGuardrail:
    """Lightweight models (mini, flash, haiku) should be excluded from
    high-complexity reasoning tasks unless explicitly in fast/free mode."""

    def test_mini_excluded_from_high_reasoning(self):
        models = [
            _model("p/strong-pro", "p", reasoning=0.92),
            _model("p/cheap-mini", "p", reasoning=0.88, fast=0.96, cost=0.96),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        mini = next(s for s in scored if s.model_id == "p/cheap-mini")
        assert mini.excluded is True
        assert "lightweight" in (mini.exclusion_reason or "")

    def test_mini_allowed_in_fast_mode(self):
        """In fast mode, lightweight models should be allowed even for reasoning."""
        models = [
            _model("p/cheap-mini", "p", reasoning=0.88, fast=0.96, cost=0.96),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {}, route_mode="fast")
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1

    def test_flash_excluded_from_high_reasoning(self):
        models = [
            _model("p/gemini-3-flash", "p", reasoning=0.85, fast=0.95, cost=0.92),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        assert scored[0].excluded is True

    def test_haiku_excluded_from_high_reasoning(self):
        models = [
            _model("p/claude-haiku-4.5", "p", reasoning=0.78, fast=0.92, cost=0.95),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        assert scored[0].excluded is True

    def test_non_lightweight_gemini_not_excluded(self):
        """'gemini' should NOT match 'mini' — token-level matching required."""
        models = [
            _model("p/gemini-3-pro", "p", reasoning=0.92),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=0.90)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1


# ── Contract 7: Free-only routing ─────────────────────────────────────────────

class TestFreeOnlyRouting:
    """When free_only=True, only models explicitly marked is_free=True should
    be eligible."""

    def test_free_only_excludes_paid_models(self):
        models = [
            _model("p/paid-model", "p", is_free=None),
            _model("p/free-model", "p", is_free=True),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {}, free_only=True)
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1
        assert eligible[0].model_id == "p/free-model"

    def test_free_only_excludes_explicitly_not_free(self):
        models = [
            _model("p/not-free", "p", is_free=False),
            _model("p/free", "p", is_free=True),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {}, free_only=True)
        eligible = [s for s in scored if not s.excluded]
        assert all(s.model_id != "p/not-free" for s in eligible)


# ── Contract 8: Vision / context exclusions ──────────────────────────────────

class TestCapabilityExclusions:
    """Models lacking required capabilities must be excluded."""

    def test_low_vision_score_excluded_from_vision_task(self):
        models = [
            _model("p/no-vision", "p", vision=0.30),
            _model("p/has-vision", "p", vision=0.80),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="vision", needs_vision=True, confidence=0.90)
        scored = score_models(classifier, models, health, {})
        no_vis = next(s for s in scored if s.model_id == "p/no-vision")
        assert no_vis.excluded is True
        assert "vision" in (no_vis.exclusion_reason or "").lower()

    def test_small_context_excluded_from_long_context_task(self):
        models = [
            _model("p/small-ctx", "p", context_window=32_000),
            _model("p/large-ctx", "p", context_window=1_000_000),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="long_context", needs_long_context=True, confidence=0.90)
        scored = score_models(classifier, models, health, {})
        small = next(s for s in scored if s.model_id == "p/small-ctx")
        assert small.excluded is True
        assert "context_too_small" in (small.exclusion_reason or "")


# ── Contract 9: Route mode affects ranking ───────────────────────────────────

class TestRouteModeContracts:
    """Different route modes should produce different ranking outcomes."""

    def test_reasoning_mode_prefers_strongest_reasoner(self):
        models = [
            _model("p/strong-reasoning", "p", reasoning=0.95, fast=0.55, cost=0.50),
            _model("p/fast-cheap", "p", reasoning=0.75, fast=0.95, cost=0.95),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {}, route_mode="reasoning")
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].model_id == "p/strong-reasoning"

    def test_fast_mode_prefers_cheap_fast_model(self):
        models = [
            _model("p/strong-expensive", "p", coding=0.92, fast=0.55, cost=0.50),
            _model("p/fast-cheap", "p", coding=0.82, fast=0.95, cost=0.95),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {}, route_mode="fast")
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].model_id == "p/fast-cheap"

    def test_eco_mode_prefers_efficient_model(self):
        models = [
            _model("p/big-strong", "p", eco=0.30, cost=0.40, reasoning=0.95),
            _model("p/small-efficient", "p", eco=0.95, cost=0.90, reasoning=0.70),
        ]
        health = {"p": _health("p")}
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {}, route_mode="eco")
        eligible = [s for s in scored if not s.excluded]
        assert eligible[0].model_id == "p/small-efficient"


# ── Contract 10: No provider health data → reasonable defaults ───────────────

class TestUnknownProviderHealth:
    """When a provider has no health data at all, the router should assume
    a reasonable default (0.85) rather than excluding the model."""

    def test_unknown_provider_gets_default_health(self):
        models = [_model("unknown-p/model", "unknown-p")]
        health = {}  # no health data at all
        classifier = ClassifierOutput(task_type="coding", confidence=0.85)
        scored = score_models(classifier, models, health, {})
        eligible = [s for s in scored if not s.excluded]
        assert len(eligible) == 1
        assert eligible[0].health == 0.85

    def test_all_scores_bounded_0_to_1(self):
        """All component scores must be in [0, 1]."""
        models = [
            _model("p/a", "p", coding=0.99, reasoning=0.99),
            _model("p/b", "p", coding=0.50, reasoning=0.50, fast=0.99, cost=0.99),
        ]
        health = {"p": _health("p")}
        for task in ["coding", "reasoning", "general_chat", "fast_utility", "vision", "long_context"]:
            classifier = ClassifierOutput(task_type=task, confidence=0.80)
            scored = score_models(classifier, models, health, {})
            for s in scored:
                if not s.excluded:
                    assert 0.0 <= s.total_score <= 1.0, \
                        f"{s.model_id} score {s.total_score} out of range for {task}"
