from src.scorer import score_models
from src.types import ClassifierOutput


def _model(model_id, provider, cost, **features):
    return {
        "id": model_id,
        "provider": provider,
        "scores": {"reasoning": 0.7, "fast": 0.7, "cost": cost, "eco": 0.7, "vision": 0.9},
        "features": {"contextWindow": 300_000, **features},
        "availability": {"authed": True},
    }


MODELS = [
    _model("openrouter/some/model:free", "openrouter", 1.0, is_free=True),
    _model("openrouter/stealth", "openrouter", 1.0),
    _model("nvidia/free-model", "nvidia", 1.0),
    _model("openai/strong", "openai", 0.3),
    _model("openai/second", "openai", 0.4),
]
POLICY = {"routing": {"reasoning": {"candidates": [
    {"model": "anthropic/not-authed", "priority": 1},
    {"model": "openai/strong", "priority": 2},
    {"model": "openai/second", "priority": 3},
    {"model": "nvidia/free-model", "priority": 4},
]}}}


def _classifier():
    return ClassifierOutput(task_type="reasoning", complexity="medium", confidence=0.9, cost_profile="balanced")


def _eligible(route_mode, policy=POLICY, free_only=False, health=None):
    scored = score_models(
        classifier=_classifier(),
        models=MODELS,
        provider_health=health or {},
        learned_stats={},
        routing_policy=policy,
        route_mode=route_mode,
        free_only=free_only,
    )
    return [m.model_id for m in scored if not m.excluded]


def test_default_modes_follow_policy_order_and_drop_unlisted_models():
    for mode in ("auto", "balanced", "off", None):
        assert _eligible(mode) == ["openai/strong", "openai/second", "nvidia/free-model"]


def test_explicit_modes_can_still_pick_outside_policy():
    assert "openrouter/stealth" in _eligible("fast")
    assert "openrouter/stealth" in _eligible("eco")


def test_shared_free_tier_only_served_when_asked_for():
    assert "openrouter/some/model:free" not in _eligible("fast")
    assert "openrouter/some/model:free" not in _eligible("auto", policy=None)
    assert _eligible("free", free_only=True) == ["openrouter/some/model:free"]


def test_policy_with_no_authed_candidate_falls_back_to_scoring():
    policy = {"routing": {"reasoning": {"candidates": [{"model": "anthropic/not-authed", "priority": 1}]}}}
    assert "openai/strong" in _eligible("auto", policy=policy)
