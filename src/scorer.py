"""
scorer.py — Model scoring engine for Nexus Router.

Takes:
- ClassifierOutput (from Nexus classifier agent)
- List of candidate models (from normalized registry)
- Provider health snapshot
- Learned model stats (from SQLite)
- Policy weights (from routing.yaml)

Returns:
- Ranked list of ModelScore objects
"""

from typing import Optional

from .types import ClassifierOutput, ModelScore, ProviderHealth

# ── Scoring weights (can be overridden from routing.yaml) ────────────────────
DEFAULT_WEIGHTS = {
    "task_fit":   0.50,
    "health":     0.20,
    "preference": 0.10,
    "learned":    0.10,
    "cost":       0.05,
    "speed":      0.05,
}

# Models scoring below this health score are excluded entirely
HEALTH_HARD_CUTOFF = 0.30

# Task type → model capability dimension used for task_fit
TASK_TO_DIMENSION = {
    "coding":          "coding",
    "code_review":     "review",
    "reasoning":       "reasoning",
    "summarization":   "summarize",
    "fast_utility":    "fast",
    "long_context":    "context",
    "vision":          "vision",
    "general_chat":    "reasoning",
}

# Minimum vision score to be eligible for vision tasks
VISION_MIN_SCORE = 0.50

# Context window thresholds for long_context task
LONG_CONTEXT_MIN_TOKENS = 200_000

# Complexity → scoring adjustments for preference dimension
COMPLEXITY_PREFERENCE = {
    "low":    {"fast": +0.10, "cost": +0.10, "coding": -0.05, "reasoning": -0.05},
    "medium": {},
    "high":   {"coding": +0.10, "reasoning": +0.10, "review": +0.10, "fast": -0.10},
}

# cost_profile → weighting adjustment
COST_PROFILE_WEIGHT = {
    "cheap":    {"cost": 0.20, "speed": 0.10, "task_fit": 0.40, "health": 0.20, "preference": 0.05, "learned": 0.05},
    "balanced": DEFAULT_WEIGHTS,
    "premium":  {"task_fit": 0.60, "health": 0.15, "preference": 0.10, "learned": 0.10, "cost": 0.02, "speed": 0.03},
}


def score_models(
    classifier: ClassifierOutput,
    models: list[dict],
    provider_health: dict[str, ProviderHealth],
    learned_stats: dict[str, dict],
    policy_weights: Optional[dict] = None,
    routing_policy: Optional[dict] = None,
    route_mode: Optional[str] = None,
) -> list[ModelScore]:
    """
    Score all candidate models and return a ranked list.

    Args:
        classifier:      Structured classifier output
        models:          List of normalized model dicts from registry
        provider_health: Dict of provider_id → ProviderHealth
        learned_stats:   Dict of model_id → stats from DB
        policy_weights:  Optional weight override (from routing.yaml)
        routing_policy:  Optional routing policy dict (for preference ordering)

    Returns:
        List of ModelScore sorted by total_score descending.
        Excluded models are appended at the end with excluded=True.
    """
    weights = _resolve_weights(classifier, policy_weights)
    task_dim = TASK_TO_DIMENSION.get(classifier.task_type, "reasoning")
    preference_order = _build_preference_order(classifier.task_type, routing_policy)

    eligible  = []
    excluded  = []

    for model_dict in models:
        model_id  = model_dict["id"]
        provider  = model_dict["provider"]
        scores_raw = model_dict.get("scores", {})
        features  = model_dict.get("features", {})
        avail     = model_dict.get("availability", {})

        # ── Hard exclusions ──────────────────────────────────────────────────

        # Must be available/authed
        if not avail.get("authed", False):
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=0.0, health=0.0,
                preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                excluded=True, exclusion_reason="not_authed",
            ))
            continue

        # Vision task: must have vision support
        if classifier.task_type == "vision" and scores_raw.get("vision", 0.0) < VISION_MIN_SCORE:
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=0.0, health=0.0,
                preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                excluded=True, exclusion_reason="no_vision_support",
            ))
            continue

        # Long context: must have large enough context window
        if classifier.needs_long_context or classifier.task_type == "long_context":
            ctx = features.get("contextWindow", 0)
            if ctx < LONG_CONTEXT_MIN_TOKENS:
                excluded.append(ModelScore(
                    model_id=model_id, provider=provider,
                    total_score=0.0, task_fit=0.0, health=0.0,
                    preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                    excluded=True, exclusion_reason=f"context_too_small:{ctx}",
                ))
                continue

        # Provider health hard cutoff
        ph = provider_health.get(provider)
        health_score = ph.health_score if ph else 0.85  # assume reasonable if no data
        if health_score < HEALTH_HARD_CUTOFF:
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=0.0, health=health_score,
                preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                excluded=True, exclusion_reason=f"health_too_low:{health_score:.2f}",
            ))
            continue

        # ── Component scores ─────────────────────────────────────────────────

        # Task fit: primary dimension score for this task type
        task_fit = scores_raw.get(task_dim, 0.70)

        # Complexity adjustment to task fit
        complexity_adj = COMPLEXITY_PREFERENCE.get(classifier.complexity, {})
        task_fit = min(1.0, task_fit + complexity_adj.get(task_dim, 0.0))

        # Health component: direct from provider health
        health = health_score

        # Preference: rank in routing policy (higher = earlier in list = more preferred)
        preference = _preference_score(model_id, preference_order)

        # Learned: derived from historical success rate + override avoidance
        learned = _learned_score(model_id, classifier.task_type, learned_stats)

        # Cost: direct from model capability profile
        cost_score = scores_raw.get("cost", 0.65)
        # cost_profile adjustments
        if classifier.cost_profile == "cheap":
            cost_score = min(1.0, cost_score + 0.10)
        elif classifier.cost_profile == "premium":
            cost_score = max(0.0, cost_score - 0.10)

        # Speed: direct from model capability profile
        speed_score = scores_raw.get("fast", 0.70)

        # ── Composite score ──────────────────────────────────────────────────
        total = (
            task_fit   * weights["task_fit"]
            + health   * weights["health"]
            + preference * weights["preference"]
            + learned  * weights["learned"]
            + cost_score * weights["cost"]
            + speed_score * weights["speed"]
        )

        if (route_mode or "").strip().lower() == "fast":
            total += _fast_mode_correction(task_fit=task_fit, cost_score=cost_score, speed_score=speed_score)

        eligible.append(ModelScore(
            model_id=model_id,
            provider=provider,
            total_score=round(total, 4),
            task_fit=round(task_fit, 4),
            health=round(health, 4),
            preference=round(preference, 4),
            learned=round(learned, 4),
            cost=round(cost_score, 4),
            speed=round(speed_score, 4),
        ))

    # Sort eligible descending by total score
    eligible.sort(key=lambda x: x.total_score, reverse=True)
    return eligible + excluded


def _resolve_weights(classifier: ClassifierOutput, policy_weights: Optional[dict]) -> dict:
    """Resolve scoring weights from cost_profile and optional policy override."""
    if policy_weights:
        return policy_weights
    return COST_PROFILE_WEIGHT.get(classifier.cost_profile, DEFAULT_WEIGHTS)


def _fast_mode_correction(task_fit: float, cost_score: float, speed_score: float) -> float:
    """
    Provider-agnostic correction for fast mode.

    Intuition:
    - Reward higher cost/speed scores beyond the neutral midpoint (0.5)
    - Apply a small guardrail so very low task-fit models are less likely to win

    This avoids vendor-specific rules while still reducing "same strong model always wins"
    behavior in fast mode.
    """
    cost_delta = cost_score - 0.5
    speed_delta = speed_score - 0.5
    task_fit_guardrail = max(0.0, 0.68 - task_fit)

    correction = (0.28 * cost_delta) + (0.12 * speed_delta) - (0.08 * task_fit_guardrail)
    return correction


def _build_preference_order(task_type: str, routing_policy: Optional[dict]) -> list[str]:
    """Extract ordered model list from routing policy for a given task type."""
    if not routing_policy:
        return []
    candidates = routing_policy.get("routing", {}).get(task_type, {}).get("candidates", [])
    return [c["model"] for c in sorted(candidates, key=lambda c: c.get("priority", 99))]


def _preference_score(model_id: str, preference_order: list[str]) -> float:
    """
    Score a model based on its position in the routing policy preference list.
    First = 1.0, second = 0.85, third = 0.70, etc. Not listed = 0.50.
    """
    if not preference_order:
        return 0.50
    try:
        idx = preference_order.index(model_id)
        return max(0.50, 1.0 - idx * 0.10)
    except ValueError:
        return 0.50


def _learned_score(model_id: str, task_type: str, learned_stats: dict[str, dict]) -> float:
    """
    Derive a learned score from historical routing outcomes.

    Factors:
    - Overall success rate (weighted most)
    - Task-specific selection frequency (as proxy for performance)
    - Override rate (high overrides = model wasn't great for this task)
    """
    stats = learned_stats.get(model_id)
    if not stats or not stats.get("total_selected"):
        return 0.75  # neutral prior for unseen models

    success_rate   = stats.get("success_rate") or 0.75
    override_rate  = stats.get("total_override", 0) / max(stats["total_selected"], 1)

    # Base from success rate
    score = success_rate

    # Penalise for high override rate (users kept switching away)
    score -= min(override_rate * 0.20, 0.20)

    return round(max(0.0, min(1.0, score)), 4)
