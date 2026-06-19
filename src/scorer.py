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

from datetime import datetime, timezone
from typing import Optional

from .types import ClassifierOutput, ModelScore, ProviderHealth

# ── Scoring weights (can be overridden from routing.yaml) ────────────────────
DEFAULT_WEIGHTS = {
    "task_fit":   0.50,
    "health":     0.20,
    "preference": 0.10,
    "learned":    0.10,
    "cost":       0.05,
    "speed":      0.03,
    "eco":        0.02,
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
    "eco":             "eco",
}

# Minimum vision score to be eligible for vision tasks
VISION_MIN_SCORE = 0.50

# Context window thresholds for long_context task
LONG_CONTEXT_MIN_TOKENS = 200_000

# Lightweight models are fine for cheap/fast turns, but should not win
# high-complexity reasoning unless the user explicitly picked fast/free routing.
LIGHTWEIGHT_MODEL_HINTS = {"mini", "flash", "haiku"}

# Complexity → scoring adjustments for preference dimension
COMPLEXITY_PREFERENCE = {
    "low":    {"fast": +0.10, "cost": +0.10, "coding": -0.05, "reasoning": -0.05},
    "medium": {},
    "high":   {"coding": +0.10, "reasoning": +0.10, "review": +0.10, "fast": -0.10},
}

# cost_profile → weighting adjustment
COST_PROFILE_WEIGHT = {
    "cheap":    {"cost": 0.20, "speed": 0.05, "eco": 0.10, "task_fit": 0.35, "health": 0.20, "preference": 0.05, "learned": 0.05},
    "balanced": DEFAULT_WEIGHTS,
    "premium":  {"task_fit": 0.60, "health": 0.15, "preference": 0.10, "learned": 0.10, "cost": 0.02, "speed": 0.02, "eco": 0.01},
    "reasoning": {"task_fit": 0.62, "health": 0.14, "preference": 0.10, "learned": 0.10, "cost": 0.01, "speed": 0.02, "eco": 0.01},
    "eco":      {"eco": 0.45, "cost": 0.20, "task_fit": 0.15, "health": 0.10, "preference": 0.05, "learned": 0.05, "speed": 0.00},
}


def score_models(
    classifier: ClassifierOutput,
    models: list[dict],
    provider_health: dict[str, ProviderHealth],
    learned_stats: dict[str, dict],
    policy_weights: Optional[dict] = None,
    routing_policy: Optional[dict] = None,
    route_mode: Optional[str] = None,
    free_only: bool = False,
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
    # Build preference config: merge routing_policy overrides over safe defaults.
    # Feedback is ALWAYS enabled regardless of routing_policy state.
    _preference_defaults: dict = {
        "enabled": True,
        "max_bump": 0.08,
        "max_penalty": 0.30,
        "min_samples": 3,
        "decay_window_days": 30,
        "hard_exclude_enabled": True,
        "hard_exclude_min_samples": 10,
        "hard_exclude_centered_score_lte": -0.8,
    }
    _policy_preference_cfg = ((routing_policy or {}).get("feedback", {}) or {}).get("model_preference", {}) or {}
    preference_cfg = {**_preference_defaults, **{k: v for k, v in _policy_preference_cfg.items() if v is not None}}

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

        # Free filter: only allow models explicitly marked free.
        if free_only and features.get("is_free") is not True:
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=0.0, health=0.0,
                preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                excluded=True, exclusion_reason="not_free",
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

        # High-complexity reasoning guardrail: do not let cheap/lightweight
        # variants win by small learned/cost/speed differences. Explicit
        # fast/free modes are the opt-out when the user really wants cheap.
        mode = (route_mode or "").strip().lower()
        if (
            classifier.task_type == "reasoning"
            and classifier.complexity == "high"
            and mode not in {"fast", "free"}
            and _is_lightweight_model(model_id)
        ):
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=0.0, health=0.0,
                preference=0.0, learned=0.0, cost=0.0, speed=0.0,
                excluded=True, exclusion_reason="lightweight_excluded_for_high_reasoning",
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

        # Generic quota pressure: if provider health says quota is low,
        # make cheap/auto style routing less eager to burn that provider.
        quota_penalty = _quota_pressure(ph, classifier, route_mode)

        # Preference: rank in routing policy (higher = earlier in list = more preferred)
        preference = _preference_score(model_id, preference_order)

        # Learned: derived from historical success rate + override avoidance
        learned = _learned_score(model_id, classifier.task_type, learned_stats)
        model_preference_bump, preference_meta = _model_preference_bump(
            model_id,
            classifier.task_type,
            learned_stats,
            preference_cfg,
        )
        learned = round(max(0.0, min(1.0, learned + model_preference_bump)), 4)

        # Cost: direct from model capability profile
        cost_score = scores_raw.get("cost", 0.65)
        # cost_profile adjustments
        if classifier.cost_profile == "cheap":
            cost_score = min(1.0, cost_score + 0.10)
        elif classifier.cost_profile == "premium":
            cost_score = max(0.0, cost_score - 0.10)
        elif classifier.cost_profile == "reasoning":
            # Slightly de-emphasize pure cost so stronger models can win when task fit is high.
            cost_score = max(0.0, cost_score - 0.05)

        # Speed: direct from model capability profile
        speed_score = scores_raw.get("fast", 0.70)
        reasoning_score = scores_raw.get("reasoning", 0.70)

        hard_exclude_enabled = bool(preference_cfg.get("hard_exclude_enabled", True))
        hard_exclude_min_samples = max(1, int(preference_cfg.get("hard_exclude_min_samples", 10) or 10))
        hard_exclude_centered_lte = float(preference_cfg.get("hard_exclude_centered_score_lte", -0.8) or -0.8)
        centered_score = preference_meta.get("centered_score")
        decay_factor = preference_meta.get("decay_factor", 1.0)
        decayed_centered_score = None
        if isinstance(centered_score, (int, float)) and isinstance(decay_factor, (int, float)):
            decayed_centered_score = float(centered_score) * float(decay_factor)
        reason_tag = str(preference_meta.get("top_reason_tag") or "")
        if (
            hard_exclude_enabled
            and int(preference_meta.get("samples", 0) or 0) >= hard_exclude_min_samples
            and isinstance(decayed_centered_score, (int, float))
            and float(decayed_centered_score) <= hard_exclude_centered_lte
        ):
            excluded.append(ModelScore(
                model_id=model_id, provider=provider,
                total_score=0.0, task_fit=round(task_fit, 4), health=round(health, 4),
                preference=round(preference, 4), learned=round(learned, 4),
                model_preference_bump=round(model_preference_bump, 4),
                model_preference_samples=int(preference_meta.get("samples", 0)),
                model_preference_reason_tag=preference_meta.get("top_reason_tag"),
                model_preference_centered_score=round(float(centered_score), 4),
                cost=round(cost_score, 4), speed=round(speed_score, 4),
                excluded=True,
                exclusion_reason=(
                    f"feedback_hard_exclude:{classifier.task_type}:samples={int(preference_meta.get('samples', 0))}"
                    f":centered={float(centered_score):.2f}"
                    f":decayed={float(decayed_centered_score):.2f}"
                    + (f":reason={reason_tag}" if reason_tag else "")
                ),
            ))
            continue

        # ── Composite score ──────────────────────────────────────────────────
        total = (
            task_fit   * weights["task_fit"]
            + health   * weights["health"]
            + preference * weights["preference"]
            + learned  * weights["learned"]
            + cost_score * weights["cost"]
            + speed_score * weights["speed"]
        )
        total -= quota_penalty

        mode = (route_mode or "").strip().lower()
        if mode == "fast":
            total += _fast_mode_correction(task_fit=task_fit, cost_score=cost_score, speed_score=speed_score)
        elif mode == "reasoning":
            total = _reasoning_mode_total(
                task_fit=task_fit,
                reasoning_score=reasoning_score,
                health=health,
                learned=learned,
                preference=preference,
                speed_score=speed_score,
                quota_penalty=quota_penalty,
            )
        elif mode == "eco":
            total = _eco_mode_total(
                task_fit=task_fit,
                eco_score=scores_raw.get("eco", 0.50),
                cost_score=cost_score,
                health=health,
                learned=learned,
                preference=preference,
                quota_penalty=quota_penalty,
            )

        eligible.append(ModelScore(
            model_id=model_id,
            provider=provider,
            total_score=round(total, 4),
            task_fit=round(task_fit, 4),
            health=round(health, 4),
            preference=round(preference, 4),
            learned=round(learned, 4),
            model_preference_bump=round(model_preference_bump, 4),
            model_preference_samples=int(preference_meta.get("samples", 0)),
            model_preference_reason_tag=preference_meta.get("top_reason_tag"),
            model_preference_centered_score=preference_meta.get("centered_score"),
            cost=round(cost_score, 4),
            speed=round(speed_score, 4),
            eco=round(scores_raw.get("eco", 0.50), 4),
        ))

    # Sort eligible descending by total score
    eligible.sort(key=lambda x: x.total_score, reverse=True)
    return eligible + excluded


def _resolve_weights(classifier: ClassifierOutput, policy_weights: Optional[dict]) -> dict:
    """Resolve scoring weights from cost_profile and optional policy override."""
    if policy_weights:
        return policy_weights
    return COST_PROFILE_WEIGHT.get(classifier.cost_profile, DEFAULT_WEIGHTS)


def _quota_pressure(
    provider_health: Optional[ProviderHealth],
    classifier: ClassifierOutput,
    route_mode: Optional[str],
) -> float:
    """
    Generic quota-aware penalty.

    Goal:
    - when a provider is on low remaining quota, avoid spending it on cheap chatter
    - still allow it to win for higher-value work if task fit is clearly better
    """
    if not provider_health:
        return 0.0

    quota = (provider_health.quota or "unknown").strip().lower()
    ratio = provider_health.quota_remaining_ratio
    penalty = 0.0

    if quota == "exhausted":
        return 1.0

    if ratio is not None:
        try:
            ratio = max(0.0, min(1.0, float(ratio)))
        except (TypeError, ValueError):
            ratio = None

    if ratio is not None:
        if ratio <= 0.10:
            penalty += 0.25
        elif ratio <= 0.25:
            penalty += 0.16
        elif ratio <= 0.50:
            penalty += 0.06
    elif quota == "low":
        penalty += 0.12

    mode = (route_mode or "").strip().lower()
    if mode in {"auto", "fast"}:
        penalty *= 1.35
    elif mode == "balanced":
        penalty *= 1.10

    if classifier.task_type in {"fast_utility", "general_chat", "summarization"}:
        penalty *= 1.25
    elif classifier.task_type in {"coding", "code_review", "reasoning", "long_context"}:
        penalty *= 0.75

    return round(min(max(penalty, 0.0), 0.45), 4)


def _fast_mode_correction(task_fit: float, cost_score: float, speed_score: float) -> float:
    """
    Provider-agnostic correction for fast mode.

    Intuition:
    - Reward higher cost/speed scores beyond the neutral midpoint (0.5)
    - Apply a guardrail so very low task-fit models are less likely to win

    This avoids vendor-specific rules while still reducing "same strong model always wins"
    behavior in fast mode.
    """
    cost_delta = cost_score - 0.5
    speed_delta = speed_score - 0.5
    task_fit_guardrail = max(0.0, 0.68 - task_fit)

    correction = (0.28 * cost_delta) + (0.12 * speed_delta) - (0.08 * task_fit_guardrail)
    return correction


def _reasoning_mode_total(
    task_fit: float,
    reasoning_score: float,
    health: float,
    learned: float,
    preference: float,
    speed_score: float,
    quota_penalty: float,
) -> float:
    """
    Dedicated scoring path for explicit reasoning mode.

    Principle:
    - explicit `route_mode=reasoning` should noticeably change the ranking
    - reasoning score should dominate the outcome
    - health / learned history are secondary guardrails
    - task fit matters, but much less once we are already in explicit reasoning mode
    - speed and generic preference should be almost negligible
    """
    return (
        (0.68 * reasoning_score)
        + (0.12 * health)
        + (0.10 * learned)
        + (0.06 * task_fit)
        + (0.02 * preference)
        + (0.01 * speed_score)
        - quota_penalty
    )


def _eco_mode_total(
    task_fit: float,
    eco_score: float,
    cost_score: float,
    health: float,
    learned: float,
    preference: float,
    quota_penalty: float,
) -> float:
    """
    Dedicated scoring path for explicit eco mode.

    Principle:
    - eco score (efficiency/size) and cost score dominate
    - task fit is a floor/filter
    - health/learned are secondary
    """
    return (
        (0.40 * eco_score)
        + (0.30 * cost_score)
        + (0.10 * health)
        + (0.10 * task_fit)
        + (0.05 * learned)
        + (0.05 * preference)
        - quota_penalty
    )


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


def _model_preference_bump(
    model_id: str,
    task_type: str,
    learned_stats: dict[str, dict],
    preference_cfg: dict,
) -> tuple[float, dict[str, object]]:
    # Feedback is always enabled; preference_cfg always has 'enabled': True from score_models.
    # Guard: if somehow called with an empty/disabled cfg, apply sensible defaults.
    if not preference_cfg.get("enabled", True):
        return 0.0, {"samples": 0}

    stats = learned_stats.get(model_id) or {}
    feedback = stats.get("feedback_preference")
    if not isinstance(feedback, dict):
        return 0.0, {"samples": 0}

    task_feedback = feedback.get(task_type) or feedback.get("*")
    if not isinstance(task_feedback, dict):
        return 0.0, {"samples": 0}

    samples = int(task_feedback.get("samples") or 0)
    min_samples = max(1, int(preference_cfg.get("min_samples", 3) or 3))
    if samples < min_samples:
        return 0.0, {"samples": samples}

    score = float(task_feedback.get("score") or 0.0)

    # Asymmetric: penalise harder than reward (bad routing costs more than missing a good model)
    max_bump = float(preference_cfg.get("max_bump", 0.08) or 0.08)
    max_penalty = float(preference_cfg.get("max_penalty", 0.30) or 0.30)
    decay_days = max(1, int(preference_cfg.get("decay_window_days", 30) or 30))
    last_feedback_at = _parse_iso_utc(task_feedback.get("last_feedback_at"))
    if last_feedback_at is None:
        decay_factor = 1.0
    else:
        age_days = max(0.0, (datetime.now(timezone.utc) - last_feedback_at).total_seconds() / 86400.0)
        decay_factor = max(0.0, 1.0 - min(age_days / decay_days, 1.0))

    sample_factor = min(1.0, samples / max(min_samples * 3, 1))

    # Convert [0,1] feedback score into centered signal [-1,+1].
    # < 0 means preference penalty, > 0 means preference boost.
    centered = max(-1.0, min(1.0, (score - 0.5) * 2.0))
    if centered >= 0:
        bump = min(max_bump, max_bump * centered * decay_factor * sample_factor)
    else:
        bump = max(-max_penalty, -max_penalty * abs(centered) * decay_factor * sample_factor)

    return round(bump, 4), {
        "samples": samples,
        "top_reason_tag": task_feedback.get("top_reason_tag"),
        "decay_factor": round(decay_factor, 4),
        "centered_score": round(centered, 4),
    }


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_lightweight_model(model_id: str) -> bool:
    """Return True for explicit lightweight model-name tokens.

    Use token matching instead of substring matching so `gemini` does not match
    `mini`.
    """
    import re

    tokens = {token for token in re.split(r"[^a-z0-9]+", model_id.lower()) if token}
    return bool(tokens & LIGHTWEIGHT_MODEL_HINTS)
