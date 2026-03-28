"""
router.py — Main routing entrypoint for Nexus Router.

Usage:
    from nexus_router import Router

    router = Router()
    decision = router.route(classifier_output, pre_signals)
    print(decision.selected_model)
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .types import ClassifierOutput, PreSignals, RoutingDecision
from .health import load_provider_health
from .scorer import score_models
from .db import ensure_schema, load_model_stats, write_decision

ROOT = Path(__file__).parent.parent

REGISTRY_FILE = ROOT / "catalog/normalized/models.json"
ROUTING_FILE  = ROOT / "policies/routing.yaml"


class Router:
    """
    Main Nexus Router.

    Typical flow:
    1. Load model registry + routing policy once at startup
    2. For each turn: call route(classifier, pre_signals)
    3. Optionally call record_outcome(decision_id, ...) after turn completes
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        routing_path: Optional[Path]  = None,
        persist: bool = True,
    ):
        """
        Args:
            registry_path: Path to normalized models.json (default: auto)
            routing_path:  Path to routing.yaml (default: auto)
            persist:       Whether to write decisions to SQLite (default: True)
        """
        self.registry_path = registry_path or REGISTRY_FILE
        self.routing_path  = routing_path or ROUTING_FILE
        self.persist       = persist

        self._registry     = self._load_registry()
        self._routing      = self._load_routing()

        if self.persist:
            ensure_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        classifier: ClassifierOutput,
        pre_signals: Optional[PreSignals] = None,
        nexus_context: Optional[dict] = None,
        route_mode: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a request to the best available model.

        Args:
            classifier:    Structured output from Nexus classifier agent
            pre_signals:   Pre-linguistic signals (optional, for logging)
            nexus_context: Optional dict with nexus_workflow_id, nexus_step_id,
                           nexus_issue_id, nexus_project

        Returns:
            RoutingDecision with selected_model, fallbacks, score, reason
        """
        pre_signals  = pre_signals or PreSignals()
        nexus_context = nexus_context or {}

        effective_classifier = _adapt_classifier_for_light_chat(classifier, pre_signals)
        effective_route_mode = (route_mode or "auto").strip().lower()
        if effective_route_mode == "reasoning":
            if effective_classifier.task_type in {"general_chat", "fast_utility", "summarization"}:
                effective_classifier = replace(
                    effective_classifier,
                    task_type="reasoning",
                    cost_profile="premium",
                )
            else:
                effective_classifier = replace(effective_classifier, cost_profile="premium")
        elif effective_route_mode == "fast":
            effective_classifier = replace(effective_classifier, cost_profile="cheap")

        # Load fresh health and learned stats on each call
        # (cheap reads; health file is small, DB lookup is indexed)
        provider_health = load_provider_health()
        learned_stats   = load_model_stats()

        # Score all models
        policy_weights = self._routing.get("scoring", {}).get("weights")
        scored = score_models(
            classifier=effective_classifier,
            models=self._registry,
            provider_health=provider_health,
            learned_stats=learned_stats,
            policy_weights=policy_weights,
            routing_policy=self._routing,
        )

        eligible = [s for s in scored if not s.excluded]
        excluded = [s for s in scored if s.excluded]

        if not eligible:
            raise RuntimeError(
                f"No eligible models for task_type={classifier.task_type!r}. "
                "Check auth, quota, and model registry."
            )

        primary  = eligible[0]
        fallbacks = [s.model_id for s in eligible[1:4]]  # top 3 fallbacks

        reason = _build_reason(primary, classifier, effective_classifier, pre_signals)
        if effective_route_mode in {"fast", "reasoning", "off"}:
            reason.append(f"route mode: {effective_route_mode}")

        decision = RoutingDecision(
            task_type=effective_classifier.task_type,
            confidence=classifier.confidence,
            selected_model=primary.model_id,
            selected_provider=primary.provider,
            fallbacks=fallbacks,
            score=primary.total_score,
            reason=reason,
            excluded_models=[
                {"model": e.model_id, "reason": e.exclusion_reason}
                for e in excluded[:10]  # cap to keep output sane
            ],
            all_scores=scored,
        )

        # Persist if enabled
        if self.persist:
            ph = provider_health.get(primary.provider)
            if ph is None:
                from .types import ProviderHealth
                ph = ProviderHealth(provider=primary.provider)

            write_decision(
                decision=decision,
                classifier=classifier,
                pre_signals=pre_signals,
                provider_health=ph,
                nexus_workflow_id=nexus_context.get("nexus_workflow_id"),
                nexus_step_id=nexus_context.get("nexus_step_id"),
                nexus_issue_id=nexus_context.get("nexus_issue_id"),
                nexus_project=nexus_context.get("nexus_project"),
            )

        return decision

    def explain(self, decision: RoutingDecision) -> str:
        """Return a human-readable explanation of a routing decision."""
        lines = [
            f"Task:     {decision.task_type}",
            f"Model:    {decision.selected_model}  (score={decision.score:.3f})",
            f"Fallbacks: {', '.join(decision.fallbacks) or 'none'}",
            f"Reason:",
        ]
        for r in decision.reason:
            lines.append(f"  - {r}")
        if decision.excluded_models:
            lines.append(f"Excluded: {len(decision.excluded_models)} models")
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_registry(self) -> list[dict]:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found at {self.registry_path}. "
                "Run: python src/generate_registry.py"
            )
        with open(self.registry_path) as f:
            data = json.load(f)
        return data.get("models", [])

    def _load_routing(self) -> dict:
        if not self.routing_path.exists():
            return {}
        if not HAS_YAML:
            return {}
        with open(self.routing_path) as f:
            return yaml.safe_load(f) or {}


# ── Reason builder ────────────────────────────────────────────────────────────

def _adapt_classifier_for_light_chat(
    classifier: ClassifierOutput,
    pre_signals: PreSignals,
) -> ClassifierOutput:
    """
    Remap lightweight general chat to fast_utility so greetings / tiny asks do not
    burn a premium model by default.

    Rules:
    - explicit low-complexity general_chat -> fast_utility
    - low-confidence generic general_chat with no rich signals and short-ish text
      also downgrades to fast_utility
    """
    if classifier.task_type != "general_chat":
        return classifier

    has_rich_signals = any([
        pre_signals.has_code,
        pre_signals.has_diff,
        pre_signals.has_logs,
        pre_signals.has_image,
    ])

    if classifier.complexity == "low" and not has_rich_signals:
        return replace(classifier, task_type="fast_utility", cost_profile="cheap")

    if (
        classifier.complexity in (None, "medium")
        and classifier.confidence <= 0.65
        and not has_rich_signals
        and pre_signals.estimated_tokens <= 120
        and pre_signals.message_length <= 500
    ):
        return replace(classifier, task_type="fast_utility", complexity="low", cost_profile="cheap")

    return classifier


def _build_reason(
    primary,
    classifier: ClassifierOutput,
    effective_classifier: ClassifierOutput,
    pre_signals: PreSignals,
) -> list[str]:
    reasons = []

    # Task classification
    reasons.append(f"task classified as '{classifier.task_type}' (confidence={classifier.confidence:.0%})")

    if effective_classifier.task_type != classifier.task_type:
        reasons.append(
            f"adapted to '{effective_classifier.task_type}' for lightweight chat"
        )

    if classifier.subtype:
        reasons.append(f"subtype: {classifier.subtype}")

    if classifier.complexity != "medium":
        reasons.append(f"complexity: {classifier.complexity}")

    # Pre-signals
    if pre_signals.has_code:
        reasons.append("code content detected")
    if pre_signals.has_diff:
        reasons.append("diff/patch detected")
    if pre_signals.has_image:
        reasons.append("image attachment detected")
    if pre_signals.has_logs:
        reasons.append("log/stack trace detected")
    if pre_signals.estimated_tokens > 50_000:
        reasons.append(f"large context estimated (~{pre_signals.estimated_tokens:,} tokens)")

    # Model selection rationale
    reasons.append(
        f"'{primary.model_id}' scored highest "
        f"(task_fit={primary.task_fit:.2f}, health={primary.health:.2f}, "
        f"pref={primary.preference:.2f}, learned={primary.learned:.2f})"
    )

    if classifier.cost_profile == "cheap":
        reasons.append("cost-sensitive routing applied")
    elif classifier.cost_profile == "premium":
        reasons.append("premium quality routing applied")

    if classifier.detected_language and classifier.detected_language not in ("en", "unknown"):
        reasons.append(f"non-English prompt detected ({classifier.detected_language})")

    return reasons
