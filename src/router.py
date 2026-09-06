"""
router.py — Main routing entrypoint for Nexus Router.

Usage:
    from nexus_router import Router

    router = Router()
    decision = router.route(classifier_output, pre_signals)
    print(decision.selected_model)
"""

import json
import os
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
from .db import ensure_schema, load_model_stats, load_model_metadata, write_decision
from .paths import ARTIFACTS_ROOT, REGISTRY_FILE, POLICIES_ROOT

ROOT = Path(__file__).parent.parent

ROUTING_FILE  = POLICIES_ROOT / "routing.yaml"
DEFAULT_CONFIDENCE_GATE_MIN = 0.65


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
        self._fitted_weights_cache: dict | None = None
        self._fitted_weights_cache_mtime: float | None = None
        # Temporary switch: keep routing.yaml on disk but ignore it unless explicitly enabled.
        # Re-enable by setting NEXUS_ROUTER_ENABLE_ROUTING_POLICY=1.
        self._use_routing_policy = os.getenv(
            "NEXUS_ROUTER_ENABLE_ROUTING_POLICY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        if self.persist:
            ensure_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        classifier: ClassifierOutput,
        pre_signals: Optional[PreSignals] = None,
        nexus_context: Optional[dict] = None,
        route_mode: Optional[str] = None,
        source_type: Optional[str] = None,
        source_tag: Optional[str] = None,
        mode: Optional[str] = None,
        message_text: Optional[str] = None,
        classifier_source: Optional[str] = None,
        free_only: bool = False,
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

        
        
        # 1. Determine effective route mode
        normalized_route_mode = str(route_mode or "auto").strip().lower()
        confidence_gate_threshold = _confidence_gate_threshold(self._routing)
        confidence_gate_triggered = False
        confidence_gate_reason: str | None = None

        # 2. Hard-force task based on explicit route_mode (precedence)
        if normalized_route_mode == "reasoning":
            effective_classifier = replace(classifier, task_type="reasoning", cost_profile="premium")
        elif normalized_route_mode == "fast":
            effective_classifier = replace(classifier, task_type="fast_utility", cost_profile="cheap")
        elif normalized_route_mode == "eco":
            effective_classifier = replace(classifier, task_type="eco", cost_profile="eco")
        elif normalized_route_mode == "free":
            effective_classifier = replace(classifier, task_type="fast_utility", cost_profile="cheap")
        else:
            # Only apply heuristics for auto/balanced
            effective_classifier = _adapt_classifier_for_light_chat(classifier, pre_signals)

        if (
            normalized_route_mode in {"auto", "balanced", "off"}
            and classifier.confidence < confidence_gate_threshold
        ):
            effective_classifier, confidence_gate_reason = _apply_confidence_gate(
                classifier=effective_classifier,
                original_classifier=classifier,
                min_confidence=confidence_gate_threshold,
            )
            confidence_gate_triggered = True
        # 4. Perform routing with hard-filtered free_only if requested
        # (Router.score will handle the is_free filtering)

        # Load fresh health and learned stats on each call
        # (cheap reads; health file is small, DB lookup is indexed)
        provider_health = load_provider_health()
        learned_stats   = load_model_stats()

        # Score all models
        routing_policy = self._routing if self._use_routing_policy else None
        policy_weights = (
            self._routing.get("scoring", {}).get("weights")
            if self._use_routing_policy
            else None
        )
        fitted_weights_bundle, fitted_weights_min_samples = self._load_fitted_weights_bundle()
        live_registry = self._registry
        if free_only or normalized_route_mode == "free":
            metadata = load_model_metadata()
            if metadata:
                live_registry = []
                for model in self._registry:
                    model_copy = dict(model)
                    features = dict(model_copy.get("features") or {})
                    override = metadata.get(model_copy.get("id", ""), {})
                    if "is_free" in override:
                        features["is_free"] = override.get("is_free")
                    model_copy["features"] = features
                    live_registry.append(model_copy)

        scored = score_models(
            classifier=effective_classifier,
            models=live_registry,
            provider_health=provider_health,
            learned_stats=learned_stats,
            policy_weights=policy_weights,
            fitted_weights_bundle=fitted_weights_bundle,
            fitted_weights_min_samples=fitted_weights_min_samples,
            routing_policy=routing_policy,
            route_mode=normalized_route_mode,
            free_only=(free_only or normalized_route_mode == "free"),
        )

        eligible = [s for s in scored if not s.excluded]
        excluded = [s for s in scored if s.excluded]

        if not eligible:
            raise RuntimeError(
                f"No eligible models for task_type={classifier.task_type!r}. "
                "Check auth, quota, and model registry."
            )

        primary  = eligible[0]
        fallbacks = _build_fallback_chain(eligible, primary_provider=primary.provider, limit=5)

        mechanisms = list(primary.score_mechanisms or [])
        if confidence_gate_triggered:
            mechanisms.append("confidence_gate")

        reason = _build_reason(primary, classifier, effective_classifier, pre_signals)
        if confidence_gate_triggered:
            reason.append(
                f"confidence gate fired (<{confidence_gate_threshold:.2f}); routed as '{effective_classifier.task_type}'"
            )
            if confidence_gate_reason:
                reason.append(confidence_gate_reason)
        if mechanisms:
            reason.append(f"mechanisms: {', '.join(dict.fromkeys(mechanisms))}")
        if normalized_route_mode in {"auto", "balanced", "fast", "reasoning", "eco", "free", "off"}:
            reason.append(f"route mode: {normalized_route_mode}")

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
            original_task_type=classifier.task_type,
            mechanisms=list(dict.fromkeys(mechanisms)),
            confidence_gate_triggered=confidence_gate_triggered,
            confidence_gate_threshold=confidence_gate_threshold,
            confidence_gate_reason=confidence_gate_reason,
            selected_component_scores=dict(primary.component_scores or {}),
            selected_component_contributions=dict(primary.component_contributions or {}),
        )

        # Persist if enabled. Skip ephemeral compiled prompt probes: they pollute
        # training data and are not actual user messages.
        should_persist = self.persist and (source_type or 'standalone') != 'compiled-prompt'
        if should_persist:
            ph = provider_health.get(primary.provider)
            if ph is None:
                from .types import ProviderHealth
                ph = ProviderHealth(provider=primary.provider)

            decision_id = write_decision(
                decision=decision,
                classifier=classifier,
                pre_signals=pre_signals,
                provider_health=ph,
                nexus_workflow_id=nexus_context.get("nexus_workflow_id"),
                nexus_step_id=nexus_context.get("nexus_step_id"),
                nexus_issue_id=nexus_context.get("nexus_issue_id"),
                nexus_project=nexus_context.get("nexus_project"),
                route_mode=normalized_route_mode,
                mode=mode,
                source_type=(source_type or 'standalone'),
                source_tag=source_tag,
                message_text=message_text,
                classifier_source=classifier_source,
            )
            decision.decision_id = decision_id

        return decision

    def _load_fitted_weights_bundle(self) -> tuple[dict | None, int]:
        cfg = ((self._routing or {}).get("scoring", {}) or {}).get("fitted_weights", {}) or {}
        enabled = bool(cfg.get("enabled", True))
        min_samples = max(1, int(cfg.get("min_samples_per_task", 40) or 40))
        if not enabled:
            return None, min_samples

        artifact_raw = str(cfg.get("artifact_path") or (ARTIFACTS_ROOT / "scorer" / "fitted_weights.active.json"))
        artifact_path = Path(os.path.expanduser(artifact_raw))
        if not artifact_path.exists():
            return None, min_samples

        try:
            stat = artifact_path.stat()
            mtime = float(stat.st_mtime)
        except OSError:
            return None, min_samples

        if self._fitted_weights_cache is not None and self._fitted_weights_cache_mtime == mtime:
            return self._fitted_weights_cache, min_samples

        try:
            payload = json.loads(artifact_path.read_text())
        except Exception:
            return None, min_samples

        if not isinstance(payload, dict):
            return None, min_samples

        self._fitted_weights_cache = payload
        self._fitted_weights_cache_mtime = mtime
        return payload, min_samples

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
    Remap only *very lightweight* general chat to fast_utility.

    Important: high-confidence classifier output should dominate routing.
    We only downgrade to fast_utility when the turn is genuinely tiny/simple.
    """
    if classifier.task_type != "general_chat":
        return classifier

    has_rich_signals = any([
        pre_signals.has_code,
        pre_signals.has_diff,
        pre_signals.has_logs,
        pre_signals.has_image,
        pre_signals.has_file_refs,
        pre_signals.has_url,
    ])

    # Be conservative with fast_utility downgrades.
    # If the classifier itself is weak/noisy, over-downgrading creates far more
    # damage than leaving the turn as general_chat.
    if (
        classifier.complexity == "low"
        and classifier.confidence < 0.35
        and not has_rich_signals
        and pre_signals.estimated_tokens <= 6
        and pre_signals.message_length <= 20
    ):
        return replace(classifier, task_type="fast_utility", cost_profile="cheap")

    # Extremely weak, tiny, structure-free chat can still downgrade.
    if (
        classifier.complexity in (None, "medium")
        and classifier.confidence <= 0.25
        and not has_rich_signals
        and pre_signals.estimated_tokens <= 4
        and pre_signals.message_length <= 12
    ):
        return replace(classifier, task_type="fast_utility", complexity="low", cost_profile="cheap")

    return classifier


def _confidence_gate_threshold(routing_policy: Optional[dict]) -> float:
    raw = ((routing_policy or {}).get("scoring", {}) or {}).get("confidence_gate", {}) or {}
    value = raw.get("min_confidence", DEFAULT_CONFIDENCE_GATE_MIN)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_CONFIDENCE_GATE_MIN
    return min(0.99, max(0.01, parsed))


def _apply_confidence_gate(
    classifier: ClassifierOutput,
    original_classifier: ClassifierOutput,
    min_confidence: float,
) -> tuple[ClassifierOutput, str]:
    target_task = "general_chat"
    reason = "safe_generalist"
    if original_classifier.needs_vision:
        target_task = "vision"
        reason = "safe_generalist_with_vision"
    elif original_classifier.needs_long_context:
        target_task = "long_context"
        reason = "safe_generalist_with_long_context"

    gated = replace(
        classifier,
        task_type=target_task,
        subtype=None,
        complexity="medium",
        cost_profile="balanced",
    )
    return (
        gated,
        f"confidence={original_classifier.confidence:.2f} below threshold={min_confidence:.2f} -> {target_task} ({reason})",
    )




def _build_fallback_chain(eligible_scores: list, *, primary_provider: str, limit: int = 5) -> list[str]:
    """Build fallback chain preferring provider diversity.

    Strategy:
    - first pass: pick highest-ranked models from providers different from primary
    - second pass: fill remaining slots from any provider by ranking order
    """
    selected: list[str] = []
    seen: set[str] = set()

    for score in eligible_scores[1:]:
        if len(selected) >= limit:
            break
        if score.provider == primary_provider:
            continue
        if score.model_id in seen:
            continue
        seen.add(score.model_id)
        selected.append(score.model_id)

    for score in eligible_scores[1:]:
        if len(selected) >= limit:
            break
        if score.model_id in seen:
            continue
        seen.add(score.model_id)
        selected.append(score.model_id)

    return selected


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

    # Pre-signals (keep these explicit so /route explain is understandable)
    if pre_signals.has_code:
        reasons.append("inline/fenced code detected")
    if pre_signals.has_diff:
        reasons.append("diff/patch detected")
    if pre_signals.has_image:
        reasons.append("image attachment detected")
    if pre_signals.has_logs:
        reasons.append("log/stack trace detected")
    if pre_signals.has_file_refs:
        reasons.append("file path/reference detected")
    if pre_signals.has_url:
        reasons.append("URL detected")
    if pre_signals.estimated_tokens > 50_000:
        reasons.append(f"large context estimated (~{pre_signals.estimated_tokens:,} tokens)")

    # Model selection rationale
    reasons.append(
        f"'{primary.model_id}' scored highest "
        f"(task_fit={primary.task_fit:.2f}, health={primary.health:.2f}, "
        f"pref={primary.preference:.2f}, learned={primary.learned:.2f})"
    )

    if primary.component_contributions:
        contrib = primary.component_contributions
        summary = (
            f"score contributions: "
            f"task_fit={float(contrib.get('task_fit', 0.0)):+.3f}, "
            f"health={float(contrib.get('health', 0.0)):+.3f}, "
            f"preference={float(contrib.get('preference', 0.0)):+.3f}, "
            f"learned={float(contrib.get('learned', 0.0)):+.3f}, "
            f"cost={float(contrib.get('cost', 0.0)):+.3f}, "
            f"speed={float(contrib.get('speed', 0.0)):+.3f}, "
            f"eco={float(contrib.get('eco', 0.0)):+.3f}, "
            f"quota_penalty={float(contrib.get('quota_penalty', 0.0)):+.3f}"
        )
        reasons.append(summary)

    if primary.model_preference_bump > 0:
        detail = (
            f"feedback preference bump +{primary.model_preference_bump:.3f} "
            f"from {primary.model_preference_samples} recent samples"
        )
        if primary.model_preference_reason_tag:
            detail += f" ({primary.model_preference_reason_tag})"
        reasons.append(detail)

    if classifier.cost_profile == "cheap":
        reasons.append("cost-sensitive routing applied")
    elif classifier.cost_profile == "premium":
        reasons.append("premium quality routing applied")

    if classifier.detected_language and classifier.detected_language not in ("en", "unknown"):
        reasons.append(f"non-English prompt detected ({classifier.detected_language})")

    return reasons
