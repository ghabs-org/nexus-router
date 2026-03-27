"""
openclaw_adapter.py — OpenClaw integration adapter for Nexus Router.

This is the thin integration layer between the Nexus Router and OpenClaw.
It is designed to be called:
  - From an OpenClaw plugin (pre-turn hook)
  - From a Nexus workflow step
  - Or directly from any Python code that has access to OpenClaw state

Typical usage in an OpenClaw plugin:

    from nexus_router.openclaw_adapter import route_turn

    async def before_turn(context):
        decision = route_turn(
            message=context.message,
            has_image=context.has_image,
        )
        context.set_model(decision.selected_model)
        context.set_fallbacks(decision.fallbacks)
"""

import json
import subprocess
import time
from typing import Optional

from .router import Router
from .classifier import extract_pre_signals, heuristic_classify, build_classification_prompt, parse_classifier_response, SYSTEM_PROMPT
from .types import ClassifierOutput, PreSignals, RoutingDecision
from .health_updater import observe_turn_outcome

OPENCLAW_BIN = "openclaw"

# Lazy-init router singleton
_router: Optional[Router] = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(persist=True)
    return _router


def route_turn(
    message: str,
    has_image: bool = False,
    conversation_context: Optional[str] = None,
    cost_profile: str = "balanced",
    nexus_context: Optional[dict] = None,
    use_llm_classifier: bool = False,
    classifier_model: Optional[str] = "openai-codex/gpt-5.4-mini",
) -> RoutingDecision:
    """
    Main entry point for routing a single chat turn.

    Args:
        message:              Incoming user message text
        has_image:            Whether an image is attached
        conversation_context: Optional recent conversation context (last 1-2 turns)
        cost_profile:         cheap|balanced|premium
        nexus_context:        Optional Nexus workflow linkage dict
        use_llm_classifier:   Whether to use LLM classifier (vs heuristic-only)
        classifier_model:     Model to use for LLM classification

    Returns:
        RoutingDecision
    """
    # Step 1: extract pre-signals (fast, no model call)
    pre_signals = extract_pre_signals(message, has_image_attachment=has_image)

    # Step 2: try fast heuristic classifier first (only if no hint provided)
    classifier_output = heuristic_classify(message, pre_signals)

    # Step 3: if heuristic wasn't confident enough and LLM classifier is enabled
    if classifier_output is None and use_llm_classifier and classifier_model:
        classifier_output = _classify_with_openclaw(
            message=message,
            pre_signals=pre_signals,
            conversation_context=conversation_context,
            model=classifier_model,
        )

    # Step 4: fallback to general_chat if still no classification
    if classifier_output is None:
        classifier_output = ClassifierOutput(
            task_type="general_chat",
            complexity="medium",
            cost_profile=cost_profile,
            confidence=0.60,
        )
    else:
        classifier_output.cost_profile = cost_profile

    # Step 5: route
    router = _get_router()
    decision = router.route(
        classifier=classifier_output,
        pre_signals=pre_signals,
        nexus_context=nexus_context or {},
    )

    return decision


def record_turn_outcome(
    decision_id: str,
    provider: str,
    success: bool,
    latency_ms: Optional[int] = None,
    http_status: Optional[int] = None,
    fallback_used: bool = False,
    fallback_model: Optional[str] = None,
    user_override: bool = False,
    user_override_model: Optional[str] = None,
):
    """
    Call this after a turn completes to record outcome and update health.
    Should be called from the OpenClaw plugin post-turn hook.
    """
    from .db import update_outcome

    # Update routing decision outcome
    update_outcome(
        decision_id=decision_id,
        success=success,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
        fallback_model=fallback_model,
        user_override=user_override,
        user_override_model=user_override_model,
    )

    # Update provider health
    error_type = None
    if http_status == 429:
        error_type = "rate_limit"
    elif http_status in (401, 403):
        error_type = "auth"
    elif http_status and http_status >= 500:
        error_type = "server"
    elif not success:
        error_type = "unknown"

    observe_turn_outcome(
        provider=provider,
        http_status=http_status or (200 if success else 500),
        latency_ms=latency_ms,
        error_type=error_type,
    )


def format_decision_for_chat(decision: RoutingDecision, verbose: bool = False) -> str:
    """
    Format a routing decision as a short human-readable note.
    Useful for debug output in chat, e.g. prefixed to a response.
    """
    line = f"[router] → {decision.selected_model} ({decision.task_type}, score={decision.score:.2f})"
    if verbose:
        fallbacks = ", ".join(decision.fallbacks[:2]) or "none"
        reasons   = "; ".join(decision.reason[:2])
        line += f"\n  fallbacks: {fallbacks}\n  reason: {reasons}"
    return line


# ── LLM classifier via OpenClaw agent ────────────────────────────────────────

def _classify_with_openclaw(
    message: str,
    pre_signals: PreSignals,
    conversation_context: Optional[str],
    model: str,
) -> Optional[ClassifierOutput]:
    """
    Call openclaw agent with a classification prompt and parse the result.
    Used only when heuristic classification is not confident enough.
    """
    prompt = build_classification_prompt(message, pre_signals, conversation_context)

    try:
        result = subprocess.run(
            [
                OPENCLAW_BIN, "agent",
                "--model", model,
                "--message", prompt,
                "--system", SYSTEM_PROMPT,
                "--no-persist",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None

        return parse_classifier_response(result.stdout.strip())

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
