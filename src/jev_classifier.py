"""Jev decision-model classifier backend for Nexus Router.

Sits between the local ONNX classifier and the heuristic fallback in the
server chain: typed Choice/Score answers with calibrated confidence, no
JSON-parse step to break. Low confidence (or any failure) returns None so
routing falls through to heuristic/LLM — the safe default wins.

Env:
    JEV_ENABLED ("true" unless set otherwise; needs OPENCODE_API_KEY or OPENROUTER_API_KEY)
    JEV_BACKEND (openrouter|opencode; default: free opencode path when keyed)
    JEV_MODEL (default ``jev-1.13-free`` on opencode, ``typesafe/jev-1.13`` on OpenRouter)
    JEV_MIN_CONFIDENCE (default 0.6)
    JEV_TIMEOUT_SECONDS (default 15)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .types import ClassifierOutput, PreSignals

JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_SYSTEMONE_URL = "https://opencode.ai/zen/v1/systemone"
JEV_DEFAULT_MODEL = "typesafe/jev-1.13"
JEV_DEFAULT_FREE_MODEL = "jev-1.13-free"

TASK_TYPE_OPTIONS = {
    "coding": "code generation, implementation, bug fixes, refactoring",
    "code_review": "reviewing diffs, PRs, auditing code quality",
    "reasoning": "planning, strategy, comparisons, architecture decisions",
    "summarization": "summaries, extraction, TL;DR, synthesis",
    "fast_utility": "quick lookups, minor edits, short questions, simple rewrites",
    "long_context": "large documents, many files, long transcripts",
    "vision": "understanding images, screenshots, diagrams",
    "general_chat": "conversational, general questions, not specialized",
}

COMPLEXITY_OPTIONS = {
    "low": "trivial, single-step, no reasoning needed",
    "medium": "some reasoning or multi-step work",
    "high": "deep reasoning, planning, or high stakes",
}


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() == "true"


def _jev_backend() -> tuple | None:
    """Resolve (endpoint, key, model): free OpenCode path first, else OpenRouter."""
    backend = (os.getenv("JEV_BACKEND") or "").strip().lower()
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    opencode_key = (os.getenv("OPENCODE_API_KEY") or "").strip()
    model = (os.getenv("JEV_MODEL") or "").strip()
    if backend == "openrouter" or (not backend and openrouter_key and not opencode_key):
        if not openrouter_key:
            return None
        return (JEV_DECISIONS_URL, openrouter_key, model or JEV_DEFAULT_MODEL)
    if backend == "opencode" or opencode_key:
        if not opencode_key:
            return None
        return (JEV_SYSTEMONE_URL, opencode_key, model or JEV_DEFAULT_FREE_MODEL)
    if openrouter_key:
        return (JEV_DECISIONS_URL, openrouter_key, model or JEV_DEFAULT_MODEL)
    return None


def _post_decisions(endpoint: str, state: Any, questions: dict, api_key: str, model: str, timeout: int) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib_request.Request(
        endpoint,
        data=body,
        # Cloudflare fronts Zen: the stdlib's default `Python-urllib/3.x`
        # UA trips its bot check (403/1010). Any real client UA passes.
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "nexus-router/1.0 (python-urllib)",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib_error.URLError, OSError, ValueError, TimeoutError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        return payload["answers"]
    return {}


@dataclass
class JevClassifierResult:
    classifier: ClassifierOutput
    confidence: float
    probabilities: dict


def classify_with_jev(
    message: str,
    pre_signals: PreSignals,
    conversation_context: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> Optional[JevClassifierResult]:
    """Classify via Jev; None when disabled, unkeyed, unsure, or failing."""
    if not _env_flag("JEV_ENABLED", True):
        return None
    backend = _jev_backend()
    if backend is None:
        return None
    endpoint, api_key, model = backend
    threshold = (
        float(min_confidence)
        if min_confidence is not None
        else float(os.getenv("JEV_MIN_CONFIDENCE", "0.6"))
    )
    state: dict[str, Any] = {"message": message}
    if conversation_context and conversation_context.strip():
        state["conversation_context"] = conversation_context.strip()
    state["has_image_attachment"] = bool(getattr(pre_signals, "has_image", False))
    questions = {
        "task_type": {
            "type": "choice",
            "instructions": "What kind of task is this message?",
            "criteria": TASK_TYPE_OPTIONS,
        },
        "complexity": {
            "type": "choice",
            "instructions": "How demanding is this task?",
            "criteria": COMPLEXITY_OPTIONS,
        },
        "needs_tools": {
            "type": "noul",
            "instructions": "Does this request need tools, code execution, or external actions?",
        },
    }
    answers = _post_decisions(
        endpoint,
        state,
        questions,
        api_key,
        model,
        int(os.getenv("JEV_TIMEOUT_SECONDS", "15")),
    )
    task = answers.get("task_type") if isinstance(answers, dict) else None
    if not isinstance(task, dict):
        return None
    choice = str(task.get("choice") or "")
    confidence = float(task.get("confidence") or 0.0)
    probabilities = dict(task.get("probabilities") or {})
    if choice not in TASK_TYPE_OPTIONS or confidence < threshold:
        return None
    complexity = answers.get("complexity") or {}
    complexity_choice = str(complexity.get("choice") or "medium")
    if complexity_choice not in COMPLEXITY_OPTIONS:
        complexity_choice = "medium"
    needs_tools = answers.get("needs_tools") or {}
    try:
        tools_prob = float(needs_tools.get("noul", 0.5))
    except (TypeError, ValueError):
        tools_prob = 0.5
    classifier = ClassifierOutput(
        task_type=choice,
        subtype=None,
        complexity=complexity_choice,
        needs_tools=tools_prob >= 0.5,
        needs_vision=bool(getattr(pre_signals, "has_image", False)),
        needs_long_context=False,
        cost_profile="balanced",
        confidence=confidence,
        detected_language=None,
        classifier_provider="jev",
        classifier_model=model,
    )
    return JevClassifierResult(
        classifier=classifier, confidence=confidence, probabilities=probabilities
    )
