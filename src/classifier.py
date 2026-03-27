"""
classifier.py — Nexus classifier interface for Nexus Router.

Provides:
- ClassifierPrompt: builds the structured classification prompt
- classify_with_model(): calls a local model to classify a message
- extract_pre_signals(): fast non-linguistic feature extraction

The classifier agent should return a JSON object matching ClassifierOutput.
For V1 this uses a lightweight prompt; it can be swapped for a Nexus workflow step.
"""

import json
import re
from typing import Optional

from .types import ClassifierOutput, PreSignals

# ── Classification prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a request classifier for an AI routing system.
Analyze the user message and output ONLY a JSON object — no prose, no markdown.

Task types:
- coding: code generation, implementation, bug fixes, refactoring
- code_review: reviewing diffs, PRs, auditing code quality
- reasoning: planning, strategy, comparisons, architecture decisions
- summarization: summaries, extraction, TL;DR, synthesis of content
- fast_utility: quick lookups, minor edits, short questions, simple rewrites
- long_context: processing large documents, many files, long transcripts
- vision: understanding images, screenshots, diagrams (only if image attached)
- general_chat: conversational, general questions, not specialized

Subtypes (optional, null if not applicable):
- coding: implementation | debugging | refactor | test_writing
- reasoning: planning | comparison | decision | architecture
- summarization: extractive | synthesis | report

Complexity: low | medium | high

Output schema (JSON only):
{
  "task_type": "...",
  "subtype": "...",
  "complexity": "...",
  "needs_tools": true,
  "needs_vision": false,
  "needs_long_context": false,
  "cost_profile": "cheap|balanced|premium",
  "confidence": 0.0,
  "detected_language": "en|it|fr|es|de|mixed|unknown"
}"""


def build_classification_prompt(
    message: str,
    pre_signals: PreSignals,
    conversation_context: Optional[str] = None,
) -> str:
    """
    Build a classification prompt including pre-signals as context hints.
    """
    hints = []
    if pre_signals.has_image:
        hints.append("- An image is attached")
    if pre_signals.has_code:
        hints.append("- Message contains code blocks")
    if pre_signals.has_diff:
        hints.append("- Message contains a diff/patch")
    if pre_signals.has_logs:
        hints.append("- Message contains stack traces or logs")
    if pre_signals.estimated_tokens > 50_000:
        hints.append(f"- Large content (~{pre_signals.estimated_tokens:,} tokens estimated)")
    if pre_signals.estimated_tokens > 200_000:
        hints.append("- Very large context — likely needs long_context routing")

    parts = []
    if hints:
        parts.append("Pre-detected signals:\n" + "\n".join(hints))
    if conversation_context:
        parts.append(f"Conversation context:\n{conversation_context}")
    parts.append(f"Message to classify:\n{message}")

    return "\n\n".join(parts)


def parse_classifier_response(raw: str) -> Optional[ClassifierOutput]:
    """
    Parse raw model response into ClassifierOutput.
    Handles JSON embedded in markdown code fences or bare JSON.
    """
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    # Extract first JSON object if there's surrounding prose
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    valid_task_types = {
        "coding", "code_review", "reasoning", "summarization",
        "fast_utility", "long_context", "vision", "general_chat",
    }

    task_type = data.get("task_type", "general_chat")
    if task_type not in valid_task_types:
        task_type = "general_chat"

    return ClassifierOutput(
        task_type=task_type,
        subtype=data.get("subtype"),
        complexity=data.get("complexity", "medium"),
        needs_tools=bool(data.get("needs_tools", True)),
        needs_vision=bool(data.get("needs_vision", False)),
        needs_long_context=bool(data.get("needs_long_context", False)),
        cost_profile=data.get("cost_profile", "balanced"),
        confidence=float(data.get("confidence", 0.75)),
        detected_language=data.get("detected_language"),
    )


# ── Pre-signal extraction ─────────────────────────────────────────────────────

# Code block patterns
_RE_CODE_FENCE  = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_RE_INLINE_CODE = re.compile(r"`[^`]+`")
_RE_DIFF_HUNK   = re.compile(r"^@@\s+-\d+", re.MULTILINE)
_RE_DIFF_LINE   = re.compile(r"^[+-]{3} ", re.MULTILINE)
_RE_STACK_TRACE = re.compile(
    r"(Traceback \(most recent call last\)|at \w+\.\w+\(|Exception in thread|"
    r"Error:.*\n\s+at |^\s+File \".*\", line \d+)",
    re.MULTILINE,
)
_RE_URL         = re.compile(r"https?://\S+")
_RE_FILE_PATH   = re.compile(r"(?:^|\s)(?:\./|/|~/)[\w./\-]+\.\w{1,6}", re.MULTILINE)

# Rough token estimate: average 4 chars per token
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def extract_pre_signals(
    message: str,
    has_image_attachment: bool = False,
) -> PreSignals:
    """
    Extract non-linguistic signals from message content.
    Fast: no model call needed.
    """
    has_code = bool(
        _RE_CODE_FENCE.search(message) or
        _RE_INLINE_CODE.search(message)
    )
    has_diff = bool(
        _RE_DIFF_HUNK.search(message) or
        _RE_DIFF_LINE.search(message)
    )
    has_logs  = bool(_RE_STACK_TRACE.search(message))
    has_url   = bool(_RE_URL.search(message))
    has_files = bool(_RE_FILE_PATH.search(message))

    # Refine: if diff detected, also flag code
    if has_diff:
        has_code = True

    return PreSignals(
        has_image=has_image_attachment,
        has_code=has_code,
        has_diff=has_diff,
        has_logs=has_logs,
        has_url=has_url,
        has_file_refs=has_files,
        estimated_tokens=_estimate_tokens(message),
        message_length=len(message),
    )


def heuristic_classify(message: str, pre_signals: PreSignals) -> Optional[ClassifierOutput]:
    """
    Fast-path heuristic classifier.
    Returns a ClassifierOutput when signals are unambiguous enough,
    skipping the LLM classifier call entirely.

    Returns None if the heuristic is not confident — caller should
    fall through to LLM classification.
    """
    # Vision: image attached with little text
    if pre_signals.has_image and pre_signals.message_length < 500:
        return ClassifierOutput(
            task_type="vision",
            complexity="medium",
            needs_vision=True,
            confidence=0.90,
        )

    # Large context: very long message
    if pre_signals.estimated_tokens > 200_000:
        return ClassifierOutput(
            task_type="long_context",
            complexity="high",
            needs_long_context=True,
            confidence=0.85,
        )

    # Diff detected: likely code review
    if pre_signals.has_diff and not pre_signals.has_image:
        return ClassifierOutput(
            task_type="code_review",
            subtype="review",
            complexity="medium",
            needs_tools=True,
            confidence=0.82,
        )

    # Stack trace: likely debugging
    if pre_signals.has_logs and pre_signals.has_code:
        return ClassifierOutput(
            task_type="coding",
            subtype="debugging",
            complexity="medium",
            needs_tools=True,
            confidence=0.80,
        )

    # Very short message, no code: likely fast utility
    if pre_signals.message_length < 120 and not pre_signals.has_code:
        return ClassifierOutput(
            task_type="fast_utility",
            complexity="low",
            needs_tools=False,
            cost_profile="cheap",
            confidence=0.72,
        )

    # Not confident enough for fast-path
    return None
