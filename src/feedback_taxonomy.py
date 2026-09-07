from __future__ import annotations

from typing import Optional

# Closed reason-tag vocabulary for route feedback.
# Keep this list stable so weekly reports remain countable/comparable.
ALLOWED_REASON_TAGS: tuple[str, ...] = (
    "task_mismatch",
    "task_and_model_mismatch",
    "quality",
    "too_cheap",
    "too_powerful",
    "latency",
    "tooling",
    "format",
    "policy",
    "other",
    "confirmation",
)

_ALLOWED = set(ALLOWED_REASON_TAGS)

_REASON_ALIASES: dict[str, str] = {
    "task": "task_mismatch",
    "wrong_task": "task_mismatch",
    "task_wrong": "task_mismatch",
    "task+model": "task_and_model_mismatch",
    "both": "task_and_model_mismatch",
    "model_quality": "quality",
    "bad": "quality",
    "too-cheap": "too_cheap",
    "cheap": "too_cheap",
    "too-powerful": "too_powerful",
    "powerful": "too_powerful",
    "slow": "latency",
    "latency_high": "latency",
    "tools": "tooling",
    "tool": "tooling",
    "style": "format",
    "unsafe": "policy",
    "ok": "confirmation",
}



def normalize_reason_tag(
    *,
    reason_tag: Optional[str],
    verdict: str,
    corrected_task: Optional[str],
    model_verdict: Optional[str],
    preferred_model: Optional[str],
) -> str:
    raw = str(reason_tag or "").strip().lower()
    if raw:
        canonical = _REASON_ALIASES.get(raw, raw)
        if canonical in _ALLOWED:
            return canonical
        return "other"

    has_task_signal = bool(str(corrected_task or "").strip())
    has_model_signal = bool(str(model_verdict or "").strip()) or bool(str(preferred_model or "").strip())
    mv = str(model_verdict or "").strip().lower()

    if has_task_signal and has_model_signal:
        return "task_and_model_mismatch"
    if has_task_signal:
        return "task_mismatch"
    if mv == "too_cheap":
        return "too_cheap"
    if mv == "too_powerful":
        return "too_powerful"
    if mv in {"bad", "good", "neutral"}:
        return "quality"
    if str(verdict or "").strip().lower() == "correct":
        return "confirmation"
    return "other"
