"""Deterministic wire-payload inspection. Never invokes a model or reads the DB."""

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class CallCategory(str, Enum):
    STATELESS_HOUSEKEEPING = "STATELESS_HOUSEKEEPING"
    SUBAGENT_WORKER = "SUBAGENT_WORKER"
    MAIN_PLANNING = "MAIN_PLANNING"
    MAIN_EXECUTION = "MAIN_EXECUTION"


def text_blocks(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_blocks(item) for item in value)
    if isinstance(value, dict):
        return text_blocks(value.get("text", value.get("content", "")))
    return ""


@dataclass(frozen=True)
class CliClassification:
    category: CallCategory
    session: str
    reason: str


class CliPayloadClassifier:
    _housekeeping = re.compile(
        r"(?:\b(?:summariz\w*|summaris\w*|compact\w*)\b.{0,65}\b(?:conversation\w*|transcript|context|session)\b"
        r"|\bsummary\s+of\b.{0,40}\b(?:conversation|transcript|session)\b"
        r"|\b(?:conversation|context)\s+(?:summary|compaction)\b"
        r"|\b(?:generate|write|return|create|produce)\b.{0,45}\b(?:title|commit message)\b"
        r"|\b(?:quota|token)[ -]probe\b)", re.I | re.S,
    )
    _worker = re.compile(r"\b(?:sub[ -]?agent|delegated (?:task|worker)|search agent|file search specialist)\b", re.I)
    _orchestrator = {"task", "agent", "todowrite", "spawn_agent", "update_plan"}

    def classify(self, payload: dict, headers: Mapping[str, str]) -> CliClassification:
        headers = {k.lower(): v for k, v in headers.items()}
        messages = payload.get("messages", [])
        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        system = [payload.get("system", ""), payload.get("instructions", "")]
        system += [m.get("content", "") for m in messages + inputs
                   if isinstance(m, dict) and m.get("role") in {"system", "developer"}]
        system_text = text_blocks(system)
        first_user = next((m.get("content", "") for m in messages + inputs
                           if isinstance(m, dict) and m.get("role") == "user"), "")
        identity = headers.get("x-session-id") or headers.get("session_id")
        # Hash explicit IDs too: diagnostics need no raw prompts or session secrets.
        prefix = identity if identity else json.dumps([system, first_user], sort_keys=True, separators=(",", ":"))
        session = hashlib.blake2s(prefix.encode(), digest_size=16).hexdigest()
        tools = payload.get("tools") or []
        names = {str(t.get("name") or t.get("function", {}).get("name", "")).lower()
                 for t in tools if isinstance(t, dict)}
        orchestrator = bool(names & self._orchestrator) or len(tools) >= 5
        model = str(payload.get("model", "")).lower()
        limit = payload.get("max_tokens", payload.get("max_completion_tokens", payload.get("max_output_tokens")))
        tiny = isinstance(limit, (int, float)) and 0 < limit <= 8
        # Empty tools alone are weak evidence. Never demote a normal coding turn
        # merely because its generic system prompt mentions titles or summaries.
        compact = len(tools) <= 1 and not orchestrator
        if compact and (tiny or self._housekeeping.search(system_text)):
            return CliClassification(CallCategory.STATELESS_HOUSEKEEPING, session, "housekeeping_instruction_or_probe")
        if not orchestrator and (re.search(r"claude-.*haiku", model) or self._worker.search(system_text)):
            return CliClassification(CallCategory.SUBAGENT_WORKER, session, "worker_model_or_instruction")
        collaboration = payload.get("collaboration_mode") or payload.get("metadata", {}).get("collaboration_mode") or {}
        phase = collaboration.get("mode") if isinstance(collaboration, dict) else collaboration
        phase = headers.get("x-codex-collaboration-mode", phase)
        marked_plan = re.search(r"<collaboration_mode>\s*(?:plan|debug)\b", system_text, re.I)
        thinking = payload.get("thinking") or {}
        thinking_on = isinstance(thinking, dict) and thinking.get("type") in {"enabled", "adaptive"}
        if model in {"opusplan", "auto_codex"} or thinking_on or phase in {"plan", "debug"} or marked_plan:
            return CliClassification(CallCategory.MAIN_PLANNING, session, "explicit_planning_phase")
        return CliClassification(CallCategory.MAIN_EXECUTION, session, "stateful_default")
