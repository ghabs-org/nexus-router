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
import os
import re
import shutil
import subprocess

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .types import ClassifierOutput, PreSignals, ProviderHealth
from .paths import REGISTRY_FILE

ROOT = Path(__file__).parent.parent
DEFAULT_CLASSIFIER_LIMIT = 4
CLASSIFIER_PROFILE = "nexus-router-classifier"
DEFAULT_DIRECT_PROVIDER_TIMEOUT_SECONDS = 15

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


@dataclass(frozen=True)
class DirectClassifierProviderAdapter:
    provider: str
    env_var: str
    base_url: str
    api_path: str


DIRECT_PROVIDER_ADAPTERS: dict[str, DirectClassifierProviderAdapter] = {
    "openai-codex": DirectClassifierProviderAdapter(
        provider="openai-codex",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com",
        api_path="/v1/chat/completions",
    ),
}

_AUTH_PROFILES_PATH = Path.home() / ".openclaw/agents/main/agent/auth-profiles.json"


def _resolve_provider_token(provider: str) -> Optional[str]:
    adapter = DIRECT_PROVIDER_ADAPTERS.get(provider)
    env_key = adapter.env_var if adapter else None

    if env_key:
        val = os.environ.get(env_key)
        if val:
            return val

    try:
        profiles = json.loads(_AUTH_PROFILES_PATH.read_text())
    except Exception:
        return None

    last_good = profiles.get("lastGood", {}).get(provider)
    all_keys = list(profiles.get("profiles", {}).keys())
    candidates = ([last_good] + [k for k in all_keys if k != last_good]) if last_good else all_keys

    for key in candidates:
        profile = profiles.get("profiles", {}).get(key)
        if not isinstance(profile, dict):
            continue
        if profile.get("provider") != provider:
            continue
        ptype = profile.get("type")
        if ptype == "oauth":
            token = profile.get("access")
            if token:
                return token
        elif ptype == "token":
            token = profile.get("token")
            if token:
                return token

    return None



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
        parts.append(
            "Conversation context (use this to infer the true intent of short replies, "
            "follow-ups, config questions, and continuation messages):\n"
            f"{conversation_context}"
        )
    parts.append(
        "Message to classify (classify the current turn together with the conversation context, "
        "not just the raw last message):\n"
        f"{message}"
    )

    return "\n\n".join(parts)


def parse_classifier_response(raw: str) -> Optional[ClassifierOutput]:
    """
    Parse raw model response into ClassifierOutput.
    Handles JSON embedded in markdown code fences, bare JSON, OpenClaw JSON wrappers,
    and a permissive key:value fallback when the model is slightly malformed.
    """
    def _build_classifier_output(data: dict) -> Optional[ClassifierOutput]:
        if not isinstance(data, dict):
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

    # OpenClaw --json wraps payloads; unwrap the first assistant text payload if present.
    try:
        wrapped = json.loads(raw)
        if isinstance(wrapped, dict) and isinstance(wrapped.get("payloads"), list):
            texts = [
                payload.get("text", "")
                for payload in wrapped["payloads"]
                if isinstance(payload, dict) and isinstance(payload.get("text"), str)
            ]
            if texts:
                raw = "\n".join(texts)
    except Exception:
        pass

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    # Extract first JSON object if there's surrounding prose
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            out = _build_classifier_output(data)
            if out is not None:
                return out
        except json.JSONDecodeError:
            pass

    # Permissive fallback for lightly malformed output like:
    # {task_type:general_chat,confidence:0.5}
    normalized = raw.replace("\\", "").strip().strip("{}")
    if not normalized:
        return None

    extracted: dict[str, str] = {}
    for key in ("task_type", "subtype", "complexity", "needs_tools", "needs_vision", "needs_long_context", "cost_profile", "confidence", "detected_language"):
        m = re.search(rf"{key}\s*[:=]\s*([^,\n}}]+)", normalized)
        if m:
            extracted[key] = m.group(1).strip().strip('"\'')

    if not extracted:
        return None

    data = {
        "task_type": extracted.get("task_type", "general_chat"),
        "subtype": extracted.get("subtype"),
        "complexity": extracted.get("complexity", "medium"),
        "needs_tools": extracted.get("needs_tools", "true"),
        "needs_vision": extracted.get("needs_vision", "false"),
        "needs_long_context": extracted.get("needs_long_context", "false"),
        "cost_profile": extracted.get("cost_profile", "balanced"),
        "confidence": extracted.get("confidence", "0.75"),
        "detected_language": extracted.get("detected_language"),
    }

    try:
        data["confidence"] = float(data["confidence"])
    except Exception:
        data["confidence"] = 0.75

    return _build_classifier_output(data)


def select_classifier_models(
    registry: Optional[list[dict]] = None,
    provider_health: Optional[dict[str, ProviderHealth]] = None,
    limit: int = DEFAULT_CLASSIFIER_LIMIT,
    has_image: bool = False,
    preferred_model: Optional[str] = None,
) -> list[str]:
    """
    Pick the most efficient available classifier models from the normalized registry.

    Primary objective: keep classifier cost low while preserving enough reasoning /
    multilingual quality to avoid brittle routing.
    """
    if registry is None:
        try:
            with open(REGISTRY_FILE) as f:
                raw = json.load(f)
            registry = raw.get("models", [])
        except Exception:
            registry = []

    scored: list[tuple[float, str]] = []
    for model in registry:
        if not model.get("availability", {}).get("authed", False):
            continue

        model_id = model.get("id")
        if not model_id:
            continue

        scores = model.get("scores", {})
        provider = model.get("provider")
        health = 0.85
        if provider_health and provider in provider_health:
            health = provider_health[provider].health_score

        if health < 0.30:
            continue

        cost = float(scores.get("cost", 0.65))
        fast = float(scores.get("fast", 0.70))
        reasoning = float(scores.get("reasoning", 0.70))
        multilingual = float(scores.get("multilingual", 0.70))
        tools = float(scores.get("tools", 0.70))
        vision = float(scores.get("vision", 0.70))

        score = (
            0.42 * cost
            + 0.22 * fast
            + 0.18 * reasoning
            + 0.08 * multilingual
            + 0.05 * tools
            + 0.05 * (vision if has_image else 0.0)
            + 0.10 * health
        )
        scored.append((score, model_id))

    ordered = [m for _, m in sorted(scored, key=lambda x: x[0], reverse=True)]
    if preferred_model:
        ordered = [preferred_model] + [m for m in ordered if m != preferred_model]
    return ordered[:limit] if limit > 0 else ordered


def _prepare_openclaw_profile(profile: str = CLASSIFIER_PROFILE) -> None:
    source = Path.home() / ".openclaw"
    target = Path.home() / f".openclaw-{profile}"

    try:
        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)

        # Keep the classifier profile lean: auth + model config only.
        ext_dir = target / "extensions"
        if ext_dir.exists():
            shutil.rmtree(ext_dir)

        config_path = target / "openclaw.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            hooks = config.get("hooks")
            if isinstance(hooks, dict):
                hooks["enabled"] = False
                hooks["internal"] = {"enabled": False, "entries": {}}
            # Remove plugins entirely for the classifier profile so the router cannot recurse.
            if "plugins" in config:
                config.pop("plugins", None)
            config_path.write_text(json.dumps(config, indent=2))
    except Exception:
        # Best effort: if the profile can't be prepared, the classifier will fall back.
        return


def _read_openclaw_default_model(binary: str = "openclaw", profile: str = CLASSIFIER_PROFILE) -> Optional[str]:
    _prepare_openclaw_profile(profile)
    try:
        result = subprocess.run(
            [binary, "--profile", profile, "models", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return None

    for key in ("primaryModel", "primary", "model", "defaultModel"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ("id", "model", "value"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def _set_openclaw_default_model(model: str, binary: str = "openclaw", profile: str = CLASSIFIER_PROFILE) -> bool:
    _prepare_openclaw_profile(profile)
    try:
        result = subprocess.run(
            [binary, "--profile", profile, "models", "set", model],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    return result.returncode == 0


def _run_codex_classifier_turn(
    prompt: str,
    model: str,
    binary: str = "codex",
    timeout_seconds: int = 15,
) -> Optional[str]:
    """
    Run a single classifier turn through the Codex CLI.
    Uses `codex exec --skip-git-repo-check -q --model <model> <prompt>`.
    Returns raw stdout string or None on failure.
    """
    if not shutil.which(binary):
        return None

    model_id = model.split("/", 1)[1] if "/" in model else model

    # Build a compact one-shot prompt asking for JSON only
    one_shot = (
        "Return ONLY a JSON object (no markdown, no explanation) matching this schema:\n"
        "{task_type, subtype?, complexity, needs_tools, needs_vision, needs_long_context, "
        "cost_profile, confidence, detected_language?}\n\n"
        f"Message to classify:\n{prompt}"
    )

    cmd = [
        binary, "exec",
        "--skip-git-repo-check",
        "--json",
        "-m", model_id,
        one_shot,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        # --json mode: scan lines for item.completed with agent_message text
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "").strip()
                    if text:
                        return text
        # Fallback: return raw stdout
        output = result.stdout.strip()
        return output if output else None
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None


def _run_openclaw_classifier_turn(
    prompt: str,
    model: str,
    binary: str,
    timeout_seconds: int,
) -> Optional[str]:
    result = None
    try:
        if not _set_openclaw_default_model(model, binary=binary):
            return None

        result = subprocess.run(
            [binary, "--profile", CLASSIFIER_PROFILE, "agent", "--agent", "main", "--json", "--thinking", "low", "--message", prompt, "--timeout", str(timeout_seconds)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 5,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if result is None or result.returncode != 0:
        return None

    return (result.stdout or "").strip()


def _split_model_ref(model: str) -> tuple[Optional[str], str]:
    trimmed = (model or "").strip()
    if not trimmed:
        return None, ""
    if "/" not in trimmed:
        return None, trimmed
    provider, model_id = trimmed.split("/", 1)
    return provider or None, model_id


def _extract_response_output_text(payload: dict) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts).strip()


def _call_direct_provider_classifier(
    prompt: str,
    model: str,
    timeout_seconds: int = DEFAULT_DIRECT_PROVIDER_TIMEOUT_SECONDS,
) -> Optional[str]:
    provider, model_id = _split_model_ref(model)
    if not provider or not model_id:
        return None

    adapter = DIRECT_PROVIDER_ADAPTERS.get(provider)
    if adapter is None:
        return None

    api_key = _resolve_provider_token(provider)
    if not api_key:
        return None

    # Use chat/completions schema which works with both API keys and OAuth tokens.
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    req = urllib_request.Request(
        f"{adapter.base_url}{adapter.api_path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, OSError):
        return None

    try:
        parsed = json.loads(raw)
    except Exception:
        return raw.strip() or None

    # Chat-completions response schema
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    # Fallback: responses API or raw
    text = _extract_response_output_text(parsed)
    if text:
        return text

    if isinstance(parsed.get("output_text"), str):
        return parsed["output_text"].strip() or None

    return raw.strip() or None


def classify_with_model(
    message: str,
    pre_signals: PreSignals,
    conversation_context: Optional[str] = None,
    model: Optional[str] = None,
    fallback_models: Optional[list[str]] = None,
    registry: Optional[list[dict]] = None,
    provider_health: Optional[dict[str, ProviderHealth]] = None,
    binary: str = "openclaw",
    timeout_seconds: int = 15,
) -> Optional[ClassifierOutput]:
    """
    Classify a message using a model-backed classifier.

    This is the language-aware path: use it when heuristics are inconclusive.
    Returns None if all model invocations fail or emit invalid JSON.
    """
    prompt = build_classification_prompt(message, pre_signals, conversation_context)

    candidate_models: list[str] = []
    if model and model != "auto":
        candidate_models.append(model)
    else:
        candidate_models.extend(
            select_classifier_models(
                registry=registry,
                provider_health=provider_health,
                has_image=pre_signals.has_image,
                preferred_model=None,
            )
        )

    if fallback_models:
        candidate_models.extend([m for m in fallback_models if m not in candidate_models])

    if not candidate_models:
        return None

    direct_attempted = False
    for candidate in candidate_models:
        provider_name = candidate.split("/", 1)[0] if "/" in candidate else ""
        stdout = _call_direct_provider_classifier(
            prompt,
            candidate,
            timeout_seconds=timeout_seconds,
        )
        if stdout is not None:
            direct_attempted = True
            parsed = parse_classifier_response(stdout)
            if parsed is not None:
                object.__setattr__(parsed, "classifier_provider", candidate.split("/", 1)[0] if "/" in candidate else None)
                object.__setattr__(parsed, "classifier_model", candidate)
                return parsed

    # Codex CLI path — preferred over openclaw agent for openai-codex provider
    codex_binary = "codex"
    if shutil.which(codex_binary):
        for candidate in candidate_models:
            if not candidate.startswith("openai-codex/"):
                continue
            stdout = _run_codex_classifier_turn(prompt, candidate, binary=codex_binary, timeout_seconds=timeout_seconds)
            if not stdout:
                continue
            parsed = parse_classifier_response(stdout)
            if parsed is not None:
                object.__setattr__(parsed, "classifier_provider", "openai-codex")
                object.__setattr__(parsed, "classifier_model", candidate)
                return parsed

    # OpenClaw CLI fallback
    if not shutil.which(binary):
        return None

    for candidate in candidate_models:
        provider_name = candidate.split("/", 1)[0] if "/" in candidate else ""
        stdout = _run_openclaw_classifier_turn(prompt, candidate, binary=binary, timeout_seconds=timeout_seconds)
        if not stdout:
            continue
        parsed = parse_classifier_response(stdout)
        if parsed is not None:
            parsed.classifier_provider = candidate.split("/", 1)[0] if "/" in candidate else None
            parsed.classifier_model = candidate
            return parsed

    return None


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

    # Short conversational greetings/PMs: classify as general_chat with high confidence
    if pre_signals.message_length < 60 and not pre_signals.has_code:
        lower = message.strip().lower()
        greetings = ("hi", "hello", "hey", "how are you", "how's your day", "how are you?")
        if any(lower.startswith(g) or g in lower for g in greetings):
            return ClassifierOutput(
                task_type="general_chat",
                complexity="low",
                needs_tools=False,
                cost_profile="cheap",
                confidence=0.90,
            )

    # Not confident enough for fast-path
    return None


class AnthropicDirectClassifierAdapter:
    provider_name = "anthropic"

    def is_available(self) -> bool:
        return anthropic is not None and bool(os.getenv("ANTHROPIC_API_KEY"))

    def classify(self, model_ref: str, prompt: str, timeout_seconds: int) -> Optional[str]:
        if anthropic is None:
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)
        try:
            message = client.messages.create(
                model=model_ref,
                max_tokens=512,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            )
        except Exception:
            return None

        chunks = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                chunks.append(block.text)
        return "\n".join(chunks).strip() if chunks else None
