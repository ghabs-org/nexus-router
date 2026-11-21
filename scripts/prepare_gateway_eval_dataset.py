#!/usr/bin/env python3
"""Prepare offline gateway evaluation datasets from local CLI session histories.

Subcommands:
  extract   Read opening substantive prompts from Claude Code / Codex history,
            redact secrets/tokens, deduplicate, and write JSONL outside git.
  distill   Assign 3-tier effort/profile labels (`utility` | `standard` | `deep`)
            with an explicit rubric and flag rule disagreements for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EVAL_DIR = Path.home() / ".local/state/nexus-router/artifacts/gateway-eval"
DEFAULT_CLAUDE_HISTORY = Path.home() / ".claude/history.jsonl"
DEFAULT_CODEX_HISTORY = Path.home() / ".codex/history.jsonl"
DEFAULT_GATEWAY_CONFIG = ROOT / "src/gateway.codex.yaml"
VALID_PROFILES = ("utility", "standard", "deep")

_TRIVIAL_PROMPTS = frozenset({
    "ok", "okay", "ok thanks", "thanks", "thank you", "yes", "yes please",
    "no", "continue", "go ahead", "done", "great", "hello", "hi", "sure",
    "looks good", "lgtm", "ship it", "proceed", "carry on",
})

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.I), r"\1 [REDACTED]"),
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AIza[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9-]{12,}|AKIA[0-9A-Z]{16})\b"), "[REDACTED_KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED_JWT]"),
    (re.compile(r"([?&](?:token|key|apikey|api_key|secret|password|auth|sig|signature)=)[^&\s]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(\b(?:api_key|secret|password|access_token|refresh_token|bearer_token|client_secret)\s*[:=]\s*['\"]?)[A-Za-z0-9._+/=-]{12,}(['\"]?)", re.I), r"\1[REDACTED_SECRET]\2"),
    (re.compile(r"(?<![A-Za-z0-9_/.-])(?=[A-Za-z0-9+/=]{48,}(?![A-Za-z0-9_]))(?=[A-Za-z0-9+/=]*[0-9])(?=[A-Za-z0-9+/=]*[A-Za-z])[A-Za-z0-9+/=]{48,}"), "[REDACTED_TOKEN]"),
)

DISTILL_SYSTEM_PROMPT = """Classify a user's opening coding/agent task into one gateway effort profile.

Return ONLY valid JSON:
{"profile": "utility" | "standard" | "deep", "confidence": 0.0-1.0, "rationale": "short phrase"}

Rubric:
- "utility" (low reasoning effort): brief rewrite, rephrase, translate, extract, format, quick status/fact lookup, or trivial one-line shell/git check.
- "standard" (medium reasoning effort): scoped bug fix, single-module feature, writing/fixing tests, focused code review, summarizing or explaining code/docs.
- "deep" (high reasoning effort): architecture or design decisions, multi-module refactor, subtle/distributed root-cause debugging, security/performance trade-off analysis, multi-step planning.
"""


def sanitize_prompt(text: str) -> str:
    """Redact API keys, bearer tokens, URL query secrets, and high-entropy blobs."""
    cleaned = str(text or "")
    for pattern, repl in _SECRET_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_substantive_prompt(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 15:
        return False
    # Skip slash commands like /clear, /compact, /model gpt-5, while keeping
    # prompts that begin with an explicit filesystem path (/home/ubuntu/...).
    if stripped.startswith("/") and not re.match(r"^/[A-Za-z0-9_.-]+/", stripped):
        return False
    if _normalize_text(stripped) in _TRIVIAL_PROMPTS:
        return False
    return True


def _extract_first_per_session(
    records: list[tuple[str, str, float, int]],
    *,
    source: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    by_session: dict[str, list[tuple[float, int, str]]] = {}
    for session_id, text, ts, idx in records:
        by_session.setdefault(session_id or f"anon-{idx}", []).append((ts, idx, text))

    extracted: list[dict[str, Any]] = []
    for session_id in sorted(by_session):
        items = sorted(by_session[session_id], key=lambda item: (item[0], item[1]))
        for _ts, _idx, raw_text in items:
            if not _is_substantive_prompt(raw_text):
                continue
            sanitized = sanitize_prompt(raw_text)[:max_chars].strip()
            if not _is_substantive_prompt(sanitized):
                continue
            extracted.append({"source": source, "session": session_id, "task": sanitized})
            break
    return extracted


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def extract_prompts(
    *,
    claude_history: Path | None = DEFAULT_CLAUDE_HISTORY,
    codex_history: Path | None = DEFAULT_CODEX_HISTORY,
    extra_jsonl: Path | None = None,
    output_path: Path,
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Extract, sanitize, and deduplicate opening prompts from local CLI histories."""
    candidates: list[dict[str, Any]] = []

    claude_records: list[tuple[str, str, float, int]] = []
    for idx, row in enumerate(_read_jsonl(claude_history)):
        text = str(row.get("display") or "").strip()
        session = str(row.get("sessionId") or "").strip()
        ts = float(row.get("timestamp") or idx)
        if text:
            claude_records.append((session, text, ts, idx))
    candidates.extend(_extract_first_per_session(claude_records, source="claude", max_chars=max_chars))

    codex_records: list[tuple[str, str, float, int]] = []
    for idx, row in enumerate(_read_jsonl(codex_history)):
        text = str(row.get("text") or "").strip()
        session = str(row.get("session_id") or "").strip()
        ts = float(row.get("ts") or idx)
        if text:
            codex_records.append((session, text, ts, idx))
    candidates.extend(_extract_first_per_session(codex_records, source="codex", max_chars=max_chars))

    for idx, row in enumerate(_read_jsonl(extra_jsonl)):
        text = str(row.get("task") or row.get("text") or row.get("prompt") or row.get("display") or "").strip()
        if not _is_substantive_prompt(text):
            continue
        sanitized = sanitize_prompt(text)[:max_chars].strip()
        if _is_substantive_prompt(sanitized):
            candidates.append({
                "source": str(row.get("source") or "extra"),
                "session": str(row.get("id") or f"extra-{idx}"),
                "task": sanitized,
            })

    seen: set[str] = set()
    rows_out: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for item in candidates:
        norm = _normalize_text(item["task"])
        if norm in seen:
            continue
        seen.add(norm)
        digest = hashlib.sha256(norm.encode()).hexdigest()[:8]
        source = item["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
        rows_out.append({
            "id": f"{source}-{digest}",
            "source": source,
            "task": item["task"],
            "char_len": len(item["task"]),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    return {
        "output_path": str(output_path),
        "candidates": len(candidates),
        "written": len(rows_out),
        "sources": source_counts,
    }


def _load_env_file_keys() -> dict[str, str]:
    found: dict[str, str] = {}
    env_file = ROOT / "deploy/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            found[key.strip()] = val.strip().strip('"').strip("'")
    openclaw_cfg = Path.home() / ".openclaw/openclaw.json"
    if openclaw_cfg.exists():
        try:
            cfg = json.loads(openclaw_cfg.read_text())
            raw_nvidia_key = ((cfg.get("models") or {}).get("providers") or {}).get("nvidia", {}).get("apiKey")
            if isinstance(raw_nvidia_key, str):
                nvidia_key = raw_nvidia_key.strip()
                if nvidia_key and not nvidia_key.startswith("${"):
                    found.setdefault("NVIDIA_API_KEY", nvidia_key)
        except Exception:
            pass
    return found


def resolve_distill_target(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str, str, str] | None:
    """Resolve (provider, endpoint_url, api_key, model) without leaking secrets."""
    env_keys = _load_env_file_keys()
    candidates = {
        "openai": (
            "https://api.openai.com/v1/chat/completions",
            os.environ.get("OPENAI_API_KEY") or env_keys.get("OPENAI_API_KEY", ""),
            "gpt-5.4-mini",
        ),
        "nvidia": (
            "https://integrate.api.nvidia.com/v1/chat/completions",
            os.environ.get("NVIDIA_API_KEY") or env_keys.get("NVIDIA_API_KEY", ""),
            "moonshotai/kimi-k3.1-pro",
        ),
        "openrouter": (
            "https://openrouter.ai/api/v1/chat/completions",
            os.environ.get("OPENROUTER_API_KEY") or env_keys.get("OPENROUTER_API_KEY", ""),
            "openai/gpt-5.4-mini",
        ),
        "opencode": (
            "https://opencode.ai/zen/v1/chat/completions",
            os.environ.get("OPENCODE_API_KEY") or env_keys.get("OPENCODE_API_KEY", ""),
            "deepseek-v4.1-flash-free",
        ),
    }
    order = [provider] if provider else ["openai", "nvidia", "openrouter", "opencode"]
    for name in order:
        if name not in candidates:
            raise ValueError(f"Unsupported distill provider: {name}")
        url, key, default_model = candidates[name]
        if key:
            return name, url, key, model or default_model
    return None


def _build_offline_policy(config_path: Path):
    import yaml
    from src.gateway_policy import GatewayPolicy

    config = yaml.safe_load(config_path.read_text())
    for prov in (config.get("providers") or {}).values():
        if prov.get("transport") == "vertex":
            prov.setdefault("project", "offline-eval")
            prov.setdefault("region", "global")
            prov.setdefault("base_url", "https://aiplatform.googleapis.com")
    config = dict(config)
    config["classifier"] = {**(config.get("classifier") or {}), "use_local": False}
    return GatewayPolicy(config)


def parse_distill_json(raw_text: str) -> tuple[str, float, str]:
    text = str(raw_text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:120]}")
    payload = json.loads(match.group(0))
    profile = str(payload.get("profile") or "").strip().lower()
    if profile not in VALID_PROFILES:
        raise ValueError(f"Invalid profile: {profile!r}")
    conf = float(payload.get("confidence", 0.0))
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"Confidence out of range: {conf}")
    rationale = str(payload.get("rationale") or "").strip()[:200]
    return profile, conf, rationale


def _call_chat_completion(url: str, api_key: str, model: str, task: str, timeout: int = 30) -> tuple[str, float, str]:
    body = json.dumps({
        "model": model,
        "temperature": 0.0,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Opening prompt:\n{task[:2000]}"},
        ],
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "nexus-router-eval/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    return parse_distill_json(content)


def distill_prompts(
    *,
    input_path: Path,
    output_path: Path,
    config_path: Path = DEFAULT_GATEWAY_CONFIG,
    provider: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    review_threshold: float = 0.85,
    caller=_call_chat_completion,
) -> dict[str, Any]:
    """Distill 3-tier gateway effort labels for extracted prompts, resuming by id."""
    items = _read_jsonl(input_path)
    if not items:
        raise ValueError(f"No prompts found in {input_path}")

    existing_rows = _read_jsonl(output_path)
    done_ids = {str(r.get("id")) for r in existing_rows if r.get("id")}
    pending = [r for r in items if str(r.get("id")) not in done_ids]
    if limit is not None and limit >= 0:
        pending = pending[:limit]

    target = resolve_distill_target(provider=provider, model=model)
    if dry_run:
        return {
            "mode": "dry_run",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_input": len(items),
            "already_done": len(done_ids),
            "pending": len(pending),
            "provider": target[0] if target else None,
            "model": target[3] if target else None,
            "review_threshold": review_threshold,
            "rubric_profiles": list(VALID_PROFILES),
        }

    if target is None:
        raise RuntimeError("No API key found for openai, nvidia, openrouter, or opencode")
    prov_name, url, api_key, resolved_model = target
    policy = _build_offline_policy(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    errors = 0
    last_error: str | None = None
    needs_review_count = 0
    profile_counts = {p: 0 for p in VALID_PROFILES}

    with output_path.open("a") as out:
        for item in pending:
            task = str(item["task"])
            rules_res = policy.classify({"model": policy.alias, "messages": [{"role": "user", "content": task}]}, {}, "offline")
            try:
                profile, confidence, rationale = caller(url, api_key, resolved_model, task)
            except (urllib.error.URLError, ValueError, TimeoutError, json.JSONDecodeError) as exc:
                errors += 1
                code = getattr(exc, "code", None)
                last_error = f"{type(exc).__name__}({code})" if code is not None else type(exc).__name__
                continue
            needs_review = bool(confidence < review_threshold or profile != rules_res.profile)
            if needs_review:
                needs_review_count += 1
            profile_counts[profile] += 1
            record = {
                "id": item["id"],
                "source": item.get("source", "unknown"),
                "task": task,
                "profile": profile,
                "confidence": round(confidence, 4),
                "rationale": rationale,
                "rules_profile": rules_res.profile,
                "rules_source": rules_res.explanation.get("source"),
                "needs_review": needs_review,
                "distill_provider": prov_name,
                "distill_model": resolved_model,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            written += 1

    summary: dict[str, Any] = {
        "mode": "distilled",
        "output_path": str(output_path),
        "provider": prov_name,
        "model": resolved_model,
        "already_done": len(done_ids),
        "attempted": len(pending),
        "written": written,
        "errors": errors,
        "needs_review": needs_review_count,
        "profile_counts": profile_counts,
    }
    if last_error is not None:
        summary["last_error"] = last_error
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare offline gateway effort evaluation datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="Extract opening prompts from Claude/Codex history")
    p_ext.add_argument("--claude-history", type=Path, default=DEFAULT_CLAUDE_HISTORY)
    p_ext.add_argument("--codex-history", type=Path, default=DEFAULT_CODEX_HISTORY)
    p_ext.add_argument("--extra-jsonl", type=Path, default=None)
    p_ext.add_argument("--output", type=Path, default=DEFAULT_EVAL_DIR / "prompts.jsonl")
    p_ext.add_argument("--max-chars", type=int, default=2000)

    p_dis = sub.add_parser("distill", help="Distill 3-tier effort labels using frontier-model rubric")
    p_dis.add_argument("--input", type=Path, default=DEFAULT_EVAL_DIR / "prompts.jsonl")
    p_dis.add_argument("--output", type=Path, default=DEFAULT_EVAL_DIR / "distilled.jsonl")
    p_dis.add_argument("--config", type=Path, default=DEFAULT_GATEWAY_CONFIG)
    p_dis.add_argument("--provider", choices=["openai", "nvidia", "openrouter", "opencode"], default=None)
    p_dis.add_argument("--model", default=None)
    p_dis.add_argument("--limit", type=int, default=None)
    p_dis.add_argument("--review-threshold", type=float, default=0.85)
    p_dis.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "extract":
        summary = extract_prompts(
            claude_history=args.claude_history,
            codex_history=args.codex_history,
            extra_jsonl=args.extra_jsonl,
            output_path=args.output,
            max_chars=args.max_chars,
        )
    else:
        summary = distill_prompts(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            provider=args.provider,
            model=args.model,
            limit=args.limit,
            dry_run=args.dry_run,
            review_threshold=args.review_threshold,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
