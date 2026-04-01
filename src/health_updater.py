"""
health_updater.py — Provider health updater for Nexus Router.

Reads health signals from OpenClaw (via model probe or response observation)
and writes them to provider health state + provider_health_log in SQLite.

Integration points:
1. Passive: called after each router turn with actual response outcome
2. Active: called periodically to probe provider auth status
"""

import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .health import record_observation
from .db import log_provider_observation

OPENCLAW_BIN = "openclaw"

# Providers we actively manage
MANAGED_PROVIDERS = [
    "openai-codex",
    "github-copilot",
    "google-gemini-cli",
]


def _infer_probe_failure(stdout: str, stderr: str) -> tuple[str, Optional[str], Optional[str], Optional[int]]:
    combined = f"{stdout}\n{stderr}".lower()

    if "429" in combined or "rate limit" in combined or "quota" in combined or "capacity" in combined:
        exhausted = any(token in combined for token in ["quota exhausted", "quota exceeded", "no capacity available", "capacity exhausted", "exhausted"])
        return "ok", "exhausted" if exhausted else "low", "rate_limit", 429
    if "401" in combined or "403" in combined or "unauthorized" in combined or "forbidden" in combined:
        return "expired", None, "auth", 401
    if "timeout" in combined:
        return "unknown", None, "timeout", None
    return "unknown", None, "unknown", None


def observe_turn_outcome(
    provider: str,
    http_status: Optional[int],
    latency_ms: Optional[int],
    error_type: Optional[str],
    quota_hint: Optional[str] = None,
    quota_remaining_ratio: Optional[float] = None,
):
    """
    Called after each model turn to passively update provider health.
    Should be called by the OpenClaw adapter after every response.

    Args:
        provider:     Provider id, e.g. 'openai-codex'
        http_status:  HTTP response code if available (e.g. 429, 500, 200)
        latency_ms:   Observed latency for this turn
        error_type:   'rate_limit' | 'auth' | 'server' | 'timeout' | None
        quota_hint:   Optional quota signal from response headers
    """
    # Determine auth status from response
    if http_status in (401, 403):
        auth_status = "expired"
    elif http_status and http_status < 400:
        auth_status = "ok"
    else:
        auth_status = "unknown"

    normalized_hint = (quota_hint or "").strip().lower() or None

    # Determine quota state
    if normalized_hint in {"exhausted", "depleted", "empty"}:
        quota_state = "exhausted"
    elif normalized_hint in {"low", "limited", "throttled"}:
        quota_state = "low"
    elif http_status == 429:
        quota_state = "exhausted" if error_type in {"rate_limit_exhausted", "capacity", "capacity_exhausted"} else "low"
    elif normalized_hint:
        quota_state = normalized_hint
    else:
        quota_state = None

    record_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state,
        error_type=error_type,
        latency_ms=latency_ms,
        http_status=http_status,
        quota_remaining_ratio=quota_remaining_ratio,
    )

    log_provider_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state or "unknown",
        http_status=http_status,
        error_type=error_type,
        latency_ms=latency_ms,
        note=(f"quota_ratio={quota_remaining_ratio:.3f}" if quota_remaining_ratio is not None else None),
    )


def probe_provider_auth(provider: str) -> dict:
    """
    Run openclaw models status --probe-provider <provider> and
    parse the result to update health state.

    Returns a summary dict with auth/quota state.
    """
    try:
        result = subprocess.run(
            [OPENCLAW_BIN, "models", "status", "--probe-provider", provider, "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            auth_status, quota_state, error_type, http_status = _infer_probe_failure(result.stdout, result.stderr)
            record_observation(
                provider=provider,
                auth_status=auth_status,
                quota_state=quota_state,
                error_type=error_type,
                http_status=http_status,
            )
            log_provider_observation(
                provider=provider,
                auth_status=auth_status,
                quota_state=quota_state or "unknown",
                http_status=http_status,
                error_type=error_type,
                note=f"probe: returncode={result.returncode}",
            )
            summary = {"provider": provider, "auth": auth_status, "error": result.stderr.strip()}
            if quota_state:
                summary["quota"] = quota_state
            return summary

        data = json.loads(result.stdout)

        # Extract auth status from OpenClaw output.
        # Newer OpenClaw returns auth.providers as a list of provider objects,
        # older versions may expose a dict-like structure.
        auth_block = data.get("auth", {}) if isinstance(data, dict) else {}
        auth_providers = auth_block.get("providers", [])
        pauth = None
        if isinstance(auth_providers, list):
            for item in auth_providers:
                if isinstance(item, dict) and item.get("provider") == provider:
                    pauth = item
                    break
        elif isinstance(auth_providers, dict):
            pauth = auth_providers.get(provider, {})
        if pauth is None:
            pauth = {}

        profiles = pauth.get("profiles") if isinstance(pauth, dict) else None
        effective = pauth.get("effective") if isinstance(pauth, dict) else None
        has_profiles = isinstance(profiles, dict) and int(profiles.get("count", 0) or 0) > 0
        auth_ok = bool(has_profiles or effective)
        auth_status = "ok" if auth_ok else "expired"

        record_observation(
            provider=provider,
            auth_status=auth_status,
            quota_state=None,
        )
        log_provider_observation(
            provider=provider,
            auth_status=auth_status,
            note=f"probe: returncode={result.returncode}",
        )
        return {"provider": provider, "auth": auth_status}

    except subprocess.TimeoutExpired:
        record_observation(provider=provider, auth_status="unknown", error_type="timeout")
        return {"provider": provider, "auth": "unknown", "error": "timeout"}
    except Exception as e:
        record_observation(provider=provider, auth_status="unknown")
        return {"provider": provider, "auth": "unknown", "error": str(e)}


def probe_all_providers(providers: Optional[list[str]] = None) -> list[dict]:
    """
    Probe all managed providers and update health state.
    Safe to call periodically (e.g. every 15 minutes).
    """
    targets = providers or MANAGED_PROVIDERS
    results = []
    for p in targets:
        result = probe_provider_auth(p)
        results.append(result)
        print(f"  [{p}] auth={result.get('auth', '?')}")
    return results


def mark_provider_healthy(provider: str, latency_ms: Optional[int] = None):
    """Shorthand: mark a provider as healthy after a successful turn."""
    observe_turn_outcome(
        provider=provider,
        http_status=200,
        latency_ms=latency_ms,
        error_type=None,
        quota_hint=None,
    )


def mark_provider_rate_limited(provider: str):
    """Shorthand: mark a provider as rate-limited."""
    observe_turn_outcome(
        provider=provider,
        http_status=429,
        latency_ms=None,
        error_type="rate_limit",
        quota_hint="low",
    )


def mark_provider_auth_failed(provider: str):
    """Shorthand: mark a provider as auth-failed."""
    observe_turn_outcome(
        provider=provider,
        http_status=401,
        latency_ms=None,
        error_type="auth",
    )
