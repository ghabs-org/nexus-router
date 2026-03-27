"""
health_updater.py — Provider health updater for Nexus Router.

Reads health signals from OpenClaw (via model probe or response observation)
and writes them to state/runtime-health.json and provider_health_log in SQLite.

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


def observe_turn_outcome(
    provider: str,
    http_status: Optional[int],
    latency_ms: Optional[int],
    error_type: Optional[str],
    quota_hint: Optional[str] = None,
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

    # Determine quota state
    if http_status == 429:
        quota_state = "low"
    elif quota_hint:
        quota_state = quota_hint
    else:
        quota_state = "unknown"

    record_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state,
        error_type=error_type,
        latency_ms=latency_ms,
        http_status=http_status,
    )

    log_provider_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state,
        http_status=http_status,
        error_type=error_type,
        latency_ms=latency_ms,
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
            record_observation(provider=provider, auth_status="unknown")
            return {"provider": provider, "auth": "unknown", "error": result.stderr.strip()}

        data = json.loads(result.stdout)

        # Extract auth status from OpenClaw output
        auth_providers = data.get("auth", {}).get("providers", {})
        pauth = auth_providers.get(provider, {})
        auth_ok = pauth.get("ok", False)
        auth_status = "ok" if auth_ok else "expired"

        record_observation(
            provider=provider,
            auth_status=auth_status,
            quota_state="unknown",
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
        quota_hint="healthy",
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
