"""
health.py — Runtime provider health manager.

Reads and updates state/runtime-health.json.
Computes a composite health score (0.0–1.0) per provider.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import ProviderHealth

HEALTH_FILE = Path(__file__).parent.parent / "state" / "runtime-health.json"

# Hard thresholds
AUTH_STATUSES_OK       = {"ok"}
AUTH_STATUSES_BLOCKED  = {"expired", "missing"}
QUOTA_BLOCKED          = {"exhausted"}
QUOTA_PENALIZED        = {"low"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_provider_health(providers: Optional[list[str]] = None) -> dict[str, ProviderHealth]:
    """
    Load runtime health from state file.
    Returns a dict of provider_id → ProviderHealth.
    """
    if not HEALTH_FILE.exists():
        return {}

    with open(HEALTH_FILE) as f:
        raw = json.load(f)

    result = {}
    for pid, data in raw.get("providers", {}).items():
        if providers and pid not in providers:
            continue
        ph = ProviderHealth(
            provider=pid,
            auth=data.get("auth", "unknown"),
            quota=data.get("quota", "unknown"),
            recent_error_rate=data.get("recentErrorRate", 0.0),
            rate_limit_risk=data.get("rateLimitRisk", 0.0),
            latency_ms_p50=data.get("latencyMsP50"),
            last_failure_at=data.get("lastFailureAt"),
            health_score=_compute_health_score(data),
        )
        result[pid] = ph

    return result


def _compute_health_score(data: dict) -> float:
    """
    Compute a composite health score from raw provider state.

    Scoring logic:
    - Auth blocked (expired/missing)  → 0.0
    - Quota exhausted                 → 0.0
    - Auth unknown                    → 0.80 (give benefit of the doubt)
    - Quota low                       → -0.15 penalty
    - High error rate                 → proportional penalty
    - High rate limit risk            → proportional penalty
    - High latency                    → mild penalty
    """
    auth  = data.get("auth", "unknown")
    quota = data.get("quota", "unknown")
    err   = data.get("recentErrorRate", 0.0)
    ratelimit = data.get("rateLimitRisk", 0.0)
    latency   = data.get("latencyMsP50")

    # Hard blocks
    if auth in AUTH_STATUSES_BLOCKED:
        return 0.0
    if quota in QUOTA_BLOCKED:
        return 0.0

    score = 1.0

    # Unknown auth — slight penalty
    if auth == "unknown":
        score -= 0.10

    # Low quota — penalty
    if quota in QUOTA_PENALIZED:
        score -= 0.15

    # Error rate penalty (linear, capped at -0.40)
    score -= min(err * 0.40, 0.40)

    # Rate limit risk penalty (linear, capped at -0.20)
    score -= min(ratelimit * 0.20, 0.20)

    # Latency penalty (mild; only above 5000ms)
    if latency and latency > 5000:
        score -= min((latency - 5000) / 20000, 0.10)

    return round(max(score, 0.0), 4)


def record_observation(
    provider: str,
    auth_status: str,
    quota_state: str = "unknown",
    error_type: Optional[str] = None,
    latency_ms: Optional[int] = None,
    http_status: Optional[int] = None,
):
    """
    Update runtime-health.json with a new observation for a provider.
    Also updates the derived health_score.
    """
    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)

    if HEALTH_FILE.exists():
        with open(HEALTH_FILE) as f:
            raw = json.load(f)
    else:
        raw = {"providers": {}}

    pdata = raw.get("providers", {}).get(provider, {})

    pdata["auth"]         = auth_status
    pdata["quota"]        = quota_state
    pdata["lastCheckAt"]  = _now_iso()

    if error_type:
        pdata["lastFailureAt"] = _now_iso()
        # Rolling error rate: EWMA with alpha=0.3
        prev = pdata.get("recentErrorRate", 0.0)
        pdata["recentErrorRate"] = round(0.3 * 1.0 + 0.7 * prev, 4)
    else:
        prev = pdata.get("recentErrorRate", 0.0)
        pdata["recentErrorRate"] = round(0.3 * 0.0 + 0.7 * prev, 4)

    if http_status == 429:
        prev = pdata.get("rateLimitRisk", 0.0)
        pdata["rateLimitRisk"] = round(0.3 * 1.0 + 0.7 * prev, 4)
    else:
        prev = pdata.get("rateLimitRisk", 0.0)
        pdata["rateLimitRisk"] = round(0.3 * 0.0 + 0.7 * prev, 4)

    if latency_ms is not None:
        # Simple EWMA for latency
        prev = pdata.get("latencyMsP50")
        if prev is None:
            pdata["latencyMsP50"] = latency_ms
        else:
            pdata["latencyMsP50"] = round(0.3 * latency_ms + 0.7 * prev, 1)

    # Recompute health score
    pdata["healthScore"] = _compute_health_score(pdata)

    raw.setdefault("providers", {})[provider] = pdata
    raw["_updated_at"] = _now_iso()

    with open(HEALTH_FILE, "w") as f:
        json.dump(raw, f, indent=2)
