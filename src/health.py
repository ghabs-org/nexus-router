"""
health.py — Runtime provider health manager.

Reads and updates provider health state in router SQLite.
Computes a composite health score (0.0–1.0) per provider.
"""

from datetime import timedelta, datetime, timezone
from typing import Optional

from .types import ProviderHealth
from .db import load_provider_health_state, load_providers, upsert_provider_health_state

AUTH_STATUSES_OK = {"ok"}
AUTH_STATUSES_BLOCKED = {"expired", "missing"}
QUOTA_BLOCKED = {"exhausted"}
QUOTA_PENALIZED = {"low"}
RATE_LIMIT_SOFT_BAN_WINDOWS = {
    1: timedelta(minutes=5),
    2: timedelta(minutes=15),
}
RATE_LIMIT_SOFT_BAN_MAX = timedelta(hours=1)
LATENCY_FRESHNESS_WINDOW = timedelta(minutes=30)
LATENCY_STALE_DECAY_WINDOW = timedelta(hours=6)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    parsed = _parse_iso8601(_now_iso())
    if parsed is not None:
        return parsed
    return datetime.now(timezone.utc)


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _soft_ban_window_for_streak(streak: int) -> timedelta:
    if streak <= 0:
        return timedelta(0)
    return RATE_LIMIT_SOFT_BAN_WINDOWS.get(streak, RATE_LIMIT_SOFT_BAN_MAX)


def _row_to_provider_health(pid: str, data: dict) -> ProviderHealth:
    return ProviderHealth(
        provider=pid,
        auth=data.get("auth", "unknown"),
        quota=data.get("quota", "unknown"),
        quota_remaining_ratio=data.get("quota_remaining_ratio"),
        recent_error_rate=float(data.get("recent_error_rate") or 0.0),
        rate_limit_risk=float(data.get("rate_limit_risk") or 0.0),
        consecutive_rate_limits=int(data.get("consecutive_rate_limits") or 0),
        rate_limit_cooldown_until=data.get("rate_limit_cooldown_until"),
        latency_ms_p50=data.get("latency_ms_p50"),
        last_failure_at=data.get("last_failure_at"),
        last_check_at=data.get("last_check_at"),
        health_score=float(data.get("health_score") or 0.0),
    )


def load_provider_health(providers: Optional[list[str]] = None) -> dict[str, ProviderHealth]:
    rows = load_provider_health_state(providers)
    provider_controls = load_providers(providers)
    result: dict[str, ProviderHealth] = {}

    # Include provider-control rows even when they have no health sample yet,
    # otherwise disabled providers absent from provider_health_state would be
    # treated by the scorer as unknown-but-healthy.
    for pid, control in provider_controls.items():
        rows.setdefault(pid, {
            "provider": pid,
            "auth": control.get("auth_status", "unknown"),
            "quota": control.get("quota_status", "unknown"),
            "quota_remaining_ratio": control.get("quota_remaining_ratio"),
            "recent_error_rate": control.get("recent_error_rate", 0.0),
            "rate_limit_risk": control.get("rate_limit_risk", 0.0),
            "consecutive_rate_limits": control.get("consecutive_rate_limits", 0),
            "rate_limit_cooldown_until": control.get("cooldown_until"),
            "latency_ms_p50": control.get("latency_ms_p50"),
            "latency_updated_at": control.get("latency_updated_at"),
            "last_failure_at": control.get("last_failure_at"),
            "last_check_at": control.get("last_check_at"),
            "health_score": control.get("health_score", 1.0),
        })

    for pid, data in rows.items():
        control = provider_controls.get(pid) or {}
        provider_enabled = bool(control.get("enabled", 1))
        provider_status = str(control.get("status") or "enabled").lower()
        if not provider_enabled or provider_status in {"disabled", "maintenance"}:
            result[pid] = ProviderHealth(
                provider=pid,
                auth=data.get("auth", "unknown"),
                quota=data.get("quota", "unknown"),
                quota_remaining_ratio=data.get("quota_remaining_ratio"),
                recent_error_rate=float(data.get("recent_error_rate") or 0.0),
                rate_limit_risk=float(data.get("rate_limit_risk") or 0.0),
                consecutive_rate_limits=int(data.get("consecutive_rate_limits") or 0),
                rate_limit_cooldown_until=data.get("rate_limit_cooldown_until"),
                latency_ms_p50=data.get("latency_ms_p50"),
                last_failure_at=data.get("last_failure_at"),
                last_check_at=data.get("last_check_at"),
                health_score=0.0,
            )
            continue

        payload = {
            "auth": data.get("auth", "unknown"),
            "quota": data.get("quota", "unknown"),
            "quotaRemainingRatio": data.get("quota_remaining_ratio"),
            "recentErrorRate": data.get("recent_error_rate", 0.0),
            "rateLimitRisk": data.get("rate_limit_risk", 0.0),
            "consecutiveRateLimits": data.get("consecutive_rate_limits", 0),
            "rateLimitCooldownUntil": data.get("rate_limit_cooldown_until"),
            "latencyMsP50": data.get("latency_ms_p50"),
            "lastFailureAt": data.get("last_failure_at"),
            "lastCheckAt": data.get("last_check_at"),
            "latencyUpdatedAt": data.get("latency_updated_at"),
        }
        payload["healthScore"] = _compute_health_score(payload)
        if abs(float(data.get("health_score") or 0.0) - payload["healthScore"]) > 1e-9:
            upsert_provider_health_state(pid, {
                "auth": payload["auth"],
                "quota": payload["quota"],
                "quota_remaining_ratio": payload["quotaRemainingRatio"],
                "recent_error_rate": payload["recentErrorRate"],
                "rate_limit_risk": payload["rateLimitRisk"],
                "consecutive_rate_limits": payload["consecutiveRateLimits"],
                "rate_limit_cooldown_until": payload["rateLimitCooldownUntil"],
                "latency_ms_p50": payload["latencyMsP50"],
                "last_failure_at": payload["lastFailureAt"],
                "last_check_at": payload["lastCheckAt"],
                "latency_updated_at": payload["latencyUpdatedAt"],
                "health_score": payload["healthScore"],
            })
            data = load_provider_health_state([pid]).get(pid, data)
        result[pid] = _row_to_provider_health(pid, data)
    return result


def _compute_health_score(data: dict) -> float:
    auth = data.get("auth", "unknown")
    quota = data.get("quota", "unknown")
    err = data.get("recentErrorRate", 0.0)
    ratelimit = data.get("rateLimitRisk", 0.0)
    latency = data.get("latencyMsP50")
    quota_ratio = data.get("quotaRemainingRatio")
    last_check_at = data.get("lastCheckAt")
    latency_updated_at = data.get("latencyUpdatedAt") or last_check_at
    cooldown_until = _parse_iso8601(data.get("rateLimitCooldownUntil"))

    latency_penalty_latency = latency

    if auth in AUTH_STATUSES_BLOCKED:
        return 0.0
    if quota in QUOTA_BLOCKED:
        return 0.0
    if cooldown_until and cooldown_until > _now_dt():
        return 0.0

    score = 1.0
    if auth == "unknown":
        score -= 0.10

    if quota_ratio is not None:
        try:
            ratio = max(0.0, min(1.0, float(quota_ratio)))
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None:
            if ratio <= 0.02:
                return 0.0
            if ratio <= 0.10:
                score -= 0.55
            elif ratio <= 0.25:
                score -= 0.35
            elif ratio <= 0.50:
                score -= 0.15
    elif quota in QUOTA_PENALIZED:
        score -= 0.35

    score -= min(float(err or 0.0) * 0.40, 0.40)
    score -= min(float(ratelimit or 0.0) * 0.25, 0.25)

    latency_check_dt = _parse_iso8601(latency_updated_at)
    if latency_check_dt:
        try:
            latency_age = _now_dt() - latency_check_dt
            if latency_age > LATENCY_FRESHNESS_WINDOW:
                if latency_age >= LATENCY_STALE_DECAY_WINDOW:
                    latency_penalty_latency = None
                elif latency_penalty_latency is not None:
                    decay_progress = (latency_age - LATENCY_FRESHNESS_WINDOW) / (LATENCY_STALE_DECAY_WINDOW - LATENCY_FRESHNESS_WINDOW)
                    latency_penalty_latency = float(latency_penalty_latency) * max(0.0, 1.0 - decay_progress)
        except Exception:
            pass

    if last_check_at:
        try:
            last_check_dt = datetime.fromisoformat(last_check_at)
            age_seconds = (_now_dt() - last_check_dt).total_seconds()
            if age_seconds > 1800:
                score -= min((age_seconds - 1800) / 21600, 0.15)
        except Exception:
            pass

    if latency_penalty_latency and latency_penalty_latency > 5000:
        score -= min((latency_penalty_latency - 5000) / 20000, 0.10)

    return round(max(score, 0.0), 4)


def record_observation(
    provider: str,
    auth_status: str,
    quota_state: Optional[str] = None,
    error_type: Optional[str] = None,
    latency_ms: Optional[int] = None,
    http_status: Optional[int] = None,
    quota_remaining_ratio: Optional[float] = None,
):
    existing = load_provider_health_state([provider]).get(provider, {})

    pdata = {
        "auth": auth_status,
        "quota": existing.get("quota", "unknown"),
        "quota_remaining_ratio": existing.get("quota_remaining_ratio"),
        "recent_error_rate": float(existing.get("recent_error_rate") or 0.0),
        "rate_limit_risk": float(existing.get("rate_limit_risk") or 0.0),
        "consecutive_rate_limits": int(existing.get("consecutive_rate_limits") or 0),
        "rate_limit_cooldown_until": existing.get("rate_limit_cooldown_until"),
        "latency_ms_p50": existing.get("latency_ms_p50"),
        "latency_updated_at": existing.get("latency_updated_at"),
        "last_failure_at": existing.get("last_failure_at"),
        "last_check_at": _now_iso(),
    }

    if quota_state is not None:
        pdata["quota"] = quota_state

    if quota_remaining_ratio is not None:
        try:
            ratio = max(0.0, min(1.0, float(quota_remaining_ratio)))
            pdata["quota_remaining_ratio"] = ratio
            if ratio <= 0.02:
                pdata["quota"] = "exhausted"
            elif ratio <= 0.25:
                pdata["quota"] = "low"
            else:
                pdata["quota"] = "healthy"
        except (TypeError, ValueError):
            pass
    elif http_status == 429 and pdata.get("quota") == "low":
        pdata["quota"] = "exhausted"

    if error_type:
        pdata["last_failure_at"] = _now_iso()
        prev = float(existing.get("recent_error_rate") or 0.0)
        pdata["recent_error_rate"] = round(0.3 * 1.0 + 0.7 * prev, 4)
    else:
        prev = float(existing.get("recent_error_rate") or 0.0)
        pdata["recent_error_rate"] = round(0.3 * 0.0 + 0.7 * prev, 4)

    if http_status == 429:
        streak = int(existing.get("consecutive_rate_limits") or 0) + 1
        pdata["consecutive_rate_limits"] = streak
        cooldown_until = _now_dt() + _soft_ban_window_for_streak(streak)
        pdata["rate_limit_cooldown_until"] = cooldown_until.isoformat()
        prev = float(existing.get("rate_limit_risk") or 0.0)
        pdata["rate_limit_risk"] = round(0.3 * 1.0 + 0.7 * prev, 4)
    else:
        pdata["consecutive_rate_limits"] = 0
        pdata["rate_limit_cooldown_until"] = None
        prev = float(existing.get("rate_limit_risk") or 0.0)
        pdata["rate_limit_risk"] = round(0.3 * 0.0 + 0.7 * prev, 4)

    if latency_ms is not None:
        prev = existing.get("latency_ms_p50")
        prev_latency_updated_at = _parse_iso8601(existing.get("latency_updated_at") or existing.get("last_check_at"))
        latency_stale = not prev_latency_updated_at or (_now_dt() - prev_latency_updated_at) > LATENCY_STALE_DECAY_WINDOW
        if prev is None or latency_stale:
            pdata["latency_ms_p50"] = latency_ms
        else:
            pdata["latency_ms_p50"] = round(0.3 * latency_ms + 0.7 * float(prev), 1)
        pdata["latency_updated_at"] = _now_iso()

    payload_for_score = {
        "auth": pdata["auth"],
        "quota": pdata["quota"],
        "quotaRemainingRatio": pdata.get("quota_remaining_ratio"),
        "recentErrorRate": pdata["recent_error_rate"],
        "rateLimitRisk": pdata["rate_limit_risk"],
        "consecutiveRateLimits": pdata["consecutive_rate_limits"],
        "rateLimitCooldownUntil": pdata.get("rate_limit_cooldown_until"),
        "latencyMsP50": pdata.get("latency_ms_p50"),
        "lastFailureAt": pdata.get("last_failure_at"),
        "lastCheckAt": pdata.get("last_check_at"),
        "latencyUpdatedAt": pdata.get("latency_updated_at"),
    }
    pdata["health_score"] = _compute_health_score(payload_for_score)

    upsert_provider_health_state(provider, pdata)
