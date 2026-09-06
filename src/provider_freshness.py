from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .paths import STATE_ROOT

FRESHNESS_STATE_PATH = STATE_ROOT / "reports" / "provider-freshness-state.json"


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _is_freshness_candidate(provider_row: dict[str, Any], health_row: dict[str, Any]) -> tuple[bool, str | None]:
    enabled = bool(provider_row.get("enabled", 1))
    status = str(provider_row.get("status") or "enabled").strip().lower()
    if not enabled or status in {"disabled", "maintenance"}:
        return False, "provider_disabled"

    auth = str((health_row.get("auth") or provider_row.get("auth_status") or "unknown")).strip().lower()
    if auth in {"expired", "missing"}:
        return False, "auth_blocked"

    quota = str((health_row.get("quota") or provider_row.get("quota_status") or "unknown")).strip().lower()
    if quota == "exhausted":
        return False, "quota_exhausted"

    health_score = health_row.get("health_score", provider_row.get("health_score"))
    try:
        if health_score is not None and float(health_score) <= 0.0:
            return False, "health_zero"
    except Exception:
        pass

    return True, None


def build_provider_freshness_snapshot(
    provider_rows: dict[str, dict[str, Any]],
    health_rows: dict[str, dict[str, Any]],
    stale_after_hours: int = 24,
) -> dict[str, Any]:
    stale_cutoff = _now_dt() - timedelta(hours=stale_after_hours)

    stale_candidates: list[str] = []
    exempt_reasons: dict[str, int] = {
        "provider_disabled": 0,
        "auth_blocked": 0,
        "quota_exhausted": 0,
        "health_zero": 0,
    }

    all_providers = sorted(set(provider_rows.keys()) | set(health_rows.keys()))

    for provider in all_providers:
        prow = provider_rows.get(provider, {"provider": provider})
        hrow = health_rows.get(provider, {"provider": provider})
        include, reason = _is_freshness_candidate(prow, hrow)
        if not include:
            if reason:
                exempt_reasons[reason] = exempt_reasons.get(reason, 0) + 1
            continue

        updated_at = hrow.get("latency_updated_at") or hrow.get("last_check_at")
        updated_dt = _parse_iso8601(str(updated_at)) if updated_at else None
        if updated_dt is None or updated_dt <= stale_cutoff:
            stale_candidates.append(provider)

    return {
        "stale_after_hours": stale_after_hours,
        "candidate_providers": len(all_providers) - sum(exempt_reasons.values()),
        "stale_candidate_count": len(stale_candidates),
        "stale_candidates": stale_candidates,
        "exempted_providers": sum(exempt_reasons.values()),
        "exempt_reasons": exempt_reasons,
    }


def _load_previous_state(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def evaluate_freshness_transitions(
    provider_rows: dict[str, dict[str, Any]],
    health_rows: dict[str, dict[str, Any]],
    stale_after_hours: int = 24,
    state_path: Path = FRESHNESS_STATE_PATH,
) -> dict[str, Any]:
    snapshot = build_provider_freshness_snapshot(provider_rows, health_rows, stale_after_hours=stale_after_hours)
    prev = _load_previous_state(state_path)
    prev_stale = set(str(p) for p in (prev.get("stale_candidates") or []))
    current_stale = set(snapshot.get("stale_candidates") or [])

    newly_stale = sorted(current_stale - prev_stale)
    recovered = sorted(prev_stale - current_stale)

    payload = {
        **snapshot,
        "newly_stale": newly_stale,
        "recovered": recovered,
        "state_file": str(state_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
