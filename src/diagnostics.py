"""GET /diagnostics: one read-only snapshot of everything that makes routing go wrong.

Used by the plugin's `/route doctor` and the Nexus Studio router panel.
Holds no prompt text, credentials or session keys.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import cooldowns, db
from .health import load_provider_health
from .paths import LOCAL_CLASSIFIER_DIR, POLICIES_ROOT, RAW_CATALOG_FILE

STALE_REGISTRY = timedelta(hours=48)
STALE_HEALTH = timedelta(hours=1)


def _mtime(path: Path):
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _iso(value):
    return value.isoformat() if value else None


def _parse(value):
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _database() -> dict:
    try:
        with closing(db._connect()) as conn:
            version = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            row = conn.execute(
                "SELECT COUNT(*) AS total, MAX(created_at) AS last_at, "
                "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS week, "
                "SUM(CASE WHEN created_at >= ? AND outcome_success IS NOT NULL THEN 1 ELSE 0 END) AS week_outcomes "
                "FROM routing_decisions",
                (since, since),
            ).fetchone()
        week = row["week"] or 0
        return {
            "path": str(db.DB_PATH),
            "schema_version": version["value"] if version else None,
            "decisions": row["total"],
            "last_decision_at": row["last_at"],
            "decisions_7d": week,
            "outcome_fill_7d": round((row["week_outcomes"] or 0) / week, 3) if week else None,
        }
    except sqlite3.Error as exc:
        return {"path": str(db.DB_PATH), "error": str(exc)}


def _policies() -> dict:
    status = {}
    for path in sorted(Path(POLICIES_ROOT).glob("*.yaml")):
        try:
            yaml.safe_load(path.read_text())
            status[path.name] = "ok"
        except (OSError, yaml.YAMLError) as exc:
            status[path.name] = f"error: {exc}"
    return status


def _classifier() -> dict:
    try:
        from .local_classifier import get_local_classifier

        error = get_local_classifier().load_error
    except Exception as exc:  # optional ML deps missing
        error = f"import_failed: {exc}"
    return {"local_available": not error, "local_error": error, "artifact_dir": str(LOCAL_CLASSIFIER_DIR)}


def collect(router=None, cli_proxy=None) -> dict:
    now = datetime.now(timezone.utc)
    warnings = []

    registry = {"path": None, "updated_at": None, "models": None, "authed_models": None}
    if router is not None:
        updated = _mtime(router.registry_path)
        registry = {
            "path": str(router.registry_path),
            "updated_at": _iso(updated),
            "models": len(router._registry),
            "authed_models": sum(1 for m in router._registry if (m.get("availability") or {}).get("authed")),
        }
        if updated is None or now - updated > STALE_REGISTRY:
            warnings.append("model registry is older than 48h; run the catalog/registry refresh")
    catalog_updated = _mtime(RAW_CATALOG_FILE)

    health = load_provider_health()
    controls = db.load_providers()
    providers = {}
    for name in sorted(set(health) | set(controls)):
        h, c = health.get(name), controls.get(name, {})
        providers[name] = {
            "enabled": bool(c.get("enabled")),
            "status": c.get("status"),
            "health_status": c.get("health_status"),
            "health_score": h.health_score if h else None,
            "auth": h.auth if h else None,
            "quota": h.quota if h else None,
            "cooldown_until": (h.rate_limit_cooldown_until if h else None) or c.get("cooldown_until"),
            "last_check_at": h.last_check_at if h else None,
        }
        if c.get("enabled") and h and h.health_score < 0.3:
            warnings.append(f"provider {name} is enabled but unhealthy ({h.health_score:.2f})")
    checks = [t for t in (_parse(p["last_check_at"]) for p in providers.values()) if t]
    last_check = max(checks) if checks else None
    if last_check is None or now - last_check > STALE_HEALTH:
        warnings.append("provider health/quota not refreshed in the last hour (quota-sync timer?)")

    policies = _policies()
    warnings += [f"policy {name}: {state}" for name, state in policies.items() if state != "ok"]
    classifier = _classifier()
    if not classifier["local_available"]:
        warnings.append("local classifier unavailable; routing falls back to Jev/heuristics")
    database = _database()
    if "error" in database:
        warnings.append(f"database: {database['error']}")

    experiment = ((router._routing if router is not None else {}) or {}).get("experiment") or {}
    return {
        "generated_at": now.isoformat(),
        "warnings": warnings,
        "registry": registry,
        "raw_catalog_updated_at": _iso(catalog_updated),
        "classifier": classifier,
        "database": database,
        "policies": policies,
        "providers": providers,
        "quota_sync_last_check_at": _iso(last_check),
        "model_cooldowns": cooldowns.active(),
        "experiment": {
            "id": experiment.get("id"),
            "split": experiment.get("split", 0.5) if experiment.get("id") else None,
            "variants": list(experiment.get("variants") or {})[:2],
        } if experiment.get("id") else None,
        "cli_proxy": cli_proxy.summary() if cli_proxy is not None else None,
    }
