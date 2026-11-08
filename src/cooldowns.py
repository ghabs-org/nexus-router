"""Per-model circuit breaker, persisted in SQLite.

Provider-wide trouble (429, auth) stays in provider health. This catches one
model failing on one scope (advisory task type, or CLI call category) while
the rest of its provider is fine: N failures inside a window -> cooldown.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import yaml

from . import db
from .paths import POLICIES_ROOT

DEFAULTS = {"failures": 3, "window_seconds": 60, "cooldown_seconds": 300}


def config() -> dict:
    """Thresholds from routing.yaml's `cooldowns`, re-read when the file changes."""
    path = POLICIES_ROOT / "routing.yaml"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    return _config(mtime)


@lru_cache(maxsize=4)
def _config(mtime) -> dict:
    try:
        with open(POLICIES_ROOT / "routing.yaml") as file:
            raw = (yaml.safe_load(file) or {}).get("cooldowns") or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    return {key: float(raw.get(key, value)) for key, value in DEFAULTS.items()}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_failure(model: str, scope: str) -> bool:
    """Count one failure; returns True when this failure opened a cooldown."""
    cfg, now = config(), _now()
    with closing(db._connect()) as conn, conn:
        row = conn.execute(
            "SELECT failures, window_started_at FROM model_cooldowns WHERE model=? AND scope=?",
            (model, scope),
        ).fetchone()
        failures, started = 1, now
        if row and now - datetime.fromisoformat(row["window_started_at"]) <= timedelta(seconds=cfg["window_seconds"]):
            failures, started = row["failures"] + 1, datetime.fromisoformat(row["window_started_at"])
        tripped = failures >= cfg["failures"]
        conn.execute(
            """INSERT INTO model_cooldowns (model, scope, failures, window_started_at, cooldown_until)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(model, scope) DO UPDATE SET failures=excluded.failures,
                 window_started_at=excluded.window_started_at,
                 cooldown_until=COALESCE(excluded.cooldown_until, model_cooldowns.cooldown_until)""",
            (
                model, scope,
                0 if tripped else failures,  # after a trip the next window starts clean
                (now if tripped else started).isoformat(),
                (now + timedelta(seconds=cfg["cooldown_seconds"])).isoformat() if tripped else None,
            ),
        )
    return tripped


def record_success(model: str, scope: str) -> None:
    with closing(db._connect()) as conn, conn:
        conn.execute("DELETE FROM model_cooldowns WHERE model=? AND scope=?", (model, scope))


def active(scope: str | None = None) -> list[dict]:
    """Cooldowns still in force, optionally for one scope."""
    sql = "SELECT model, scope, cooldown_until FROM model_cooldowns WHERE cooldown_until > ?"
    args: tuple = (_now().isoformat(),)
    if scope is not None:
        sql, args = sql + " AND scope=?", args + (scope,)
    try:
        with closing(db._connect()) as conn:
            return [dict(row) for row in conn.execute(sql + " ORDER BY cooldown_until", args)]
    except sqlite3.OperationalError:  # schema not created yet: nothing is cooling
        return []


def cooling_models(scope: str) -> set[str]:
    return {row["model"] for row in active(scope)}


if __name__ == "__main__":  # self-check against a throwaway DB
    import tempfile
    from pathlib import Path

    db.DB_PATH = Path(tempfile.mkdtemp()) / "router.sqlite"
    db.ensure_schema()
    assert not record_failure("p/m", "coding") and not record_failure("p/m", "coding")
    assert record_failure("p/m", "coding") and cooling_models("coding") == {"p/m"}
    assert cooling_models("reasoning") == set()
    record_success("p/m", "coding")
    assert cooling_models("coding") == set()
    print("cooldowns ok")
