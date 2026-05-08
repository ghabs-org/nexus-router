"""
db.py — SQLite interaction layer for Nexus Router.

Handles:
- Initialising the database (from schema.sql)
- Writing routing decisions
- Updating decision outcomes
- Reading learned model stats
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from .types import ClassifierOutput, PreSignals, ProviderHealth, RoutingDecision
    from .paths import ROUTER_DB_PATH, REGISTRY_FILE, ensure_parent
except ImportError:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent))
    from types import SimpleNamespace as _ignore  # noqa: F401
    from paths import ROUTER_DB_PATH, REGISTRY_FILE, ensure_parent  # type: ignore
    from importlib import util as _importlib_util
    _types_spec = _importlib_util.spec_from_file_location("nexus_router_local_types", _Path(__file__).parent / "types.py")
    _types_mod = _importlib_util.module_from_spec(_types_spec)
    assert _types_spec and _types_spec.loader
    _types_spec.loader.exec_module(_types_mod)
    ClassifierOutput = _types_mod.ClassifierOutput
    PreSignals = _types_mod.PreSignals
    ProviderHealth = _types_mod.ProviderHealth
    RoutingDecision = _types_mod.RoutingDecision

DB_PATH     = ROUTER_DB_PATH
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    ensure_parent(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema():
    """Create tables if they don't exist yet."""
    with open(SCHEMA_PATH) as f:
        sql = f.read()
    conn = _connect()
    conn.executescript(sql)
    _ensure_feedback_table_compat(conn)
    _ensure_routing_decisions_compat(conn)
    _ensure_model_metadata_compat(conn)
    _ensure_route_mode_preferences_compat(conn)
    _ensure_providers_compat(conn)
    conn.commit()
    conn.close()


def _ensure_feedback_table_compat(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(route_feedback)").fetchall()}
    if not columns:
        return
    expected = {
        "corrected_task": "TEXT",
        "model_verdict": "TEXT",
        "preferred_model": "TEXT",
        "reason_tag": "TEXT",
        "source_surface": "TEXT",
        "source_channel": "TEXT",
        "source_message_id": "TEXT",
        "source_user_id": "TEXT",
        "metadata": "TEXT",
    }
    for column, column_type in expected.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE route_feedback ADD COLUMN {column} {column_type}")


def _ensure_route_mode_preferences_compat(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(route_mode_preferences)").fetchall()}
    if not columns:
        return
    expected = {
        "free_filter": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, column_type in expected.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE route_mode_preferences ADD COLUMN {column} {column_type}")


def _ensure_routing_decisions_compat(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
    if not columns:
        return
    expected = {
        "route_mode": "TEXT",
        "mode": "TEXT",
        "source_type": "TEXT",
        "source_tag": "TEXT",
        "message_text": "TEXT",       # truncated user message for classifier training
        "classifier_source": "TEXT",  # llm | heuristic | local | explicit | fallback
    }
    for column, column_type in expected.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE routing_decisions ADD COLUMN {column} {column_type}")

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
    if "mode" in columns:
        conn.execute(
            """
            UPDATE routing_decisions
            SET mode = CASE
                WHEN provenance_mode IS NOT NULL AND TRIM(provenance_mode) != '' THEN provenance_mode
                ELSE 'route'
            END
            WHERE mode IS NULL OR TRIM(mode) = ''
            """
            if "provenance_mode" in columns
            else """
            UPDATE routing_decisions
            SET mode = 'route'
            WHERE mode IS NULL OR TRIM(mode) = ''
            """
        )

    if "source_type" in columns:
        conn.execute(
            """
            UPDATE routing_decisions
            SET source_type = 'standalone'
            WHERE source_type IS NULL OR TRIM(source_type) = ''
            """
        )


def _ensure_providers_compat(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS providers (
          provider                   TEXT PRIMARY KEY,
          enabled                    INTEGER NOT NULL DEFAULT 0,
          status                     TEXT NOT NULL DEFAULT 'unknown',
          priority                   INTEGER DEFAULT 100,
          allow_primary              INTEGER NOT NULL DEFAULT 1,
          allow_fallback             INTEGER NOT NULL DEFAULT 1,
          weight_multiplier          REAL NOT NULL DEFAULT 1.0,
          max_concurrency            INTEGER,
          notes                      TEXT,
          health_status              TEXT NOT NULL DEFAULT 'unknown',
          health_score               REAL DEFAULT 1,
          auth_status                TEXT DEFAULT 'unknown',
          quota_status               TEXT DEFAULT 'unknown',
          quota_remaining_ratio      REAL,
          recent_error_rate          REAL DEFAULT 0,
          rate_limit_risk            REAL DEFAULT 0,
          consecutive_rate_limits    INTEGER DEFAULT 0,
          cooldown_until             TEXT,
          latency_ms_p50             REAL,
          latency_ms_p95             REAL,
          latency_updated_at         TEXT,
          last_check_at              TEXT,
          last_success_at            TEXT,
          last_failure_at            TEXT,
          last_error_type            TEXT,
          last_error_message         TEXT,
          metadata_json              TEXT,
          created_at                 TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at                 TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    expected = {
        "enabled": "INTEGER NOT NULL DEFAULT 0",
        "status": "TEXT NOT NULL DEFAULT 'unknown'",
        "priority": "INTEGER DEFAULT 100",
        "allow_primary": "INTEGER NOT NULL DEFAULT 1",
        "allow_fallback": "INTEGER NOT NULL DEFAULT 1",
        "weight_multiplier": "REAL NOT NULL DEFAULT 1.0",
        "max_concurrency": "INTEGER",
        "notes": "TEXT",
        "health_status": "TEXT NOT NULL DEFAULT 'unknown'",
        "health_score": "REAL DEFAULT 1",
        "auth_status": "TEXT DEFAULT 'unknown'",
        "quota_status": "TEXT DEFAULT 'unknown'",
        "quota_remaining_ratio": "REAL",
        "recent_error_rate": "REAL DEFAULT 0",
        "rate_limit_risk": "REAL DEFAULT 0",
        "consecutive_rate_limits": "INTEGER DEFAULT 0",
        "cooldown_until": "TEXT",
        "latency_ms_p50": "REAL",
        "latency_ms_p95": "REAL",
        "latency_updated_at": "TEXT",
        "last_check_at": "TEXT",
        "last_success_at": "TEXT",
        "last_failure_at": "TEXT",
        "last_error_type": "TEXT",
        "last_error_message": "TEXT",
        "metadata_json": "TEXT",
        "created_at": "TEXT NOT NULL DEFAULT (datetime('now'))",
        "updated_at": "TEXT NOT NULL DEFAULT (datetime('now'))",
    }
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(providers)").fetchall()}
    for column, column_type in expected.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE providers ADD COLUMN {column} {column_type}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_providers_health_status ON providers(health_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_providers_updated_at ON providers(updated_at)")
    _seed_known_providers(conn)
    conn.execute("UPDATE schema_meta SET value='7' WHERE key='schema_version'")
    conn.execute("INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '7')")


def _seed_known_providers(conn: sqlite3.Connection) -> None:
    enabled_defaults = {"anthropic", "openai-codex", "google-gemini-cli", "openrouter", "nvidia"}
    providers = _known_provider_names(conn)
    now = _now_iso()
    for provider in sorted(providers):
        enabled = 1 if provider in enabled_defaults else 0
        status = "enabled" if enabled else "disabled"
        conn.execute(
            """
            INSERT INTO providers (provider, enabled, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO NOTHING
            """,
            (provider, enabled, status, now, now),
        )

    conn.execute(
        """
        UPDATE providers
        SET
          health_score = COALESCE((SELECT health_score FROM provider_health_state phs WHERE phs.provider = providers.provider), health_score),
          auth_status = COALESCE((SELECT auth FROM provider_health_state phs WHERE phs.provider = providers.provider), auth_status, 'unknown'),
          quota_status = COALESCE((SELECT quota FROM provider_health_state phs WHERE phs.provider = providers.provider), quota_status, 'unknown'),
          quota_remaining_ratio = COALESCE((SELECT quota_remaining_ratio FROM provider_health_state phs WHERE phs.provider = providers.provider), quota_remaining_ratio),
          recent_error_rate = COALESCE((SELECT recent_error_rate FROM provider_health_state phs WHERE phs.provider = providers.provider), recent_error_rate, 0),
          rate_limit_risk = COALESCE((SELECT rate_limit_risk FROM provider_health_state phs WHERE phs.provider = providers.provider), rate_limit_risk, 0),
          consecutive_rate_limits = COALESCE((SELECT consecutive_rate_limits FROM provider_health_state phs WHERE phs.provider = providers.provider), consecutive_rate_limits, 0),
          cooldown_until = COALESCE((SELECT rate_limit_cooldown_until FROM provider_health_state phs WHERE phs.provider = providers.provider), cooldown_until),
          latency_ms_p50 = COALESCE((SELECT latency_ms_p50 FROM provider_health_state phs WHERE phs.provider = providers.provider), latency_ms_p50),
          latency_updated_at = COALESCE((SELECT latency_updated_at FROM provider_health_state phs WHERE phs.provider = providers.provider), latency_updated_at),
          last_check_at = COALESCE((SELECT last_check_at FROM provider_health_state phs WHERE phs.provider = providers.provider), last_check_at),
          last_failure_at = COALESCE((SELECT last_failure_at FROM provider_health_state phs WHERE phs.provider = providers.provider), last_failure_at),
          updated_at = ?
        WHERE provider IN (SELECT provider FROM provider_health_state)
        """,
        (now,),
    )
    conn.execute(
        """
        UPDATE providers
        SET health_status = CASE
          WHEN cooldown_until IS NOT NULL AND cooldown_until > datetime('now') THEN 'cooldown'
          WHEN health_score IS NULL THEN 'unknown'
          WHEN health_score >= 0.80 THEN 'healthy'
          WHEN health_score >= 0.40 THEN 'degraded'
          ELSE 'unhealthy'
        END
        """
    )
    conn.execute(
        """
        UPDATE providers
        SET last_success_at = (
              SELECT MAX(observed_at)
              FROM provider_health_log phl
              WHERE phl.provider = providers.provider
                AND COALESCE(phl.auth_status, 'ok') IN ('ok', 'unknown')
                AND COALESCE(phl.quota_state, 'healthy') != 'exhausted'
                AND phl.error_type IS NULL
            ),
            last_error_type = (
              SELECT error_type FROM provider_health_log phl
              WHERE phl.provider = providers.provider AND phl.error_type IS NOT NULL
              ORDER BY observed_at DESC LIMIT 1
            ),
            last_error_message = (
              SELECT note FROM provider_health_log phl
              WHERE phl.provider = providers.provider AND (phl.error_type IS NOT NULL OR phl.http_status >= 400)
              ORDER BY observed_at DESC LIMIT 1
            )
        """
    )


def _known_provider_names(conn: sqlite3.Connection) -> set[str]:
    providers: set[str] = set()
    for table in ("provider_health_state", "provider_health_log", "model_stats", "routing_decisions"):
        try:
            if table == "model_stats":
                rows = conn.execute("SELECT DISTINCT provider FROM model_stats WHERE provider IS NOT NULL AND TRIM(provider) != ''").fetchall()
            elif table == "routing_decisions":
                rows = conn.execute("SELECT DISTINCT selected_provider AS provider FROM routing_decisions WHERE selected_provider IS NOT NULL AND TRIM(selected_provider) != ''").fetchall()
            else:
                rows = conn.execute(f"SELECT DISTINCT provider FROM {table} WHERE provider IS NOT NULL AND TRIM(provider) != ''").fetchall()
            providers.update(str(row["provider"]) for row in rows if row["provider"])
        except sqlite3.OperationalError:
            pass

    try:
        if REGISTRY_FILE.exists():
            payload = json.loads(REGISTRY_FILE.read_text())
            models = payload.get("models", []) if isinstance(payload, dict) else payload
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and model.get("provider"):
                        providers.add(str(model["provider"]))
    except Exception:
        # Registry seeding is best-effort; DB-backed observations above are enough
        # for live routing state.
        pass
    return providers


def _ensure_model_metadata_compat(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metadata (
          model       TEXT PRIMARY KEY,
          is_free     INTEGER,
          updated_at  TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_metadata_updated_at ON model_metadata(updated_at)")
    conn.execute("UPDATE schema_meta SET value='6' WHERE key='schema_version'")
    conn.execute("INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '6')")


def write_decision(
    decision: RoutingDecision,
    classifier: ClassifierOutput,
    pre_signals: PreSignals,
    provider_health: ProviderHealth,
    nexus_workflow_id: Optional[str] = None,
    nexus_step_id: Optional[str] = None,
    nexus_issue_id: Optional[str] = None,
    nexus_project: Optional[str] = None,
    route_mode: Optional[str] = None,
    mode: Optional[str] = None,
    source_type: Optional[str] = None,
    source_tag: Optional[str] = None,
    message_text: Optional[str] = None,
    classifier_source: Optional[str] = None,
) -> str:
    """
    Write a new routing decision to the DB.
    Returns the generated decision ID (UUID).
    """
    decision_id = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO routing_decisions (
              id, created_at,
              task_type, subtype, complexity,
              needs_tools, needs_vision, needs_long_context,
              cost_profile, classifier_confidence, detected_language,
              has_image, has_code, has_diff, has_logs, estimated_tokens,
              selected_model, selected_provider,
              fallbacks, routing_score, reason, excluded_models,
              provider_health_score, quota_state, provider_auth_ok,
              nexus_workflow_id, nexus_step_id, nexus_issue_id, nexus_project,
              route_mode, mode, source_type, source_tag,
              message_text, classifier_source
            ) VALUES (
              ?,?,  ?,?,?,  ?,?,?,  ?,?,?,
              ?,?,?,?,?,  ?,?,  ?,?,?,?,  ?,?,?,  ?,?,?,?,
              ?,?,?,?,  ?,?
            )
            """,
            (
                decision_id, _now_iso(),
                classifier.task_type, classifier.subtype, classifier.complexity,
                int(classifier.needs_tools), int(classifier.needs_vision), int(classifier.needs_long_context),
                classifier.cost_profile, classifier.confidence, classifier.detected_language,
                int(pre_signals.has_image), int(pre_signals.has_code),
                int(pre_signals.has_diff), int(pre_signals.has_logs),
                pre_signals.estimated_tokens,
                decision.selected_model, decision.selected_provider,
                json.dumps(decision.fallbacks), decision.score,
                json.dumps(decision.reason),
                json.dumps(decision.excluded_models),
                provider_health.health_score, provider_health.quota,
                int(provider_health.auth == "ok"),
                nexus_workflow_id, nexus_step_id, nexus_issue_id, nexus_project,
                route_mode,
                mode,
                source_type or 'standalone',
                source_tag,
                # Truncate to 512 chars — enough for classifier, avoids storing full docs
                (message_text or "")[:512] or None,
                classifier_source,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return decision_id


def update_outcome(
    decision_id: str,
    success: bool,
    latency_ms: Optional[int] = None,
    fallback_used: bool = False,
    fallback_model: Optional[str] = None,
    user_override: bool = False,
    user_override_model: Optional[str] = None,
):
    """
    Record the outcome of a routing decision after the turn completes.
    Also updates model_stats for the selected model.
    """
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE routing_decisions
            SET outcome_success=?, latency_ms=?,
                fallback_used=?, fallback_model=?,
                user_override=?, user_override_model=?
            WHERE id=?
            """,
            (
                int(success), latency_ms,
                int(fallback_used), fallback_model,
                int(user_override), user_override_model,
                decision_id,
            ),
        )

        # fetch decision for stats update
        row = conn.execute(
            "SELECT selected_model, selected_provider, task_type, routing_score FROM routing_decisions WHERE id=?",
            (decision_id,),
        ).fetchone()

        if row:
            _update_model_stats(
                conn,
                model=row["selected_model"],
                provider=row["selected_provider"],
                task_type=row["task_type"],
                routing_score=row["routing_score"],
                success=success,
                latency_ms=latency_ms,
                fallback_used=fallback_used,
                user_override=user_override,
            )

        conn.commit()
    finally:
        conn.close()


def _update_model_stats(
    conn: sqlite3.Connection,
    model: str,
    provider: str,
    task_type: Optional[str],
    routing_score: Optional[float],
    success: bool,
    latency_ms: Optional[int],
    fallback_used: bool,
    user_override: bool,
):
    row = conn.execute(
        "SELECT * FROM model_stats WHERE model=?", (model,)
    ).fetchone()

    task_col = {
        "coding": "coding_selected",
        "code_review": "review_selected",
        "reasoning": "reasoning_selected",
        "summarization": "summarize_selected",
        "fast_utility": "fast_selected",
        "long_context": "longctx_selected",
        "vision": "vision_selected",
    }.get(task_type, None)

    now = _now_iso()

    if row is None:
        # First time we've seen this model
        task_val = {task_col: 1} if task_col else {}
        conn.execute(
            f"""
            INSERT INTO model_stats
              (model, provider,
               total_selected, total_success, total_fallback, total_override,
               {', '.join(task_val.keys()) + ',' if task_val else ''}
               avg_latency_ms, avg_routing_score, success_rate,
               first_used_at, last_used_at)
            VALUES
              (?,?,  1,?,?,?,  {'1,' if task_val else ''}  ?,?,?,  ?,?)
            """,
            (
                model, provider,
                int(success), int(fallback_used), int(user_override),
                latency_ms, routing_score, float(success),
                now, now,
            ),
        )
    else:
        total   = row["total_selected"] + 1
        success_count = row["total_success"] + int(success)

        # EWMA for latency and score
        prev_lat = row["avg_latency_ms"]
        new_lat  = (0.3 * latency_ms + 0.7 * prev_lat) if (latency_ms and prev_lat) else (latency_ms or prev_lat)
        prev_sc  = row["avg_routing_score"] or 0.0
        new_sc   = (0.3 * (routing_score or 0.0) + 0.7 * prev_sc) if routing_score else prev_sc

        task_clause = f", {task_col}={task_col}+1" if task_col else ""

        conn.execute(
            f"""
            UPDATE model_stats
            SET total_selected=?,
                total_success=?,
                total_fallback=total_fallback+?,
                total_override=total_override+?,
                avg_latency_ms=?,
                avg_routing_score=?,
                success_rate=?,
                last_used_at=?
                {task_clause}
            WHERE model=?
            """,
            (
                total, success_count,
                int(fallback_used), int(user_override),
                new_lat, new_sc,
                success_count / total,
                now,
                model,
            ),
        )


def load_outcome_counts() -> dict[str, int]:
    """
    Return a mapping of model_id -> total_selected (number of routing outcomes recorded).
    Used by generate_registry to decide whether a model has "graduated" from cold-start
    family priors to fully data-driven scoring.
    """
    if not DB_PATH.exists():
        return {}
    conn = _connect()
    try:
        try:
            rows = conn.execute(
                "SELECT model, total_selected FROM model_stats WHERE total_selected IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {row["model"]: int(row["total_selected"] or 0) for row in rows}
    finally:
        conn.close()


def load_model_stats() -> dict[str, dict]:
    """
    Load learned model stats from DB.
    Returns dict of model_id → stats dict.
    Used by scorer to apply learned adjustments.
    """
    if not DB_PATH.exists():
        return {}
    conn = _connect()
    try:
        try:
            rows = conn.execute("SELECT * FROM model_stats").fetchall()
        except sqlite3.OperationalError:
            return {}
        stats = {row["model"]: dict(row) for row in rows}
        for model, feedback in _load_feedback_preferences(conn).items():
            stats.setdefault(model, {"model": model})["feedback_preference"] = feedback
        return stats
    finally:
        conn.close()


def load_model_metadata() -> dict[str, dict[str, Any]]:
    if not DB_PATH.exists():
        return {}
    conn = _connect()
    try:
        try:
            rows = conn.execute("SELECT model, is_free, updated_at FROM model_metadata").fetchall()
        except sqlite3.OperationalError:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            raw = row["is_free"]
            result[row["model"]] = {
                "is_free": None if raw is None else bool(raw),
                "updated_at": row["updated_at"],
            }
        return result
    finally:
        conn.close()


def set_model_is_free(model: str, is_free: Optional[bool]) -> None:
    model = str(model or "").strip()
    if not model:
        raise ValueError("model required")
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO model_metadata (model, is_free, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(model)
            DO UPDATE SET is_free=excluded.is_free, updated_at=excluded.updated_at
            """,
            (model, None if is_free is None else int(bool(is_free)), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def log_provider_observation(
    provider: str,
    auth_status: str,
    quota_state: str = "unknown",
    http_status: Optional[int] = None,
    error_type: Optional[str] = None,
    latency_ms: Optional[int] = None,
    note: Optional[str] = None,
):
    """Append one health observation to provider_health_log."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO provider_health_log
              (id, observed_at, provider, auth_status, quota_state,
               http_status, error_type, latency_ms, note)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()), _now_iso(),
                provider, auth_status, quota_state,
                http_status, error_type, latency_ms, note,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _load_feedback_preferences(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        rows = conn.execute(
            """
            SELECT
              COALESCE(NULLIF(rf.preferred_model, ''), rd.selected_model) AS model,
              COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type, '*') AS task_type,
              COUNT(*) AS sample_count,
              AVG(
                CASE rf.model_verdict
                  WHEN 'good' THEN 1.0
                  WHEN 'neutral' THEN 0.4
                  WHEN 'bad' THEN 0.0
                  WHEN 'too_cheap' THEN 0.0
                  WHEN 'too_powerful' THEN 0.0
                  ELSE CASE rf.verdict WHEN 'correct' THEN 0.75 ELSE 0.0 END
                END
              ) AS preference_score,
              MAX(rf.created_at) AS last_feedback_at,
              MAX(COALESCE(rf.reason_tag, '')) AS top_reason_tag
            FROM route_feedback rf
            LEFT JOIN routing_decisions rd ON rd.id = rf.decision_id
            WHERE rf.preferred_model IS NOT NULL
               OR rd.selected_model IS NOT NULL
               OR rf.model_verdict IS NOT NULL
            GROUP BY COALESCE(NULLIF(rf.preferred_model, ''), rd.selected_model),
                     COALESCE(NULLIF(rf.corrected_task, ''), rd.task_type, '*')
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], {})[row["task_type"]] = {
            "samples": int(row["sample_count"] or 0),
            "score": round(float(row["preference_score"] or 0.0), 4),
            "last_feedback_at": row["last_feedback_at"],
            "top_reason_tag": row["top_reason_tag"] or None,
        }
    return grouped


def set_route_mode_preference(pref_key: str, mode: str, scope: str = "conversation", free_filter: bool = False) -> None:
    pref_key = str(pref_key or "").strip()
    mode = str(mode or "").strip().lower()
    scope = str(scope or "conversation").strip().lower()
    if not pref_key:
        raise ValueError("pref_key required")
    if mode not in {"auto", "balanced", "fast", "reasoning", "eco", "free", "off"}:
        raise ValueError("mode must be auto|balanced|fast|reasoning|eco|free|off")
    if scope not in {"conversation", "session", "channel"}:
        raise ValueError("scope must be conversation|session|channel")

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO route_mode_preferences (pref_key, scope, mode, free_filter, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scope, pref_key)
            DO UPDATE SET mode=excluded.mode, free_filter=excluded.free_filter, updated_at=excluded.updated_at
            """,
            (pref_key, scope, mode, int(bool(free_filter)), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def get_route_mode_preference(pref_key: str, scope: str = "conversation") -> Optional[dict[str, Any]]:
    pref_key = str(pref_key or "").strip()
    scope = str(scope or "conversation").strip().lower()
    if not pref_key:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT pref_key, scope, mode, free_filter, updated_at FROM route_mode_preferences WHERE pref_key=? AND scope=?",
            (pref_key, scope),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["free_filter"] = bool(payload.get("free_filter"))
        return payload
    finally:
        conn.close()


def load_benchmark_scores() -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        try:
            rows = conn.execute("SELECT * FROM benchmark_model_scores").fetchall()
        except sqlite3.OperationalError:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = dict(row)
            model_id = str(payload.pop("model_id"))
            source = payload.pop("source", None)
            updated_at = payload.pop("updated_at", None)
            metadata_json = payload.pop("metadata_json", None)
            entry = {k: v for k, v in payload.items() if v is not None}
            if source is not None:
                entry["_source"] = source
            if updated_at is not None:
                entry["_updated_at"] = updated_at
            if metadata_json:
                try:
                    entry["_metadata"] = json.loads(metadata_json)
                except Exception:
                    entry["_metadata"] = metadata_json
            result[model_id] = entry
        return result
    finally:
        conn.close()


def replace_benchmark_scores(models: dict[str, dict[str, Any]]) -> None:
    conn = _connect()
    try:
        # Schema migration for eco
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(benchmark_model_scores)").fetchall()}
        if columns and "eco" not in columns:
            conn.execute("ALTER TABLE benchmark_model_scores ADD COLUMN eco REAL")
            
        conn.execute("DELETE FROM benchmark_model_scores")
        for model_id, scores in models.items():
            conn.execute(
                """
                INSERT INTO benchmark_model_scores (
                  model_id, source, updated_at,
                  coding, review, reasoning, summarize, fast,
                  cost, speed, eco, context, vision, tools, multilingual,
                  metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    scores.get("_source"),
                    scores.get("_updated_at"),
                    scores.get("coding"),
                    scores.get("review"),
                    scores.get("reasoning"),
                    scores.get("summarize"),
                    scores.get("fast"),
                    scores.get("cost"),
                    scores.get("speed"),
                    scores.get("eco"),
                    scores.get("context"),
                    scores.get("vision"),
                    scores.get("tools"),
                    scores.get("multilingual"),
                    json.dumps(scores.get("_metadata")) if scores.get("_metadata") is not None else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_providers(providers: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        try:
            _ensure_providers_compat(conn)
            if providers:
                placeholders = ",".join("?" for _ in providers)
                rows = conn.execute(
                    f"SELECT * FROM providers WHERE provider IN ({placeholders})",
                    tuple(providers),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM providers").fetchall()
        except sqlite3.OperationalError:
            return {}
        return {row["provider"]: dict(row) for row in rows}
    finally:
        conn.close()


def upsert_provider_health_state(provider: str, state: dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO provider_health_state (
              provider, auth, quota, quota_remaining_ratio,
              recent_error_rate, rate_limit_risk, consecutive_rate_limits,
              rate_limit_cooldown_until, latency_ms_p50, latency_updated_at, last_failure_at,
              last_check_at, health_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider)
            DO UPDATE SET
              auth=excluded.auth,
              quota=excluded.quota,
              quota_remaining_ratio=excluded.quota_remaining_ratio,
              recent_error_rate=excluded.recent_error_rate,
              rate_limit_risk=excluded.rate_limit_risk,
              consecutive_rate_limits=excluded.consecutive_rate_limits,
              rate_limit_cooldown_until=excluded.rate_limit_cooldown_until,
              latency_ms_p50=excluded.latency_ms_p50,
              latency_updated_at=excluded.latency_updated_at,
              last_failure_at=excluded.last_failure_at,
              last_check_at=excluded.last_check_at,
              health_score=excluded.health_score
            """,
            (
                provider,
                state.get("auth", "unknown"),
                state.get("quota", "unknown"),
                state.get("quota_remaining_ratio"),
                state.get("recent_error_rate", 0.0),
                state.get("rate_limit_risk", 0.0),
                state.get("consecutive_rate_limits", 0),
                state.get("rate_limit_cooldown_until"),
                state.get("latency_ms_p50"),
                state.get("latency_updated_at"),
                state.get("last_failure_at"),
                state.get("last_check_at"),
                state.get("health_score", 1.0),
            ),
        )
        # Keep the provider control/current-state table in sync with health observations.
        _ensure_providers_compat(conn)
        now = _now_iso()
        enabled = 1 if provider in {"anthropic", "openai-codex", "google-gemini-cli", "openrouter", "nvidia"} else 0
        status = "enabled" if enabled else "disabled"
        conn.execute(
            """
            INSERT INTO providers (provider, enabled, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO NOTHING
            """,
            (provider, enabled, status, now, now),
        )
        conn.execute(
            """
            UPDATE providers
            SET
              health_score=?, auth_status=?, quota_status=?, quota_remaining_ratio=?,
              recent_error_rate=?, rate_limit_risk=?, consecutive_rate_limits=?,
              cooldown_until=?, latency_ms_p50=?, latency_updated_at=?,
              last_failure_at=?, last_check_at=?, updated_at=?,
              health_status = CASE
                WHEN ? IS NOT NULL AND ? > datetime('now') THEN 'cooldown'
                WHEN ? IS NULL THEN 'unknown'
                WHEN ? >= 0.80 THEN 'healthy'
                WHEN ? >= 0.40 THEN 'degraded'
                ELSE 'unhealthy'
              END
            WHERE provider=?
            """,
            (
                state.get("health_score", 1.0), state.get("auth", "unknown"), state.get("quota", "unknown"),
                state.get("quota_remaining_ratio"), state.get("recent_error_rate", 0.0), state.get("rate_limit_risk", 0.0),
                state.get("consecutive_rate_limits", 0), state.get("rate_limit_cooldown_until"), state.get("latency_ms_p50"),
                state.get("latency_updated_at"), state.get("last_failure_at"), state.get("last_check_at"), now,
                state.get("rate_limit_cooldown_until"), state.get("rate_limit_cooldown_until"),
                state.get("health_score"), state.get("health_score", 1.0), state.get("health_score", 1.0), provider,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_provider_health_state_compat(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(provider_health_state)").fetchall()}
    if not columns:
        return
    expected = {
        "latency_updated_at": "TEXT",
    }
    for column, column_type in expected.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE provider_health_state ADD COLUMN {column} {column_type}")


def load_provider_health_state(providers: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        _ensure_provider_health_state_compat(conn)
        try:
            if providers:
                placeholders = ",".join("?" for _ in providers)
                rows = conn.execute(
                    f"SELECT * FROM provider_health_state WHERE provider IN ({placeholders})",
                    tuple(providers),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM provider_health_state").fetchall()
        except sqlite3.OperationalError:
            return {}
        return {row["provider"]: dict(row) for row in rows}
    finally:
        conn.close()


def record_feedback(
    decision_id: str,
    verdict: str,
    corrected_task: Optional[str] = None,
    model_verdict: Optional[str] = None,
    preferred_model: Optional[str] = None,
    reason_tag: Optional[str] = None,
    source_surface: Optional[str] = None,
    source_channel: Optional[str] = None,
    source_message_id: Optional[str] = None,
    source_user_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    feedback_id = str(uuid.uuid4())
    clean_decision_id = str(decision_id or "").strip()
    if not clean_decision_id:
        raise ValueError("decision_id required")

    conn = _connect()
    try:
        linked = conn.execute(
            "SELECT 1 FROM routing_decisions WHERE id=? LIMIT 1",
            (clean_decision_id,),
        ).fetchone()
        if not linked:
            raise ValueError("decision_id_not_found")

        conn.execute(
            """
            INSERT INTO route_feedback (
              id, created_at, decision_id, verdict, corrected_task,
              model_verdict, preferred_model, reason_tag,
              source_surface, source_channel, source_message_id, source_user_id, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                feedback_id,
                _now_iso(),
                clean_decision_id,
                verdict,
                corrected_task,
                model_verdict,
                preferred_model,
                reason_tag,
                source_surface,
                source_channel,
                source_message_id,
                source_user_id,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return feedback_id
