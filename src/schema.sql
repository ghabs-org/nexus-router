-- Nexus Router — SQLite schema
-- Version: 4
-- Location: data/router.sqlite

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ────────────────────────────────────────────────────────────────────────────
-- routing_decisions
-- One row per routing decision made by the router.
-- Outcome columns are updated after the turn completes.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS routing_decisions (
  -- identity
  id                    TEXT    PRIMARY KEY,   -- UUID v4
  created_at            TEXT    NOT NULL,      -- ISO-8601 UTC

  -- classifier output
  task_type             TEXT,                  -- coding|code_review|reasoning|summarization|fast_utility|long_context|vision|general_chat
  subtype               TEXT,                  -- e.g. implementation|debugging|refactor
  complexity            TEXT,                  -- low|medium|high
  needs_tools           INTEGER,               -- 0/1
  needs_vision          INTEGER,               -- 0/1
  needs_long_context    INTEGER,               -- 0/1
  cost_profile          TEXT,                  -- cheap|balanced|premium
  classifier_confidence REAL,                  -- 0.0–1.0
  detected_language     TEXT,                  -- e.g. en, it, mixed

  -- pre-signals (non-linguistic features extracted before classifier)
  has_image             INTEGER,               -- 0/1
  has_code              INTEGER,               -- 0/1
  has_diff              INTEGER,               -- 0/1
  has_logs              INTEGER,               -- 0/1
  estimated_tokens      INTEGER,               -- rough input size estimate

  -- routing result
  selected_model        TEXT    NOT NULL,      -- e.g. openai-codex/gpt-5.4
  selected_provider     TEXT,                  -- e.g. openai-codex
  fallbacks             TEXT,                  -- JSON array of model ids
  routing_score         REAL,                  -- final composite score 0.0–1.0
  reason                TEXT,                  -- JSON array of reason strings
  excluded_models       TEXT,                  -- JSON array of {model, reason}

  -- provider state at decision time
  provider_health_score REAL,                  -- health score used in routing
  quota_state           TEXT,                  -- healthy|low|unknown|exhausted
  provider_auth_ok      INTEGER,               -- 0/1

  -- outcome (filled in after turn completes)
  outcome_success       INTEGER,               -- 0/1; NULL = not yet recorded
  fallback_used         INTEGER,               -- 0/1
  fallback_model        TEXT,                  -- model actually used if fallback triggered
  latency_ms            INTEGER,               -- wall-clock time for the turn
  user_override         INTEGER,               -- 0/1; user manually switched model
  user_override_model   TEXT,                  -- model user switched to

  -- optional Nexus linkage (nullable; populated when called from Nexus context)
  nexus_workflow_id     TEXT,
  nexus_step_id         TEXT,
  nexus_issue_id        TEXT,
  nexus_project         TEXT,

  -- routing strategy + provenance / hygiene markers
  route_mode            TEXT,
  mode                  TEXT,                  -- backfilled from provenance_mode on migration; 'route' | 'shadow'
  source_type           TEXT,
  source_tag            TEXT,

  -- classifier training data
  message_text          TEXT,                  -- user message, truncated to 512 chars
  classifier_source     TEXT                   -- llm | heuristic | local | explicit | fallback
);

-- ────────────────────────────────────────────────────────────────────────────
-- provider_health_log
-- Time-series of observed provider health.
-- Used to compute rolling health scores and detect degradation trends.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS provider_health_log (
  id                TEXT    PRIMARY KEY,  -- UUID v4
  observed_at       TEXT    NOT NULL,     -- ISO-8601 UTC
  provider          TEXT    NOT NULL,     -- e.g. openai-codex
  auth_status       TEXT,                 -- ok|expired|missing|unknown
  quota_state       TEXT,                 -- healthy|low|exhausted|unknown
  http_status       INTEGER,              -- last HTTP response code if available
  error_type        TEXT,                 -- rate_limit|auth|server|timeout|null
  latency_ms        INTEGER,              -- probe latency if available
  note              TEXT                  -- freeform observation
);

-- ────────────────────────────────────────────────────────────────────────────
-- model_stats
-- Aggregated per-model performance. Updated after each turn.
-- Lightweight learned layer — not a full ML training set.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_stats (
  model             TEXT    PRIMARY KEY,  -- e.g. openai-codex/gpt-5.4
  provider          TEXT    NOT NULL,
  -- counts
  total_selected    INTEGER NOT NULL DEFAULT 0,
  total_success     INTEGER NOT NULL DEFAULT 0,
  total_fallback    INTEGER NOT NULL DEFAULT 0,
  total_override    INTEGER NOT NULL DEFAULT 0,  -- user manually overrode
  -- per task type counts (updated incrementally)
  coding_selected   INTEGER NOT NULL DEFAULT 0,
  review_selected   INTEGER NOT NULL DEFAULT 0,
  reasoning_selected INTEGER NOT NULL DEFAULT 0,
  summarize_selected INTEGER NOT NULL DEFAULT 0,
  fast_selected     INTEGER NOT NULL DEFAULT 0,
  longctx_selected  INTEGER NOT NULL DEFAULT 0,
  vision_selected   INTEGER NOT NULL DEFAULT 0,
  -- performance
  avg_latency_ms    REAL,
  avg_routing_score REAL,
  success_rate      REAL,                 -- total_success / total_selected
  -- timestamps
  first_used_at     TEXT,                 -- ISO-8601 UTC
  last_used_at      TEXT                  -- ISO-8601 UTC
);

-- ────────────────────────────────────────────────────────────────────────────
-- schema_meta
-- Tracks schema version for future migrations.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS route_feedback (
  id                    TEXT PRIMARY KEY,
  created_at            TEXT NOT NULL,
  decision_id           TEXT NOT NULL,
  verdict               TEXT NOT NULL,
  corrected_task        TEXT,
  model_verdict         TEXT,
  preferred_model       TEXT,
  reason_tag            TEXT,
  source_surface        TEXT,
  source_channel        TEXT,
  source_message_id     TEXT,
  source_user_id        TEXT,
  metadata              TEXT,
  FOREIGN KEY(decision_id) REFERENCES routing_decisions(id)
);

CREATE TABLE IF NOT EXISTS route_mode_preferences (
  pref_key              TEXT NOT NULL,
  scope                 TEXT NOT NULL DEFAULT 'conversation',
  mode                  TEXT NOT NULL,
  updated_at            TEXT NOT NULL,
  PRIMARY KEY (scope, pref_key)
);

CREATE TABLE IF NOT EXISTS provider_health_state (
  provider                   TEXT PRIMARY KEY,
  auth                       TEXT,
  quota                      TEXT,
  quota_remaining_ratio      REAL,
  recent_error_rate          REAL DEFAULT 0,
  rate_limit_risk            REAL DEFAULT 0,
  consecutive_rate_limits    INTEGER DEFAULT 0,
  rate_limit_cooldown_until  TEXT,
  latency_ms_p50             REAL,
  last_failure_at            TEXT,
  last_check_at              TEXT,
  health_score               REAL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS benchmark_model_scores (
  model_id               TEXT PRIMARY KEY,
  source                 TEXT,
  updated_at             TEXT,
  coding                 REAL,
  review                 REAL,
  reasoning              REAL,
  summarize              REAL,
  fast                   REAL,
  cost                   REAL,
  speed                  REAL,
  context                REAL,
  vision                 REAL,
  tools                  REAL,
  multilingual           REAL,
  metadata_json          TEXT
);

CREATE INDEX IF NOT EXISTS idx_route_feedback_decision_id ON route_feedback(decision_id);
CREATE INDEX IF NOT EXISTS idx_route_feedback_created_at ON route_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_route_feedback_preferred_model ON route_feedback(preferred_model);
CREATE INDEX IF NOT EXISTS idx_route_mode_preferences_updated_at ON route_mode_preferences(updated_at);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '4');
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('created_at', datetime('now'));

-- ────────────────────────────────────────────────────────────────────────────
-- Indexes
-- ────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_rd_task_type       ON routing_decisions(task_type);
CREATE INDEX IF NOT EXISTS idx_rd_model           ON routing_decisions(selected_model);
CREATE INDEX IF NOT EXISTS idx_rd_provider        ON routing_decisions(selected_provider);
CREATE INDEX IF NOT EXISTS idx_rd_created_at      ON routing_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_rd_nexus_issue     ON routing_decisions(nexus_issue_id);
CREATE INDEX IF NOT EXISTS idx_rd_nexus_workflow  ON routing_decisions(nexus_workflow_id);
CREATE INDEX IF NOT EXISTS idx_rd_outcome         ON routing_decisions(outcome_success);

CREATE INDEX IF NOT EXISTS idx_phl_provider       ON provider_health_log(provider);
CREATE INDEX IF NOT EXISTS idx_phl_observed_at    ON provider_health_log(observed_at);

-- ────────────────────────────────────────────────────────────────────────────
-- Useful views
-- ────────────────────────────────────────────────────────────────────────────

-- Recent routing decisions with outcome (last 500)
CREATE VIEW IF NOT EXISTS v_recent_decisions AS
SELECT
  id, created_at, task_type, complexity,
  selected_model, routing_score,
  outcome_success, fallback_used, latency_ms, user_override,
  nexus_issue_id, nexus_workflow_id
FROM routing_decisions
ORDER BY created_at DESC
LIMIT 500;

-- Per-model success rates
CREATE VIEW IF NOT EXISTS v_model_success_rates AS
SELECT
  selected_model,
  selected_provider,
  COUNT(*) AS total,
  SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) AS success,
  ROUND(
    100.0 * SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) / COUNT(*), 1
  ) AS success_pct,
  AVG(latency_ms) AS avg_latency_ms,
  SUM(CASE WHEN user_override = 1 THEN 1 ELSE 0 END) AS user_overrides
FROM routing_decisions
WHERE outcome_success IS NOT NULL
GROUP BY selected_model, selected_provider
ORDER BY success_pct DESC;

-- Task type distribution
CREATE VIEW IF NOT EXISTS v_task_distribution AS
SELECT
  task_type,
  COUNT(*) AS total,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM routing_decisions), 1) AS pct,
  AVG(classifier_confidence) AS avg_confidence,
  AVG(routing_score) AS avg_routing_score
FROM routing_decisions
GROUP BY task_type
ORDER BY total DESC;

-- Provider health recent trend (last 48h)
CREATE VIEW IF NOT EXISTS v_provider_health_trend AS
SELECT
  provider,
  COUNT(*) AS observations,
  SUM(CASE WHEN auth_status = 'ok' THEN 1 ELSE 0 END) AS auth_ok_count,
  SUM(CASE WHEN error_type IS NOT NULL THEN 1 ELSE 0 END) AS error_count,
  AVG(latency_ms) AS avg_latency_ms,
  MAX(observed_at) AS last_observed
FROM provider_health_log
WHERE observed_at >= datetime('now', '-48 hours')
GROUP BY provider
ORDER BY provider;
