-- Nexus Router — SQLite schema
-- Version: 6
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
  selected_cost_score   REAL,                  -- normalized model cost score at decision time
  reason                TEXT,                  -- JSON array of reason strings
  excluded_models       TEXT,                  -- JSON array of {model, reason}
  original_task_type    TEXT,                  -- classifier task before confidence/mode overrides
  mechanisms            TEXT,                  -- JSON array: static_weights|fitted_weights|confidence_gate|bandit_*
  confidence_gate_triggered INTEGER DEFAULT 0,
  confidence_gate_threshold REAL,
  confidence_gate_reason TEXT,
  selected_component_scores TEXT,              -- JSON object raw component values
  selected_component_contributions TEXT,       -- JSON object weighted contributions

  -- provider state at decision time
  provider_health_score REAL,                  -- health score used in routing
  quota_state           TEXT,                  -- healthy|low|unknown|exhausted
  provider_auth_ok      INTEGER,               -- 0/1

  -- outcome (filled in after turn completes)
  outcome_success       INTEGER,               -- 0/1; NULL = not yet recorded
  fallback_used         INTEGER,               -- 0/1
  fallback_model        TEXT,                  -- model actually used if fallback triggered
  latency_ms            INTEGER,               -- wall-clock time for the turn
  input_tokens          INTEGER,               -- prompt tokens (if runtime reports usage)
  output_tokens         INTEGER,               -- completion tokens
  total_tokens          INTEGER,               -- total runtime tokens
  estimated_cost_usd    REAL,                  -- estimated cost for the turn
  estimated_co2e_grams  REAL,                  -- reserved for eco/CO2 pipeline (nullable)
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
  free_filter           INTEGER NOT NULL DEFAULT 0,
  updated_at            TEXT NOT NULL,
  PRIMARY KEY (scope, pref_key)
);

CREATE TABLE IF NOT EXISTS model_metadata (
  model       TEXT PRIMARY KEY,
  is_free     INTEGER,
  updated_at  TEXT NOT NULL
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
  latency_updated_at         TEXT,
  last_failure_at            TEXT,
  last_check_at              TEXT,
  health_score               REAL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS providers (
  provider                   TEXT PRIMARY KEY,

  -- operator controls
  enabled                    INTEGER NOT NULL DEFAULT 0,
  status                     TEXT NOT NULL DEFAULT 'unknown', -- enabled|disabled|maintenance|unknown
  priority                   INTEGER DEFAULT 100,
  allow_primary              INTEGER NOT NULL DEFAULT 1,
  allow_fallback             INTEGER NOT NULL DEFAULT 1,
  weight_multiplier          REAL NOT NULL DEFAULT 1.0,
  max_concurrency            INTEGER,
  notes                      TEXT,

  -- current runtime state, mirrored from provider_health_state when available
  health_status              TEXT NOT NULL DEFAULT 'unknown', -- healthy|degraded|unhealthy|cooldown|unknown
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
  eco                    REAL,
  context                REAL,
  vision                 REAL,
  tools                  REAL,
  multilingual           REAL,
  metadata_json          TEXT
);

-- Per-model circuit breaker. scope = task type (advisory) or CLI category.
CREATE TABLE IF NOT EXISTS model_cooldowns (
  model                 TEXT NOT NULL,
  scope                 TEXT NOT NULL,
  failures              INTEGER NOT NULL DEFAULT 0,
  window_started_at     TEXT NOT NULL,
  cooldown_until        TEXT,
  PRIMARY KEY (model, scope)
);

-- CLI wire proxy state. Kept apart from routing_decisions so learned stats
-- and classifier training never see wire-proxy traffic. No prompt text.
CREATE TABLE IF NOT EXISTS cli_proxy_pins (
  session               TEXT PRIMARY KEY,   -- hashed session key
  provider              TEXT NOT NULL,
  model                 TEXT NOT NULL,
  category              TEXT NOT NULL,
  touched_at            REAL NOT NULL       -- unix seconds
);

CREATE TABLE IF NOT EXISTS cli_proxy_turns (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at            REAL NOT NULL,      -- unix seconds
  session               TEXT NOT NULL,
  category              TEXT NOT NULL,
  reason                TEXT,
  target                TEXT,               -- provider:model, NULL when nothing answered
  cache_pinned          INTEGER NOT NULL DEFAULT 0,
  failures_json         TEXT,
  http_status           INTEGER,
  latency_ms            INTEGER,
  input_tokens          INTEGER,
  output_tokens         INTEGER,
  converted             INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cli_turns_created_at ON cli_proxy_turns(created_at);
CREATE INDEX IF NOT EXISTS idx_cli_turns_target ON cli_proxy_turns(target, created_at);

-- Enterprise gateway state. Identities are hashes; prompts and credentials are never stored.
CREATE TABLE IF NOT EXISTS gateway_task_pins (
  session TEXT PRIMARY KEY,
  principal TEXT NOT NULL,
  target TEXT NOT NULL,
  profile TEXT NOT NULL,
  requested_model TEXT NOT NULL,
  choices_json TEXT NOT NULL,
  touched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS gateway_turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL NOT NULL,
  principal TEXT NOT NULL,
  session TEXT NOT NULL,
  profile TEXT NOT NULL,
  target TEXT,
  effort TEXT,
  http_status INTEGER,
  latency_ms INTEGER,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_5m_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
  usage_complete INTEGER NOT NULL DEFAULT 0,
  estimated_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_gateway_turns_principal ON gateway_turns(principal, created_at);
CREATE INDEX IF NOT EXISTS idx_gateway_turns_session ON gateway_turns(session, created_at);

CREATE INDEX IF NOT EXISTS idx_route_feedback_decision_id ON route_feedback(decision_id);
CREATE INDEX IF NOT EXISTS idx_route_feedback_created_at ON route_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_route_feedback_preferred_model ON route_feedback(preferred_model);
CREATE INDEX IF NOT EXISTS idx_route_mode_preferences_updated_at ON route_mode_preferences(updated_at);
CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled);
CREATE INDEX IF NOT EXISTS idx_providers_health_status ON providers(health_status);
CREATE INDEX IF NOT EXISTS idx_providers_updated_at ON providers(updated_at);
CREATE INDEX IF NOT EXISTS idx_model_metadata_updated_at ON model_metadata(updated_at);

INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '6');
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

CREATE TABLE IF NOT EXISTS gateway_task_explanations (
 session TEXT PRIMARY KEY, explanation_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gateway_turn_explanations (
 turn_id INTEGER PRIMARY KEY, explanation_json TEXT NOT NULL,
 FOREIGN KEY(turn_id) REFERENCES gateway_turns(id)
);

CREATE TABLE IF NOT EXISTS gateway_budget_reservations (
 request_id TEXT PRIMARY KEY, principal TEXT NOT NULL, session TEXT NOT NULL,
 created_at REAL NOT NULL, reserved_microusd INTEGER NOT NULL, held_microusd INTEGER NOT NULL,
 state TEXT NOT NULL, attempts INTEGER NOT NULL, owner_pid INTEGER NOT NULL,
 finished_at REAL, metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_gateway_budget_principal ON gateway_budget_reservations(principal,created_at);
CREATE TABLE IF NOT EXISTS gateway_budget_turns (
 turn_id INTEGER PRIMARY KEY, request_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gateway_budget_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, created_at REAL NOT NULL,
 previous_microusd INTEGER NOT NULL, actual_microusd INTEGER NOT NULL, note TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gateway_cache_hints (
 cache_key TEXT PRIMARY KEY, cached_tokens INTEGER NOT NULL, observed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS gateway_escalation_tickets (
 ticket TEXT PRIMARY KEY, principal TEXT NOT NULL, root TEXT NOT NULL, level INTEGER NOT NULL,
 source_session TEXT NOT NULL, new_session_id TEXT NOT NULL, child_session TEXT,
 profile TEXT NOT NULL, wire TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL,
 UNIQUE(principal,root,level)
);
CREATE TABLE IF NOT EXISTS gateway_feedback (
 id INTEGER PRIMARY KEY AUTOINCREMENT, principal TEXT NOT NULL, session TEXT NOT NULL,
 rating TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gateway_feedback_principal ON gateway_feedback(principal,created_at);
