"""Durable task profiles, usage accounting and admission limits for the gateway."""

import json
import threading
import time
from contextlib import closing

from . import db
from .gateway_policy import PolicyError


class GatewayState:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.active = {}

    def pin(self, session):
        with closing(db._connect()) as conn, conn:
            cutoff = time.time() - self.config.get("pin_ttl_seconds", 86400)
            conn.execute("DELETE FROM gateway_task_pins WHERE touched_at < ?", (cutoff,))
            conn.execute("DELETE FROM gateway_task_explanations WHERE session NOT IN (SELECT session FROM gateway_task_pins)")
            row = conn.execute("SELECT * FROM gateway_task_pins WHERE session=?", (session,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["choices"] = json.loads(result.pop("choices_json"))
            explanation = conn.execute("SELECT explanation_json FROM gateway_task_explanations WHERE session=?", (session,)).fetchone()
            result["explanation"] = json.loads(explanation[0]) if explanation else {}
            return result

    def save_pin(self, classification, target):
        with closing(db._connect()) as conn, conn:
            conn.execute("INSERT INTO gateway_task_pins VALUES (?,?,?,?,?,?,?) ON CONFLICT(session) DO UPDATE SET "
                         "target=excluded.target, profile=excluded.profile, choices_json=excluded.choices_json, touched_at=excluded.touched_at",
                         (classification.session, classification.principal, target.label, classification.profile,
                          classification.requested_model, json.dumps(classification.choices), time.time()))
            conn.execute("INSERT INTO gateway_task_explanations VALUES (?,?) ON CONFLICT(session) DO UPDATE SET explanation_json=excluded.explanation_json",
                         (classification.session, json.dumps(classification.explanation)))
            maximum = self.config.get("max_pins", 4096)
            conn.execute("DELETE FROM gateway_task_pins WHERE session IN (SELECT session FROM gateway_task_pins "
                         "ORDER BY touched_at DESC LIMIT -1 OFFSET ?)", (maximum,))

    def admit(self, principal, session, limits):
        # One gateway process owns this SQLite database. Multiple replicas need
        # a shared transactional admission store, not separate SQLite files.
        with self.lock:
            with closing(db._connect()) as conn:
                window = conn.execute("SELECT COALESCE(SUM(input_tokens+output_tokens),0), "
                                      "COALESCE(SUM(CASE WHEN usage_complete=0 AND target IS NOT NULL "
                                      "AND (http_status < 300 OR http_status >= 499) THEN 1 ELSE 0 END),0) "
                                      "FROM gateway_turns WHERE principal=? AND created_at>=?",
                                      (principal, time.time() - 86400)).fetchone()
                task = conn.execute("SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM gateway_turns "
                                    "WHERE principal=? AND session=?", (principal, session)).fetchone()[0]
            if (limits.get("max_tokens_24h") or limits.get("max_task_tokens")) and window[1]:
                raise PolicyError("Usage is incomplete for an earlier request; ask the gateway administrator to reconcile it", 402)
            if limits.get("max_tokens_24h") and window[0] >= limits["max_tokens_24h"]:
                raise PolicyError("Your 24-hour token allowance is exhausted; contact your administrator", 402)
            if limits.get("max_task_tokens") and task >= limits["max_task_tokens"]:
                raise PolicyError("This task's token allowance is exhausted; contact your administrator", 402)
            if self.active.get(principal, 0) >= limits.get("max_concurrent", 4):
                raise PolicyError("Too many concurrent gateway requests", 429)
            self.active[principal] = self.active.get(principal, 0) + 1

    def release(self, principal):
        with self.lock:
            self.active[principal] = max(0, self.active.get(principal, 1) - 1)

    def record(self, classification, target, status, latency_ms, usage):
        usage = usage or {}
        choice = next((c for c in classification.choices if target and c["target"] == target.label), {})
        prices = self.config["models"].get(target.label, {}).get("pricing") if target else None
        complete = (status is not None and 200 <= status < 300 and "input_tokens" in usage
                    and "output_tokens" in usage and usage.get("stream_complete", True))
        cost = None
        if prices and complete:
            writes = usage.get("cache_creation_input_tokens", 0)
            five, hour = usage.get("ephemeral_5m_input_tokens", 0), usage.get("ephemeral_1h_input_tokens", 0)
            # Without the provider's TTL split, do not invent an exact cache-write tariff.
            if five + hour == writes:
                cost = sum(usage.get(field, 0) * prices[key] for field, key in (
                    ("uncached_input_tokens", "input"), ("output_tokens", "output"),
                    ("cache_read_input_tokens", "cache_read"), ("ephemeral_5m_input_tokens", "cache_write_5m"),
                    ("ephemeral_1h_input_tokens", "cache_write_1h"))) / 1_000_000
        values = (time.time(), classification.principal, classification.session, classification.profile,
                  target.label if target else None, choice.get("effort"), status, latency_ms,
                  usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("uncached_input_tokens", 0),
                  usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0),
                  usage.get("ephemeral_5m_input_tokens", 0), usage.get("ephemeral_1h_input_tokens", 0), int(complete), cost)
        with closing(db._connect()) as conn, conn:
            cursor = conn.execute("INSERT INTO gateway_turns (created_at,principal,session,profile,target,effort,http_status,latency_ms,"
                         "input_tokens,output_tokens,uncached_input_tokens,cache_read_input_tokens,cache_creation_input_tokens,"
                         "cache_write_5m_tokens,cache_write_1h_tokens,usage_complete,estimated_cost_usd) VALUES "
                         "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            conn.execute("INSERT INTO gateway_turn_explanations VALUES (?,?)", (cursor.lastrowid, json.dumps(classification.explanation)))

        from .gateway_budget import BudgetLedger
        BudgetLedger().finish(classification.request_id, cursor.lastrowid, cost, complete)

    def status(self, principal):
        with closing(db._connect()) as conn:
            summary = dict(conn.execute("SELECT COUNT(*) AS turns, COALESCE(SUM(input_tokens),0) AS input_tokens, "
                                       "COALESCE(SUM(output_tokens),0) AS output_tokens, "
                                       "COALESCE(SUM(cache_read_input_tokens),0) AS cache_read_input_tokens, "
                                       "COALESCE(SUM(cache_creation_input_tokens),0) AS cache_creation_input_tokens, "
                                       "SUM(estimated_cost_usd) AS estimated_cost_usd, "
                                       "SUM(CASE WHEN estimated_cost_usd IS NULL THEN 1 ELSE 0 END) AS unpriced_turns "
                                       "FROM gateway_turns WHERE principal=? AND created_at>=?",
                                       (principal, time.time() - 86400)).fetchone())
            recent = [dict(row) for row in conn.execute("SELECT id,session,profile,target,effort,http_status,latency_ms,"
                                                       "usage_complete,estimated_cost_usd FROM gateway_turns "
                                                       "WHERE principal=? ORDER BY id DESC LIMIT 50", (principal,))]
            for row in recent:
                explanation = conn.execute("SELECT explanation_json FROM gateway_turn_explanations WHERE turn_id=?", (row.pop("id"),)).fetchone()
                row["explanation"] = json.loads(explanation[0]) if explanation else {}
            feedback=[dict(row) for row in conn.execute('SELECT session,rating,created_at FROM gateway_feedback WHERE principal=? ORDER BY id DESC LIMIT 50',(principal,))]
        from .gateway_budget import BudgetLedger
        return {"feedback":feedback,"usage_24h": summary, "recent_turns": recent, "budget": BudgetLedger().status(principal)}
