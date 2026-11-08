"""GET /metrics: Prometheus text computed from SQLite on each scrape.

Counts survive restarts because they come from the decision/turn tables, not
in-memory counters. No prompt text, credentials or session keys are exported.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

from . import cooldowns, db
from .health import load_provider_health


def _label(value) -> str:
    """Escape a Prometheus label value (backslash, quote, newline)."""
    text = str(value if value is not None else "")
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _rows(sql: str) -> list:
    try:
        with closing(db._connect()) as conn:
            return conn.execute(sql).fetchall()
    except sqlite3.OperationalError:  # table not created yet
        return []


def prometheus_text() -> str:
    out: list[str] = []

    def metric(name, kind, help_text, samples):
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            rendered = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
            out.append(f"{name}{{{rendered}}} {value}" if rendered else f"{name} {value}")

    metric("nexus_router_decisions_total", "counter", "Routing decisions by task type and model.", [
        ({"task_type": r[0], "model": r[1]}, r[2]) for r in _rows(
            "SELECT task_type, selected_model, COUNT(*) FROM routing_decisions GROUP BY 1, 2")
    ])
    metric("nexus_router_outcomes_total", "counter", "Recorded turn outcomes by model and success.", [
        ({"model": r[0], "success": "true" if r[1] else "false"}, r[2]) for r in _rows(
            "SELECT selected_model, outcome_success, COUNT(*) FROM routing_decisions "
            "WHERE outcome_success IS NOT NULL GROUP BY 1, 2")
    ])
    metric("nexus_router_estimated_cost_usd_total", "counter", "Estimated spend by model (see cost_source).", [
        ({"model": r[0]}, round(r[1] or 0.0, 6)) for r in _rows(
            "SELECT selected_model, SUM(estimated_cost_usd) FROM routing_decisions "
            "WHERE estimated_cost_usd IS NOT NULL GROUP BY 1")
    ])
    metric("nexus_router_cli_turns_total", "counter", "CLI wire proxy turns by target and status.", [
        ({"target": r[0] or "none", "status": r[1] if r[1] is not None else "none"}, r[2]) for r in _rows(
            "SELECT target, http_status, COUNT(*) FROM cli_proxy_turns GROUP BY 1, 2")
    ])
    tokens = _rows("SELECT target, SUM(input_tokens), SUM(output_tokens) FROM cli_proxy_turns GROUP BY 1")
    metric("nexus_router_cli_tokens_total", "counter", "CLI wire proxy tokens by target and direction.", [
        ({"target": r[0] or "none", "direction": direction}, int(value or 0))
        for r in tokens for direction, value in (("input", r[1]), ("output", r[2]))
    ])
    metric("nexus_router_provider_health", "gauge", "Provider health score (0-1).", [
        ({"provider": name}, round(health.health_score, 4)) for name, health in sorted(load_provider_health().items())
    ])
    metric("nexus_router_model_cooldowns_active", "gauge", "Models currently in a per-model cooldown.", [
        ({}, len(cooldowns.active()))
    ])
    return "\n".join(out) + "\n"
