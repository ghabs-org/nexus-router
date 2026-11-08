#!/usr/bin/env python3
"""Compare A/B routing policies (routing_decisions.policy_id). Read-only.

Per policy: decisions, outcome success %, avg latency, fallback %, user override %,
and feedback correct % (route_feedback verdicts).

    python scripts/policy_experiment_report.py [--experiment ID] [--days 14] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ROUTER_DB_PATH  # noqa: E402

QUERY = """
SELECT d.policy_id AS policy_id,
       COUNT(*) AS decisions,
       SUM(d.outcome_success IS NOT NULL) AS outcomes,
       AVG(d.outcome_success) AS success_rate,
       AVG(d.latency_ms) AS avg_latency_ms,
       AVG(COALESCE(d.fallback_used, 0)) AS fallback_rate,
       AVG(COALESCE(d.user_override, 0)) AS override_rate,
       SUM(f.correct) AS feedback_correct,
       SUM(f.total) AS feedback_total
FROM routing_decisions d
LEFT JOIN (
    SELECT decision_id, SUM(verdict = 'correct') AS correct, COUNT(*) AS total
    FROM route_feedback GROUP BY decision_id
) f ON f.decision_id = d.id
WHERE d.policy_id IS NOT NULL AND d.created_at >= ? AND (? IS NULL OR d.policy_id LIKE ? || ':%')
GROUP BY d.policy_id
ORDER BY d.policy_id
"""


def report(conn: sqlite3.Connection, experiment: str | None, days: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn.row_factory = sqlite3.Row
    rows = []
    for row in conn.execute(QUERY, (since, experiment, experiment)):
        item = dict(row)
        total = item.pop("feedback_total") or 0
        correct = item.pop("feedback_correct") or 0
        item["feedback"] = total
        item["feedback_correct_rate"] = round(correct / total, 3) if total else None
        for key in ("success_rate", "fallback_rate", "override_rate"):
            item[key] = round(item[key], 3) if item[key] is not None else None
        item["avg_latency_ms"] = round(item["avg_latency_ms"]) if item["avg_latency_ms"] is not None else None
        rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(ROUTER_DB_PATH))
    parser.add_argument("--experiment", help="experiment id (policy_id prefix)")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
            rows = report(conn, args.experiment, args.days)
    except sqlite3.OperationalError as exc:
        # Read-only on purpose: the router server adds policy_id at startup.
        print(f"Cannot read policy data from {args.db}: {exc} (start the router once to migrate)", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No decisions with a policy_id in the window.")
        return 0
    header = ("policy", "n", "outcomes", "success", "latency", "fallback", "override", "feedback", "correct")
    print("  ".join(f"{h:>10}" for h in header))
    for r in rows:
        values = (r["policy_id"], r["decisions"], r["outcomes"], r["success_rate"], r["avg_latency_ms"],
                  r["fallback_rate"], r["override_rate"], r["feedback"], r["feedback_correct_rate"])
        print("  ".join(f"{'-' if v is None else v!s:>10}" for v in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
