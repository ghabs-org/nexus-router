#!/usr/bin/env python3
"""Audit `general_chat` traffic and propose split candidates.

Constraints:
- Candidates must have >= min samples (default 50)
- Candidates must use distinct routing policies
- No new labels are introduced
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = Path.home() / ".local/state/nexus-router/data/router.sqlite"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit general_chat traffic")
    p.add_argument("--db", default=str(DB_PATH), help="Path to router.sqlite")
    p.add_argument("--source-type", default="raw-user", help="Filter routing_decisions.source_type")
    p.add_argument("--mode", default=None, help="Optional filter for routing mode")
    p.add_argument("--min-samples", type=int, default=50, help="Minimum examples per candidate split")
    return p.parse_args()


def _cluster_name(row: sqlite3.Row) -> str:
    text = str(row["message_text"] or "").strip()
    words = len(text.split())
    tokens = int(row["estimated_tokens"] or 0)
    if bool(row["has_code"]) or bool(row["has_diff"]) or bool(row["has_logs"]):
        return "technical_embedded"
    if bool(row["has_image"]):
        return "image_chat"
    if tokens > 12000:
        return "long_chat_context"
    if words <= 18:
        return "brief_chitchat"
    if "?" in text and words >= 20:
        return "advice_qna"
    return "general_misc"


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(Path(args.db).expanduser())
    conn.row_factory = sqlite3.Row

    decision_columns = {row["name"] for row in conn.execute("PRAGMA table_info(routing_decisions)").fetchall()}
    mode_expr = (
        "COALESCE(NULLIF(mode, ''), NULLIF(provenance_mode, ''), CASE WHEN COALESCE(shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "mode" in decision_columns and "provenance_mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(mode, ''), NULLIF(provenance_mode, ''), 'route')"
        if "mode" in decision_columns and "provenance_mode" in decision_columns
        else "COALESCE(NULLIF(mode, ''), CASE WHEN COALESCE(shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(mode, ''), 'route')"
        if "mode" in decision_columns
        else "COALESCE(NULLIF(provenance_mode, ''), CASE WHEN COALESCE(shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END)"
        if "provenance_mode" in decision_columns and "shadow_mode" in decision_columns
        else "COALESCE(NULLIF(provenance_mode, ''), 'route')"
        if "provenance_mode" in decision_columns
        else "CASE WHEN COALESCE(shadow_mode, 0) = 1 THEN 'shadow' ELSE 'route' END"
    )

    filters: list[str] = ["task_type='general_chat'", "message_text IS NOT NULL", "TRIM(message_text) != ''"]
    params: list[object] = []
    if args.source_type:
        filters.append("COALESCE(NULLIF(source_type, ''), 'standalone') = ?")
        params.append(args.source_type)
    if args.mode:
        filters.append(f"{mode_expr} = ?")
        params.append(args.mode)

    where_sql = "WHERE " + " AND ".join(filters)
    rows = conn.execute(
        f"""
        SELECT message_text, has_code, has_diff, has_logs, has_image, estimated_tokens
        FROM routing_decisions
        {where_sql}
        """,
        params,
    ).fetchall()
    conn.close()

    cluster_counts = Counter(_cluster_name(r) for r in rows)

    # Candidate policy proposals (must remain distinct once selected).
    policy_map = {
        "brief_chitchat": "fast",
        "advice_qna": "reasoning",
        "technical_embedded": "balanced",
        "long_chat_context": "balanced",
        "image_chat": "balanced",
        "general_misc": "balanced",
    }

    candidates = []
    used_policies: set[str] = set()
    for cluster, count in cluster_counts.most_common():
        policy = policy_map.get(cluster, "balanced")
        if count >= args.min_samples and policy not in used_policies:
            candidates.append({
                "cluster": cluster,
                "samples": int(count),
                "proposed_route_mode": policy,
            })
            used_policies.add(policy)

    payload = {
        "ok": True,
        "db": str(Path(args.db).expanduser()),
        "source_type": args.source_type,
        "mode": args.mode,
        "samples": len(rows),
        "clusters": [
            {"cluster": cluster, "samples": int(count)}
            for cluster, count in cluster_counts.most_common()
        ],
        "candidate_splits": candidates,
        "status": "candidates_found" if candidates else "no_candidate_split",
        "note": (
            "No split qualifies under min sample + distinct-policy constraints."
            if not candidates
            else "Only clusters with >= min samples and distinct route policy are listed."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
