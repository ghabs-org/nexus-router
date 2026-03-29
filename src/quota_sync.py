#!/usr/bin/env python3
"""
quota_sync.py — Generic provider quota sync job for Nexus Router.

This job ingests provider health snapshots from JSON and updates the router's
runtime health state. It is intentionally provider-agnostic: a separate
collector can decide *how* to measure quota, and this module only records the
result.

Supported input shapes:
- a single snapshot object
- a list of snapshot objects
- an object with a top-level "providers" list

Each snapshot can include:
- provider: required provider id (e.g. openai-codex)
- auth: optional auth status (ok|expired|missing|unknown)
- quota_state: optional quota state (healthy|low|exhausted|unknown)
- quota_remaining_ratio: optional float in [0.0, 1.0]
- latency_ms: optional latency sample
- http_status: optional HTTP status sample
- error_type: optional error label
- note: optional free-form note
- source: optional source label for logging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from .db import log_provider_observation
from .health import record_observation


def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("providers"), list):
        return [item for item in payload["providers"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("quota snapshot must be an object, list, or {providers:[...]}")


def _derive_quota_state(snapshot: dict[str, Any]) -> Optional[str]:
    quota_state = snapshot.get("quota_state")
    if quota_state:
        return str(quota_state).strip().lower()

    ratio = snapshot.get("quota_remaining_ratio")
    if ratio is None:
        return None

    try:
        ratio_f = max(0.0, min(1.0, float(ratio)))
    except (TypeError, ValueError):
        return None

    if ratio_f <= 0.02:
        return "exhausted"
    if ratio_f <= 0.25:
        return "low"
    return "healthy"


def sync_provider_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    provider = str(snapshot.get("provider", "")).strip()
    if not provider:
        raise ValueError("provider is required")

    auth_status = str(snapshot.get("auth", "unknown") or "unknown").strip().lower()
    quota_state = _derive_quota_state(snapshot)
    quota_remaining_ratio = snapshot.get("quota_remaining_ratio")
    latency_ms = snapshot.get("latency_ms")
    http_status = snapshot.get("http_status")
    error_type = snapshot.get("error_type")
    note = snapshot.get("note") or snapshot.get("source")

    if quota_remaining_ratio is not None:
        try:
            quota_remaining_ratio = max(0.0, min(1.0, float(quota_remaining_ratio)))
        except (TypeError, ValueError):
            quota_remaining_ratio = None

    record_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state,
        error_type=error_type,
        latency_ms=int(latency_ms) if latency_ms is not None else None,
        http_status=int(http_status) if http_status is not None else None,
        quota_remaining_ratio=quota_remaining_ratio,
    )

    log_provider_observation(
        provider=provider,
        auth_status=auth_status,
        quota_state=quota_state or "unknown",
        http_status=int(http_status) if http_status is not None else None,
        error_type=error_type,
        latency_ms=int(latency_ms) if latency_ms is not None else None,
        note=note,
    )

    return {
        "provider": provider,
        "auth": auth_status,
        "quota": quota_state or "unknown",
        "quota_remaining_ratio": quota_remaining_ratio,
        "latency_ms": latency_ms,
        "http_status": http_status,
        "error_type": error_type,
        "note": note,
    }


def sync_snapshots(snapshots: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for snapshot in snapshots:
        results.append(sync_provider_snapshot(snapshot))
    return results


def load_snapshots_from_path(path: str, skip_missing: bool = False) -> list[dict[str, Any]]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        file_path = Path(path)
        if not file_path.exists():
            if skip_missing:
                return []
            raise FileNotFoundError(path)
        raw = file_path.read_text()

    payload = json.loads(raw)
    return _normalize_payload(payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync provider quota snapshots into Nexus Router health")
    parser.add_argument("--input", default="-", help="JSON file path or - for stdin")
    parser.add_argument("--skip-missing", action="store_true", help="Exit 0 if the input file is missing")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print sync results")
    args = parser.parse_args(argv)

    snapshots = load_snapshots_from_path(args.input, skip_missing=args.skip_missing)
    if not snapshots and args.skip_missing:
        if args.pretty:
            print(json.dumps({"synced": []}, indent=2))
        else:
            print(json.dumps({"synced": 0}, separators=(",", ":")))
        return 0

    results = sync_snapshots(snapshots)

    if args.pretty:
        print(json.dumps({"synced": results}, indent=2))
    else:
        print(json.dumps({"synced": len(results)}, separators=(",", ":")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
