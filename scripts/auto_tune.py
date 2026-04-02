#!/usr/bin/env python3
"""Safe bounded auto-tuner for routing policies.

Reads scripts/feedback_calibration_report.py output (JSON) or router sqlite and
computes small bounded preference adjustments per task/model. By default runs
in --dry-run and prints a unified diff preview. Use --apply to write a
timestamped snapshot backup and update policies/tuning_overrides.yaml.

Features:
- bounded deltas (--max-delta)
- min samples gate (--min-samples)
- cooldown window between applies per task-model (--cooldown-days)
- dry-run default
- apply journal in state/tuning/journal.log

"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = Path(__file__).resolve().parent / "feedback_calibration_report.py"
POLICIES_PATH = ROOT / "policies"
OVERRIDES_PATH = POLICIES_PATH / "tuning_overrides.yaml"
STATE_DIR = Path.home() / ".local" / "state" / "nexus-router" / "tuning"
SNAP_DIR = STATE_DIR / "backups"
JOURNAL = STATE_DIR / "journal.log"
STATE_FILE = STATE_DIR / "state.json"

DEFAULTS = {
    "max_delta": 0.03,
    "min_samples": 5,
    "cooldown_days": 3,
}


def load_report() -> Dict[str, Any]:
    # Run report script and parse JSON
    if not REPORT_SCRIPT.exists():
        raise FileNotFoundError(f"report script missing: {REPORT_SCRIPT}")
    proc = os.popen(f'python3 "{REPORT_SCRIPT}"')
    out = proc.read()
    try:
        return json.loads(out)
    except Exception as e:
        raise RuntimeError(f"failed to parse report output: {e}\n{out}")


def load_overrides() -> Dict[str, Any]:
    if not OVERRIDES_PATH.exists():
        return {}
    with open(OVERRIDES_PATH) as f:
        data = yaml.safe_load(f)
        return data or {}


def save_overrides(data: Dict[str, Any]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDES_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def snapshot(path: Path) -> Path:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = SNAP_DIR / f"{path.name}.{ts}.bak"
    shutil.copy2(path, target)
    return target


def write_state(state: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def read_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"last_applied": {}}
    return json.load(open(STATE_FILE))


def append_journal(entry: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(entry + "\n")


def compute_adjustments(report: Dict[str, Any], max_delta: float, min_samples: int) -> Dict[str, Dict[str, float]]:
    """Return adjustments[task][model] = delta (signed float)

    Uses report['model_task_signals_top'] entries with 'centered_signal' in [-1,1]
    scaled to max_delta and gated by min_samples.
    """
    adjustments: Dict[str, Dict[str, float]] = {}
    for row in report.get("model_task_signals_top", []):
        samples = int(row.get("samples", 0))
        if samples < min_samples:
            continue
        task = row["task"]
        model = row["model"]
        centered = float(row.get("centered_signal", 0.0))
        # centered in [-1,1], positive means prefer bump, negative penalty
        delta = max(-max_delta, min(max_delta, centered * max_delta))
        if abs(delta) < 1e-8:
            continue
        adjustments.setdefault(task, {})[model] = round(delta, 6)
    return adjustments


def apply_adjustments(adjustments: Dict[str, Dict[str, float]], args) -> Dict[str, Any]:
    """Apply adjustments to OVERRIDES_PATH, respecting cooldown and recording state.

    Returns a dict with 'applied' and 'skipped' lists.
    """
    state = read_state()
    last_applied = state.get("last_applied", {})
    now_ts = int(time.time())
    cooldown_seconds = args.cooldown_days * 86400

    current = load_overrides()
    if not current.get("routing"):
        current["routing"] = {}

    applied = []
    skipped = []

    for task, models in adjustments.items():
        task_map = current["routing"].setdefault(task, {}).setdefault("preference_adjustments", {})
        for model, delta in models.items():
            key = f"{task}::{model}"
            last = last_applied.get(key, 0)
            if now_ts - last < cooldown_seconds:
                skipped.append({"task": task, "model": model, "reason": "cooldown"})
                continue
            prev = float(task_map.get(model, 0.0))
            new = prev + delta
            # enforce bounds [-0.5,0.5] absolute to avoid runaway
            new = max(-0.5, min(0.5, new))
            task_map[model] = round(new, 6)
            last_applied[key] = now_ts
            applied.append({"task": task, "model": model, "delta": delta, "new": task_map[model]})

    if args.apply and applied:
        # snapshot existing overrides (if present)
        if OVERRIDES_PATH.exists():
            snap = snapshot(OVERRIDES_PATH)
            append_journal(f"[{datetime.utcnow().isoformat()}] snapshot: {snap}")
        save_overrides(current)
        write_state({"last_applied": last_applied})
        append_journal(f"[{datetime.utcnow().isoformat()}] applied: {json.dumps(applied)}")

    return {"applied": applied, "skipped": skipped, "overrides_preview": current}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", dest="apply", action="store_false", help="preview only (default)")
    p.add_argument("--apply", dest="apply", action="store_true", help="apply changes (dangerous)")
    p.add_argument("--max-delta", type=float, default=DEFAULTS["max_delta"], help="maximum per-update delta magnitude")
    p.add_argument("--min-samples", type=int, default=DEFAULTS["min_samples"], help="minimum feedback samples required to act")
    p.add_argument("--cooldown-days", type=int, default=DEFAULTS["cooldown_days"], help="cooldown window for same task/model")
    args = p.parse_args(argv)

    # default mode dry-run
    if not args.apply:
        print("Running in dry-run mode. Use --apply to persist changes.")

    report = load_report()
    if not report.get("ok"):
        print("report not ok, aborting", report)
        return 2

    adjustments = compute_adjustments(report, args.max_delta, args.min_samples)
    if not adjustments:
        print("No actionable adjustments found.")
        return 0

    result = apply_adjustments(adjustments, args)

    # print concise summary + preview
    print(json.dumps({"applied": result["applied"], "skipped": result["skipped"]}, indent=2))
    # when dry-run, also print yaml preview
    if not args.apply:
        print("--- overrides preview ---")
        print(yaml.safe_dump(result["overrides_preview"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
