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
from datetime import datetime, timedelta, timezone
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
    "min_samples": 8,
    "cooldown_days": 3,
    "max_feedback_age_hours": 48,
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


def _parse_iso(ts: str | None):
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


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
    p.set_defaults(apply=False, guarded_apply=False)
    p.add_argument("--dry-run", dest="apply", action="store_false", help="preview only (default)")
    p.add_argument("--apply", dest="apply", action="store_true", help="apply changes (dangerous)")
    p.add_argument("--guarded-apply", dest="guarded_apply", action="store_true", help="attempt a guarded apply: gates must pass to persist")
    p.add_argument("--max-delta", type=float, default=DEFAULTS["max_delta"], help="maximum per-update delta magnitude")
    p.add_argument("--min-samples", type=int, default=DEFAULTS["min_samples"], help="minimum feedback samples required to act")
    p.add_argument("--min-total-samples", type=int, default=50, help="minimum total feedback samples across tasks to allow guarded apply")
    p.add_argument("--max-changes", type=int, default=10, help="maximum number of changed entries allowed for guarded apply")
    p.add_argument("--unknown-policy", choices=["ignore","require_higher"], default="ignore", help="how to treat task='unknown' rows in tuning")
    p.add_argument("--cooldown-days", type=int, default=DEFAULTS["cooldown_days"], help="cooldown window for same task/model")
    p.add_argument("--max-feedback-age-hours", type=int, default=DEFAULTS["max_feedback_age_hours"], help="ignore stale feedback signals older than this many hours")
    args = p.parse_args(argv)

    # mode messages
    mode = "dry-run"
    if args.apply:
        mode = "apply"
    if args.guarded_apply:
        mode = "guarded"

    if mode == "dry-run":
        print("Running in dry-run mode. Use --apply to persist changes or --guarded-apply for gated apply.")
    else:
        print(f"Running in {mode} mode.")

    report = load_report()
    if not report.get("ok"):
        print("report not ok, aborting", report)
        return 2

    # compute adjustments with unknown policy
    # If unknown-policy=require_higher, raise min_samples for unknown tasks
    base_min_samples = args.min_samples
    def min_samples_for(task):
        if task == "unknown" and args.unknown_policy == "require_higher":
            return max(base_min_samples * 3, base_min_samples + 10)
        return base_min_samples

    # prepare adjustments considering unknown policy
    adjustments = {}
    total_samples = 0
    raw_adjustments = []
    now_utc = datetime.now(timezone.utc)
    for row in report.get("model_task_signals_top", []):
        task = row.get("task")
        samples = int(row.get("samples", 0))

        # stale-signal gate: if we have recency info and it's too old, skip
        last_feedback_at = _parse_iso(row.get("last_feedback_at"))
        if last_feedback_at is not None:
            age_hours = (now_utc - last_feedback_at.astimezone(timezone.utc)).total_seconds() / 3600.0
            if age_hours > max(1, int(args.max_feedback_age_hours or 24)):
                continue

        total_samples += samples
        min_s = min_samples_for(task)
        if samples < min_s:
            continue
        model = row.get("model")
        centered = float(row.get("centered_signal", 0.0))
        delta = max(-args.max_delta, min(args.max_delta, centered * args.max_delta))
        if abs(delta) < 1e-8:
            continue
        adjustments.setdefault(task, {})[model] = round(delta, 6)
        raw_adjustments.append((task, model))

    if not adjustments:
        print("No actionable adjustments found.")
        return 0

    # gating for guarded apply
    gates_pass = True
    reasons = []
    if args.guarded_apply:
        if total_samples < args.min_total_samples:
            gates_pass = False
            reasons.append(f"total_samples ({total_samples}) < min_total_samples ({args.min_total_samples})")
        num_changes = sum(len(m) for m in adjustments.values())
        if num_changes > args.max_changes:
            gates_pass = False
            reasons.append(f"num_changes ({num_changes}) > max_changes ({args.max_changes})")
        # Guarded apply should only block on an explicit upstream critical-drift signal.
        # Do not infer critical drift from centered_signal alone: in this dataset,
        # explicit model feedback is often one-sided negative, which legitimately
        # produces centered_signal=-1.0 for many rows and would otherwise block
        # every bounded nightly tuning run.
        critical = bool(report.get("critical_drift", False))
        if critical:
            gates_pass = False
            reasons.append("critical drift detected")

    # decide whether to actually apply
    will_apply = bool(args.apply or (args.guarded_apply and gates_pass))

    class _ApplyArgs:
        def __init__(self, base_args, apply: bool):
            self.apply = apply
            self.cooldown_days = base_args.cooldown_days

    result = apply_adjustments(adjustments, _ApplyArgs(args, will_apply))

    # prepare concise summary for notifications
    recommendations_count = sum(len(m) for m in adjustments.values())
    applied_count = len(result.get("applied", []))
    snapshot_path = None
    if will_apply and applied_count:
        # last snapshot entry in journal could be parsed, but we recorded in apply_adjustments
        # For simplicity, list backups dir for latest
        if SNAP_DIR.exists():
            snaps = sorted(SNAP_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
            if snaps:
                snapshot_path = str(snaps[-1])

    # Print machine-friendly summary lines (one per line)
    print(f"mode={mode};recommendations={recommendations_count};applied={applied_count};snapshot={snapshot_path or ''}")
    if args.guarded_apply:
        print(f"gates_pass={gates_pass};reasons={'|'.join(reasons)}")

    # legacy json/preview output
    print(json.dumps({"applied": result["applied"], "skipped": result["skipped"]}, indent=2))
    if not will_apply:
        print("--- overrides preview ---")
        print(yaml.safe_dump(result["overrides_preview"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
