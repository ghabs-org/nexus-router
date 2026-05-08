#!/usr/bin/env python3
"""Rollback utility for auto_tune snapshots.

Usage:
  --latest : restore the most recent snapshot
  --snapshot <path> : restore specific snapshot

Will validate snapshot exists, write a journal entry, and copy snapshot to policies/tuning_overrides.yaml
"""
from pathlib import Path
import argparse
import shutil
import json
import os
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "policies"
OVERRIDES = POLICIES / "tuning_overrides.yaml"
STATE_DIR = Path(os.environ.get("NEXUS_ROUTER_TUNING_STATE_DIR", Path.home() / ".local" / "state" / "nexus-router" / "tuning"))
SNAP_DIR = STATE_DIR / "backups"
JOURNAL = STATE_DIR / "journal.log"


def append_journal(line: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(line + "\n")


def find_latest_snapshot():
    if not SNAP_DIR.exists():
        return None
    snaps = sorted(SNAP_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    return snaps[-1] if snaps else None


def main(argv=None):
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="restore most recent snapshot")
    group.add_argument("--snapshot", type=str, help="path to snapshot file to restore")
    args = p.parse_args(argv)

    if args.latest:
        snap = find_latest_snapshot()
        if not snap:
            print("no snapshots available")
            return 2
    else:
        snap = Path(args.snapshot)
        if not snap.exists():
            print(f"snapshot not found: {snap}")
            return 2

    # validate
    if not snap.exists():
        print(f"snapshot not found: {snap}")
        return 2

    # perform restore
    POLICIES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snap, OVERRIDES)
    append_journal(f"[{datetime.utcnow().isoformat()}] rollback: restored {snap}")
    print(f"restored snapshot to {OVERRIDES}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
