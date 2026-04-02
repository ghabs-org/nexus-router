import json
import os
import sys
from pathlib import Path
import tempfile
import time

# allow importing scripts package during tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_tune import compute_adjustments, apply_adjustments, write_state, read_state, STATE_DIR, OVERRIDES_PATH


def make_report(entries):
    return {"ok": True, "model_task_signals_top": entries}


def test_bounded_delta_and_min_samples(tmp_path, monkeypatch):
    entries = [
        {"task": "coding", "model": "openai-codex/gpt-5.4", "samples": 10, "centered_signal": 0.8},
        {"task": "coding", "model": "openai-codex/gpt-5.3-codex", "samples": 2, "centered_signal": 1.0},
    ]
    report = make_report(entries)
    adjustments = compute_adjustments(report, max_delta=0.04, min_samples=3)
    assert "coding" in adjustments
    # second entry below min_samples should be skipped
    assert "openai-codex/gpt-5.4" in adjustments["coding"]
    assert "openai-codex/gpt-5.3-codex" not in adjustments["coding"]
    # capped to max_delta
    assert adjustments["coding"]["openai-codex/gpt-5.4"] <= 0.04


def test_cooldown_and_snapshot(tmp_path, monkeypatch):
    # ensure a clean state dir
    if STATE_DIR.exists():
        for f in STATE_DIR.glob('*'):
            try:
                f.unlink()
            except Exception:
                pass
    # prepare overrides file
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text('routing:\n')

    entries = [{"task": "general_chat", "model": "openai-codex/gpt-5.4", "samples": 5, "centered_signal": -1.0}]
    report = make_report(entries)
    adjustments = compute_adjustments(report, max_delta=0.02, min_samples=3)

    class Args: pass
    args = Args()
    args.apply = True
    args.cooldown_days = 1

    res1 = apply_adjustments(adjustments, args)
    # first apply should apply
    assert len(res1["applied"]) == 1
    # second immediate apply should be skipped due to cooldown
    res2 = apply_adjustments(adjustments, args)
    assert len(res2["applied"]) == 0
    assert len(res2["skipped"]) >= 1


if __name__ == '__main__':
    import pytest
    pytest.main([os.path.realpath(__file__)])
