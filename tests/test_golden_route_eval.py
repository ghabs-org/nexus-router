from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.golden_route_eval import evaluate_golden_set


def test_golden_route_set_has_no_regressions():
    result = evaluate_golden_set()
    assert result["total"] >= 50
    assert result["ok"], result["failures"]
