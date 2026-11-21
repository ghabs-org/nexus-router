from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_router_classifier as evaluator


def test_compute_classification_metrics():
    y_true = ["coding", "coding", "reasoning", "fast_utility"]
    y_pred = ["coding", "reasoning", "reasoning", "fast_utility"]
    metrics = evaluator.compute_classification_metrics(y_true, y_pred)

    assert metrics["total"] == 4
    assert metrics["correct"] == 3
    assert metrics["accuracy"] == 0.75
    assert metrics["per_label"]["fast_utility"]["f1"] == 1.0
    assert metrics["per_label"]["coding"]["recall"] == 0.5
    assert metrics["confusion_top"] == [{"expected": "coding", "predicted": "reasoning", "count": 1}]
    assert 0.7 < metrics["macro_f1"] < 0.85


def test_evaluate_and_gate_blocks_regression(tmp_path, monkeypatch):
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        "\n".join([
            json.dumps({"text": "Fix bug in router.py", "label": "coding"}),
            json.dumps({"text": "Compare Raft vs Paxos tradeoffs", "label": "reasoning"}),
            json.dumps({"text": "Translate this paragraph", "label": "fast_utility"}),
        ]) + "\n"
    )
    cand_dir = tmp_path / "cand"
    base_dir = tmp_path / "base"
    for d in (cand_dir, base_dir):
        d.mkdir()
        (d / "model.onnx").write_bytes(b"stub")

    class StubClassifier:
        available = True
        load_error = None

        def __init__(self, artifact_dir):
            self.dir = Path(artifact_dir)

        def classify(self, text, _signals):
            if self.dir == base_dir:
                # Baseline gets 3/3 right
                pred = "coding" if "Fix bug" in text else ("reasoning" if "Raft" in text else "fast_utility")
            else:
                # Candidate misclassifies reasoning -> coding
                pred = "fast_utility" if "Translate" in text else "coding"
            return SimpleNamespace(top_label=pred, confidence=0.85, margin=0.25)

    monkeypatch.setattr(evaluator, "LocalRouteClassifier", StubClassifier)
    report = evaluator.evaluate_and_gate(
        eval_file=eval_file,
        onnx_dir=cand_dir,
        baseline_onnx_dir=base_dir,
        min_macro_f1=0.5,
        min_macro_f1_delta=0.0,
    )
    assert report["baseline"]["macro_f1"] == 1.0
    assert report["candidate"]["macro_f1"] < 1.0
    assert report["promote"] is False

    # Passing candidate improves or ties baseline
    class GoodCandClassifier(StubClassifier):
        def classify(self, text, _signals):
            pred = "coding" if "Fix bug" in text else ("reasoning" if "Raft" in text else "fast_utility")
            return SimpleNamespace(top_label=pred, confidence=0.9, margin=0.4)

    monkeypatch.setattr(evaluator, "LocalRouteClassifier", GoodCandClassifier)
    report_ok = evaluator.evaluate_and_gate(
        eval_file=eval_file,
        onnx_dir=cand_dir,
        baseline_onnx_dir=base_dir,
    )
    assert report_ok["promote"] is True
