#!/usr/bin/env python3
"""Evaluate candidate ONNX classifier artifacts against held-out eval data.

Uses src.local_classifier.LocalRouteClassifier directly so tokenization and
softmax parity match production runtime exactly. Exits 0 if promotion gate
passes, 2 if the gate blocks promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_classifier import LocalRouteClassifier
from src.local_classifier_labels import ROUTER_TASK_LABELS
from src.types import PreSignals


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: tuple[str, ...] = ROUTER_TASK_LABELS,
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted F1 (across supported true labels), and confusions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    seen = set(labels) | set(y_true) | set(y_pred)

    per_label: dict[str, dict[str, Any]] = {}
    supported_f1s: list[float] = []
    weighted_f1_sum = 0.0

    for label in sorted(seen):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn
        if support == 0 and fp == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            supported_f1s.append(f1)
            weighted_f1_sum += f1 * support

    confusions = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    confusion_top = [
        {"expected": exp, "predicted": pred, "count": cnt}
        for (exp, pred), cnt in confusions.most_common(10)
    ]

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(supported_f1s) / max(1, len(supported_f1s)), 4),
        "weighted_f1": round(weighted_f1_sum / total, 4),
        "per_label": per_label,
        "confusion_top": confusion_top,
    }


def load_eval_rows(eval_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(eval_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        text = str(item.get("text", "")).strip()
        label = str(item.get("label", "")).strip()
        if not text or not label:
            raise ValueError(f"{eval_path}: line {idx} missing text or label")
        rows.append(item)
    if not rows:
        raise ValueError(f"{eval_path}: no evaluation rows found")
    return rows


def evaluate_onnx_dir(onnx_dir: Path, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifier = LocalRouteClassifier(artifact_dir=onnx_dir)
    if not classifier.available:
        raise RuntimeError(f"ONNX classifier unavailable at {onnx_dir}: {classifier.load_error}")

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    margins: list[float] = []

    for row in eval_rows:
        text = str(row["text"])
        result = classifier.classify(text, PreSignals(message_length=len(text)))
        y_true.append(str(row["label"]))
        if result is None:
            y_pred.append("unknown")
        else:
            y_pred.append(result.top_label)
            confidences.append(result.confidence)
            margins.append(result.margin)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics.update(
        {
            "onnx_dir": str(onnx_dir),
            "mean_confidence": round(sum(confidences) / max(1, len(confidences)), 4),
            "mean_margin": round(sum(margins) / max(1, len(margins)), 4),
        }
    )
    return metrics


def check_promotion_gate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    *,
    min_macro_f1: float = 0.0,
    min_macro_f1_delta: float = 0.0,
) -> tuple[bool, str]:
    cand_f1 = float(candidate_metrics["macro_f1"])
    if cand_f1 < float(min_macro_f1):
        return False, f"candidate macro_f1 {cand_f1:.4f} below floor {min_macro_f1:.4f}"
    if baseline_metrics is not None:
        base_f1 = float(baseline_metrics["macro_f1"])
        delta = cand_f1 - base_f1
        if delta < float(min_macro_f1_delta):
            return (
                False,
                f"candidate macro_f1 {cand_f1:.4f} regressed vs baseline {base_f1:.4f} "
                f"(delta={delta:+.4f}, min_delta={min_macro_f1_delta:+.4f})",
            )
        return (
            True,
            f"candidate macro_f1 {cand_f1:.4f} passed vs baseline {base_f1:.4f} (delta={delta:+.4f})",
        )
    return True, f"candidate macro_f1 {cand_f1:.4f} passed floor {min_macro_f1:.4f} (no baseline)"


def evaluate_and_gate(
    *,
    eval_file: Path,
    onnx_dir: Path,
    baseline_onnx_dir: Path | None = None,
    min_macro_f1: float = 0.0,
    min_macro_f1_delta: float = 0.0,
) -> dict[str, Any]:
    eval_rows = load_eval_rows(eval_file)
    candidate_metrics = evaluate_onnx_dir(onnx_dir, eval_rows)

    baseline_metrics: dict[str, Any] | None = None
    if baseline_onnx_dir is not None and (baseline_onnx_dir / "model.onnx").exists():
        try:
            baseline_metrics = evaluate_onnx_dir(baseline_onnx_dir, eval_rows)
        except Exception as exc:
            baseline_metrics = None
            baseline_error = str(exc)
        else:
            baseline_error = None
    else:
        baseline_error = None

    passed, reason = check_promotion_gate(
        candidate_metrics,
        baseline_metrics,
        min_macro_f1=min_macro_f1,
        min_macro_f1_delta=min_macro_f1_delta,
    )
    report: dict[str, Any] = {
        "eval_file": str(eval_file),
        "samples": len(eval_rows),
        "promote": passed,
        "gate_reason": reason,
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
    }
    if baseline_error:
        report["baseline_error"] = baseline_error
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate ONNX classifier and check promotion gate")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--baseline-onnx-dir", type=Path, default=None)
    parser.add_argument("--min-macro-f1", type=float, default=0.0)
    parser.add_argument("--min-macro-f1-delta", type=float, default=0.0)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_and_gate(
        eval_file=args.eval_file,
        onnx_dir=args.onnx_dir,
        baseline_onnx_dir=args.baseline_onnx_dir,
        min_macro_f1=args.min_macro_f1,
        min_macro_f1_delta=args.min_macro_f1_delta,
    )
    text = json.dumps(report, indent=2)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n")
    print(text)
    return 0 if report["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
