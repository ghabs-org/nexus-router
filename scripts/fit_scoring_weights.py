#!/usr/bin/env python3
"""Fit per-task scoring weights from routing outcomes.

Uses lightweight logistic regression on selected-model score components and
turn outcome_success (win/loss). Output is a task->weight map for scorer.

Activation guard:
- candidate weights are always written
- active weights are only written when golden-set eval passes
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from scripts.golden_route_eval import evaluate_golden_set
from src.db import _connect
from src.scorer import DEFAULT_WEIGHTS

WEIGHT_KEYS = ["task_fit", "health", "preference", "learned", "cost", "speed", "eco"]


@dataclass
class TaskFitResult:
    task_type: str
    samples: int
    positive_rate: float
    weights: dict[str, float]
    blend_factor: float


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_logistic_weights(x: list[list[float]], y: list[float], *, epochs: int = 700, lr: float = 0.20, l2: float = 0.08) -> list[float]:
    n = len(x)
    d = len(x[0]) if n else 0
    w = [0.0] * d
    p = min(0.99, max(0.01, sum(y) / n))
    b = math.log(p / (1.0 - p))

    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for i in range(n):
            z = b
            row = x[i]
            for j in range(d):
                z += row[j] * w[j]
            pred = _sigmoid(z)
            err = pred - y[i]
            grad_b += err
            for j in range(d):
                grad_w[j] += row[j] * err

        inv_n = 1.0 / float(n)
        for j in range(d):
            grad = (grad_w[j] * inv_n) + (l2 * w[j])
            w[j] -= lr * grad
        b -= lr * (grad_b * inv_n)

    return w


def _normalize_positive_weights(raw: list[float], fallback: dict[str, float], blend_factor: float) -> dict[str, float]:
    positive = [max(0.0, v) for v in raw]
    total_pos = sum(positive)
    if total_pos <= 1e-9:
        base = [float(fallback[k]) for k in WEIGHT_KEYS]
    else:
        base = [v / total_pos for v in positive]

    default = [float(fallback[k]) for k in WEIGHT_KEYS]
    mixed = [((1.0 - blend_factor) * default[i]) + (blend_factor * base[i]) for i in range(len(WEIGHT_KEYS))]
    mixed = [max(0.0, v) for v in mixed]
    total = sum(mixed) or 1.0
    return {k: round(float(v / total), 6) for k, v in zip(WEIGHT_KEYS, mixed)}


def _fetch_rows(include_shadow: bool) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        mode_filter = "" if include_shadow else "AND COALESCE(NULLIF(mode, ''), 'route') = 'route'"
        rows = conn.execute(
            f"""
            SELECT task_type, outcome_success, selected_component_scores
            FROM routing_decisions
            WHERE selected_component_scores IS NOT NULL
              AND TRIM(selected_component_scores) != ''
              AND outcome_success IS NOT NULL
              {mode_filter}
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _build_matrix(task_rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[float]]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []

    for row in task_rows:
        try:
            payload = json.loads(row["selected_component_scores"])
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        feature: list[float] = []
        valid = True
        for key in WEIGHT_KEYS:
            value = payload.get(key)
            if not isinstance(value, (int, float)):
                valid = False
                break
            feature.append(float(value))
        if not valid:
            continue

        y_value = row.get("outcome_success")
        if y_value is None:
            continue
        y_rows.append(float(int(y_value)))
        x_rows.append(feature)

    return x_rows, y_rows


def fit_task_weights(rows: list[dict[str, Any]], min_samples: int) -> tuple[dict[str, TaskFitResult], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = str(row.get("task_type") or "general_chat")
        by_task.setdefault(task, []).append(row)

    fitted: dict[str, TaskFitResult] = {}
    skipped: list[dict[str, Any]] = []

    for task, task_rows in sorted(by_task.items()):
        x, y = _build_matrix(task_rows)
        samples = len(y)
        positives = int(sum(y))
        negatives = samples - positives

        if samples < min_samples:
            skipped.append({"task": task, "reason": f"too_few_samples:{samples}", "samples": samples})
            continue
        if positives == 0 or negatives == 0:
            skipped.append({"task": task, "reason": "single_class_labels", "samples": samples})
            continue

        raw = _fit_logistic_weights(x, y)
        blend_factor = min(0.70, samples / (samples + 120.0))
        weights = _normalize_positive_weights(raw, DEFAULT_WEIGHTS, blend_factor)

        fitted[task] = TaskFitResult(
            task_type=task,
            samples=samples,
            positive_rate=round(float(positives / samples), 4),
            weights=weights,
            blend_factor=round(float(blend_factor), 4),
        )

    return fitted, skipped


def _jsonable_task_fit(result: TaskFitResult) -> dict[str, Any]:
    return {
        "samples": result.samples,
        "positive_rate": result.positive_rate,
        "blend_factor": result.blend_factor,
        "weights": result.weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-task scoring weights from routing outcomes")
    parser.add_argument("--candidate-output", required=True, help="Path for candidate JSON artifact")
    parser.add_argument("--active-output", required=True, help="Path for active JSON artifact")
    parser.add_argument("--min-samples-per-task", type=int, default=40)
    parser.add_argument("--include-shadow", action="store_true", help="Include shadow mode outcomes")
    parser.add_argument("--activate-on-golden-pass", action="store_true")
    args = parser.parse_args()

    rows = _fetch_rows(include_shadow=bool(args.include_shadow))
    fitted, skipped = fit_task_weights(rows, min_samples=max(1, int(args.min_samples_per_task)))

    candidate_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_samples_per_task": int(args.min_samples_per_task),
        "activated": False,
        "golden_ok": False,
        "rows_scanned": len(rows),
        "task_weights": {task: _jsonable_task_fit(data) for task, data in fitted.items()},
        "skipped": skipped,
    }

    candidate_path = Path(args.candidate_output).expanduser()
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate_payload, indent=2) + "\n")

    active_payload = dict(candidate_payload)
    if args.activate_on_golden_pass and fitted:
        result = evaluate_golden_set(fitted_weights_bundle=candidate_payload)
        active_payload["golden_ok"] = bool(result.get("ok"))
        active_payload["golden_result"] = result
        active_payload["activated"] = bool(result.get("ok"))
    else:
        active_payload["golden_result"] = {"ok": False, "reason": "golden gate not requested or no fitted tasks"}

    active_path = Path(args.active_output).expanduser()
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(active_payload, indent=2) + "\n")

    print(
        json.dumps(
            {
                "rows_scanned": len(rows),
                "fitted_tasks": sorted(fitted.keys()),
                "candidate_output": str(candidate_path),
                "active_output": str(active_path),
                "activated": bool(active_payload.get("activated", False)),
                "golden_ok": bool(active_payload.get("golden_ok", False)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
