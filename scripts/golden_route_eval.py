#!/usr/bin/env python3
"""Golden-set routing regression check.

Purpose:
- freeze a prompt/task corpus with acceptable routing outcomes
- run on scorer/policy changes via tests
- fail if selected route regresses outside acceptable set
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataclasses import replace

from src.scorer import score_models
from src.types import ClassifierOutput, ProviderHealth

GOLDEN_PATH = ROOT / "tests" / "golden_route_set.json"


def _models() -> list[dict[str, Any]]:
    return [
        {
            "id": "lab/quality-coder",
            "provider": "lab-quality",
            "scores": {"coding": 0.96, "review": 0.92, "reasoning": 0.90, "summarize": 0.76, "fast": 0.55, "cost": 0.35, "eco": 0.35, "context": 0.86, "vision": 0.66},
            "features": {"contextWindow": 600_000, "is_free": False},
            "availability": {"authed": True},
        },
        {
            "id": "lab/deep-reasoner",
            "provider": "lab-reason",
            "scores": {"coding": 0.86, "review": 0.84, "reasoning": 0.98, "summarize": 0.78, "fast": 0.50, "cost": 0.30, "eco": 0.30, "context": 0.88, "vision": 0.64},
            "features": {"contextWindow": 700_000, "is_free": False},
            "availability": {"authed": True},
        },
        {
            "id": "lab/fast-cheap",
            "provider": "lab-fast",
            "scores": {"coding": 0.78, "review": 0.74, "reasoning": 0.72, "summarize": 0.82, "fast": 0.97, "cost": 0.97, "eco": 0.72, "context": 0.55, "vision": 0.42},
            "features": {"contextWindow": 120_000, "is_free": True},
            "availability": {"authed": True},
        },
        {
            "id": "lab/eco-efficient",
            "provider": "lab-eco",
            "scores": {"coding": 0.74, "review": 0.72, "reasoning": 0.75, "summarize": 0.80, "fast": 0.90, "cost": 0.92, "eco": 0.98, "context": 0.58, "vision": 0.40},
            "features": {"contextWindow": 100_000, "is_free": True},
            "availability": {"authed": True},
        },
    ]


def _health() -> dict[str, ProviderHealth]:
    return {
        "lab-quality": ProviderHealth(provider="lab-quality", auth="ok", quota="healthy", health_score=0.95, latency_ms_p50=1550, last_check_at="2026-01-01T00:00:00+00:00"),
        "lab-reason": ProviderHealth(provider="lab-reason", auth="ok", quota="healthy", health_score=0.93, latency_ms_p50=1800, last_check_at="2026-01-01T00:00:00+00:00"),
        "lab-fast": ProviderHealth(provider="lab-fast", auth="ok", quota="healthy", health_score=0.92, latency_ms_p50=180, last_check_at="2026-01-01T00:00:00+00:00"),
        "lab-eco": ProviderHealth(provider="lab-eco", auth="ok", quota="healthy", health_score=0.90, latency_ms_p50=240, last_check_at="2026-01-01T00:00:00+00:00"),
    }




def _apply_golden_confidence_gate(classifier: ClassifierOutput, route_mode: str, row: dict[str, Any]) -> tuple[ClassifierOutput, list[str]]:
    mechanisms = ["static_weights"]
    threshold = float(row.get("confidence_gate_threshold") or 0.65)
    if route_mode not in {"auto", "balanced", "off"}:
        return classifier, mechanisms
    if classifier.confidence >= threshold:
        return classifier, mechanisms

    gated = replace(classifier, task_type="general_chat", subtype=None, complexity="medium", cost_profile="balanced")
    mechanisms.append("confidence_gate")
    return gated, mechanisms

def evaluate_golden_set(path: Path = GOLDEN_PATH, fitted_weights_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("golden set must be a JSON array")

    models = _models()
    health = _health()
    failures: list[dict[str, Any]] = []

    for idx, row in enumerate(payload):
        task = str(row.get("expected_task") or "general_chat")
        route_mode = str(row.get("route_mode") or "auto")
        acceptable = [str(x) for x in (row.get("acceptable_routes") or []) if str(x).strip()]
        if not acceptable:
            raise ValueError(f"row {idx} missing acceptable_routes")

        classifier = ClassifierOutput(
            task_type=task,
            complexity=str(row.get("complexity") or "medium"),
            cost_profile=str(row.get("cost_profile") or "balanced"),
            confidence=float(row.get("confidence") or 0.82),
        )
        effective_classifier, mechanisms = _apply_golden_confidence_gate(classifier, route_mode, row)

        scored = score_models(
            classifier=effective_classifier,
            models=models,
            provider_health=health,
            learned_stats={},
            route_mode=route_mode,
            fitted_weights_bundle=fitted_weights_bundle,
            free_only=bool(row.get("free_only") or False),
        )
        eligible = [s for s in scored if not s.excluded]
        selected = eligible[0].model_id if eligible else None
        case_id = row.get("id") or f"case-{idx}"

        if selected not in acceptable:
            failures.append(
                {
                    "index": idx,
                    "id": case_id,
                    "prompt": row.get("prompt"),
                    "expected_task": task,
                    "effective_task": effective_classifier.task_type,
                    "route_mode": route_mode,
                    "selected": selected,
                    "acceptable_routes": acceptable,
                }
            )

        expected_effective_task = row.get("expected_effective_task")
        if expected_effective_task and effective_classifier.task_type != str(expected_effective_task):
            failures.append({
                "index": idx,
                "id": case_id,
                "error": "effective_task_mismatch",
                "expected_effective_task": expected_effective_task,
                "actual_effective_task": effective_classifier.task_type,
            })

        expected_mechanisms = [str(x) for x in (row.get("expected_mechanisms") or []) if str(x).strip()]
        if expected_mechanisms and any(m not in mechanisms for m in expected_mechanisms):
            failures.append({
                "index": idx,
                "id": case_id,
                "error": "mechanism_missing",
                "expected_mechanisms": expected_mechanisms,
                "actual_mechanisms": mechanisms,
            })

        if row.get("require_score_breakdown", True):
            top = eligible[0] if eligible else None
            if not top or not top.component_scores or not top.component_contributions:
                failures.append({
                    "index": idx,
                    "id": case_id,
                    "error": "missing_score_breakdown",
                })

    return {
        "ok": not failures,
        "total": len(payload),
        "failures": failures,
    }


def main() -> None:
    result = evaluate_golden_set()
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
