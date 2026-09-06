from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_scoring_weights import fit_task_weights
from src.scorer import score_models
from src.types import ClassifierOutput, ProviderHealth


def _row(task: str, success: int, *, task_fit: float, health: float, preference: float, learned: float, cost: float, speed: float, eco: float) -> dict:
    return {
        "task_type": task,
        "outcome_success": int(success),
        "selected_component_scores": json.dumps(
            {
                "task_fit": task_fit,
                "health": health,
                "preference": preference,
                "learned": learned,
                "cost": cost,
                "speed": speed,
                "eco": eco,
            }
        ),
    }


def test_fit_task_weights_generates_normalized_weights():
    rows = []
    for _ in range(35):
        rows.append(_row("coding", 1, task_fit=0.92, health=0.90, preference=0.55, learned=0.80, cost=0.35, speed=0.45, eco=0.35))
    for _ in range(25):
        rows.append(_row("coding", 0, task_fit=0.55, health=0.65, preference=0.45, learned=0.52, cost=0.85, speed=0.88, eco=0.70))

    fitted, skipped = fit_task_weights(rows, min_samples=40)

    assert not skipped
    assert "coding" in fitted
    weights = fitted["coding"].weights
    assert pytest.approx(1.0, abs=1e-4) == sum(weights.values())
    assert set(weights.keys()) == {"task_fit", "health", "preference", "learned", "cost", "speed", "eco"}


def test_score_models_uses_fitted_weights_when_activated():
    models = [
        {
            "id": "p/quality",
            "provider": "p",
            "scores": {
                "coding": 0.96,
                "review": 0.80,
                "reasoning": 0.80,
                "summarize": 0.75,
                "fast": 0.40,
                "cost": 0.25,
                "eco": 0.30,
            },
            "features": {"contextWindow": 300_000, "is_free": False},
            "availability": {"authed": True},
        },
        {
            "id": "p/cheap",
            "provider": "p",
            "scores": {
                "coding": 0.72,
                "review": 0.70,
                "reasoning": 0.68,
                "summarize": 0.72,
                "fast": 0.92,
                "cost": 0.95,
                "eco": 0.84,
            },
            "features": {"contextWindow": 300_000, "is_free": True},
            "availability": {"authed": True},
        },
    ]
    health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
    classifier = ClassifierOutput(task_type="coding", complexity="medium", cost_profile="balanced", confidence=0.9)

    fitted_bundle = {
        "activated": True,
        "task_weights": {
            "coding": {
                "samples": 80,
                "weights": {
                    "task_fit": 0.08,
                    "health": 0.12,
                    "preference": 0.03,
                    "learned": 0.05,
                    "cost": 0.48,
                    "speed": 0.18,
                    "eco": 0.06,
                },
            }
        },
    }

    scored = score_models(
        classifier=classifier,
        models=models,
        provider_health=health,
        learned_stats={},
        fitted_weights_bundle=fitted_bundle,
        fitted_weights_min_samples=40,
    )
    eligible = [s for s in scored if not s.excluded]

    assert eligible[0].model_id == "p/cheap"
    assert "fitted_weights" in eligible[0].score_mechanisms


def test_score_models_falls_back_to_static_when_fitted_samples_thin():
    models = [
        {
            "id": "p/high-fit",
            "provider": "p",
            "scores": {"coding": 0.95, "review": 0.7, "reasoning": 0.7, "summarize": 0.7, "fast": 0.45, "cost": 0.35, "eco": 0.40},
            "features": {"contextWindow": 300_000, "is_free": False},
            "availability": {"authed": True},
        },
        {
            "id": "p/cheap",
            "provider": "p",
            "scores": {"coding": 0.72, "review": 0.7, "reasoning": 0.7, "summarize": 0.7, "fast": 0.94, "cost": 0.95, "eco": 0.85},
            "features": {"contextWindow": 300_000, "is_free": True},
            "availability": {"authed": True},
        },
    ]
    health = {"p": ProviderHealth(provider="p", auth="ok", quota="healthy", health_score=0.95)}
    classifier = ClassifierOutput(task_type="coding", complexity="medium", cost_profile="balanced", confidence=0.9)

    fitted_bundle = {
        "activated": True,
        "task_weights": {
            "coding": {
                "samples": 8,
                "weights": {
                    "task_fit": 0.05,
                    "health": 0.1,
                    "preference": 0.05,
                    "learned": 0.1,
                    "cost": 0.5,
                    "speed": 0.15,
                    "eco": 0.05,
                },
            }
        },
    }

    scored = score_models(
        classifier=classifier,
        models=models,
        provider_health=health,
        learned_stats={},
        fitted_weights_bundle=fitted_bundle,
        fitted_weights_min_samples=40,
    )
    eligible = [s for s in scored if not s.excluded]

    assert eligible[0].model_id == "p/high-fit"
    assert "static_weights" in eligible[0].score_mechanisms
