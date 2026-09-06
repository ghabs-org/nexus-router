from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.router import _select_primary_with_bandit
from src.types import ModelScore


def _score(model_id: str, total: float, task_fit: float) -> ModelScore:
    return ModelScore(
        model_id=model_id,
        provider="p",
        total_score=total,
        task_fit=task_fit,
        health=0.95,
        preference=0.50,
        learned=0.75,
        cost=0.70,
        speed=0.70,
        eco=0.60,
        component_scores={"task_fit": task_fit},
        component_contributions={"task_fit": round(task_fit * 0.5, 4)},
        score_mechanisms=["static_weights"],
    )


def test_bandit_shadow_mode_keeps_primary_and_logs_challenger():
    eligible = [
        _score("p/top", 0.91, 0.90),
        _score("p/newcomer", 0.89, 0.88),
    ]
    learned = {
        "p/top": {"total_selected": 120, "total_success": 108},
        "p/newcomer": {"total_selected": 0, "total_success": 0},
    }
    cfg = {
        "scoring": {
            "exploration": {
                "enabled": True,
                "epsilon": 1.0,
                "top_k": 2,
                "family_prior_strength": 6.0,
                "uncertainty_bonus": 0.2,
                "shadow_log_challenger": True,
                "seed": 42,
            }
        }
    }

    primary, mechanisms, note = _select_primary_with_bandit(
        eligible_scores=eligible,
        learned_stats=learned,
        route_mode="auto",
        provenance_mode="shadow",
        routing_config=cfg,
    )

    assert primary.model_id == "p/top"
    assert "bandit_shadow" in mechanisms
    assert note is not None and "bandit shadow challenger" in note


def test_bandit_route_mode_can_explore_uncertain_challenger():
    eligible = [
        _score("p/top", 0.92, 0.58),
        _score("p/newcomer", 0.90, 0.95),
    ]
    learned = {
        "p/top": {"total_selected": 300, "total_success": 285},
        "p/newcomer": {"total_selected": 0, "total_success": 0},
    }
    cfg = {
        "scoring": {
            "exploration": {
                "enabled": True,
                "epsilon": 1.0,
                "top_k": 2,
                "family_prior_strength": 8.0,
                "uncertainty_bonus": 0.5,
                "seed": 7,
            }
        }
    }

    primary, mechanisms, note = _select_primary_with_bandit(
        eligible_scores=eligible,
        learned_stats=learned,
        route_mode="auto",
        provenance_mode="route",
        routing_config=cfg,
    )

    assert "bandit_explore_thompson" in mechanisms
    assert note is not None and "bandit exploration selected challenger" in note
    assert primary.model_id == "p/newcomer"
    # Selection carries debug fields for explainability and cold-start audits.
    assert "bandit_sample" in primary.component_scores
    assert "bandit_uncertainty" in primary.component_scores
