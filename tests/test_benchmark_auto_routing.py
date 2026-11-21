"""Published task evidence must outrank stale candidate lists and cold-start priors."""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import router as router_module
from src.generate_registry import normalize
from src.scorer import score_models
from src.types import ClassifierOutput, ProviderHealth


OLD = "openai/incumbent"
NEW = "openai/future-model"
POLICY = {"routing": {task: {"candidates": [{"model": OLD, "priority": 1}]}
                      for task in ("reasoning", "general_chat", "coding", "code_review", "summarization", "vision", "long_context")}}


def model(key, value, source="benchmark:artificialanalysis", dimension="reasoning"):
    sources = {dimension: source}
    if source.startswith("derived:benchmark_"):
        sources["reasoning"] = "benchmark:artificialanalysis"
        if "coding_reasoning" in source:
            sources["coding"] = "benchmark:artificialanalysis"
    return {
        "id": key, "provider": key.split("/", 1)[0],
        "availability": {"authed": True, "available": True},
        "features": {"contextWindow": 400_000, "supportsVision": True},
        "scores": {"reasoning": .7, "coding": .7, "review": .7, "summarize": .7,
                   "vision": .5, "context": .88, "cost": .7, "fast": .7, dimension: value},
        "scoreSource": sources,
    }


def score(models, mode="auto", task="reasoning", **kwargs):
    return score_models(
        ClassifierOutput(task_type=task, complexity="medium", confidence=.99),
        models, kwargs.pop("provider_health", {}), kwargs.pop("learned_stats", {}),
        routing_policy=POLICY, route_mode=mode, **kwargs,
    )


@pytest.mark.parametrize("mode", ["auto", "balanced", "off", None])
def test_unlisted_benchmark_leader_beats_static_order_and_cold_start(mode):
    ranked = score([model(OLD, .93), model(NEW, .96)], mode=mode,
                   learned_stats={OLD: {"total_selected": 1000, "success_rate": 1.0}})
    assert ranked[0].model_id == NEW
    assert ranked[0].total_score < ranked[1].total_score  # no incumbent/history bias
    assert ranked[0].learned == .75
    assert ranked[0].benchmark_score == .96
    assert "benchmark_priority" in ranked[0].score_mechanisms
    assert not ranked[0].excluded


def test_static_first_candidate_does_not_override_a_better_listed_benchmark():
    ranked = score_models(
        ClassifierOutput(task_type="reasoning"), [model(OLD, .8), model(NEW, .95)], {}, {},
        routing_policy={"routing": {"reasoning": {"candidates": [
            {"model": OLD, "priority": 1}, {"model": NEW, "priority": 2},
        ]}}}, route_mode="auto",
    )
    assert ranked[0].model_id == NEW


@pytest.mark.parametrize("source", ["global:default", "family:default", "override:manual",
                                     "benchmark:catalog-only", "benchmark:unknown",
                                     "benchmark-inherited:other/m:unknown",
                                     "benchmark-inherited:other/m:catalog-only",
                                     "twin:other/m:benchmark:unknown", ""])
def test_unmeasured_priors_do_not_masquerade_as_benchmarks(source):
    ranked = score([model(OLD, .8), model(NEW, .999, source)])
    new = next(s for s in ranked if s.model_id == NEW)
    assert ranked[0].model_id == OLD
    assert new.excluded and new.exclusion_reason == "not_in_task_policy"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True, "0.99"])
def test_invalid_benchmark_values_are_not_evidence(value):
    ranked = score([model(OLD, .8), model(NEW, value)])
    assert ranked[0].model_id == OLD
    assert next(s for s in ranked if s.model_id == NEW).excluded


@pytest.mark.parametrize("source", ["benchmark-inherited:other/same-model:artificialanalysis",
                                     "twin:other/same-model:benchmark:artificialanalysis"])
def test_same_model_cross_provider_provenance_qualifies(source):
    assert score([model(OLD, .8), model(NEW, .95, source)])[0].model_id == NEW


@pytest.mark.parametrize("task,dimension,source", [
    ("general_chat", "reasoning", "benchmark:artificialanalysis"),
    ("coding", "coding", "benchmark:livebench"),
    ("code_review", "review", "derived:benchmark_coding_reasoning"),
    ("summarization", "summarize", "derived:benchmark_reasoning"),
    ("vision", "reasoning", "benchmark:artificialanalysis"),
    ("long_context", "reasoning", "benchmark:artificialanalysis"),
])
def test_automatic_eligibility_uses_task_relevant_evidence(task, dimension, source):
    ranked = score([model(OLD, .8, source, dimension), model(NEW, .95, source, dimension)], task=task)
    assert ranked[0].model_id == NEW
    assert ranked[0].benchmark_dimension == dimension


def test_coding_only_evidence_does_not_claim_reasoning_quality():
    ranked = score([model(OLD, .8), model(NEW, .99, dimension="coding")])
    assert ranked[0].model_id == OLD
    assert next(s for s in ranked if s.model_id == NEW).excluded


@pytest.mark.parametrize("failure,expected", [
    ("auth", "not_authed"), ("available", "not_available"),
    ("cooldown", "model_cooldown"), ("health", "health_too_low"),
    ("vision", "no_vision_support"), ("context", "context_too_small"),
])
def test_benchmark_leader_cannot_bypass_safety_constraints(failure, expected):
    old, new = model(OLD, .8), model("challenger/future", .99)
    task, kwargs = "reasoning", {}
    if failure == "auth": new["availability"]["authed"] = False
    elif failure == "available": new["availability"]["available"] = False
    elif failure == "cooldown": kwargs["cooling_models"] = {new["id"]}
    elif failure == "health": kwargs["provider_health"] = {"challenger": ProviderHealth(provider="challenger", health_score=.1)}
    elif failure == "vision":
        task = "vision"
        new["scores"]["vision"] = 0
    elif failure == "context":
        task = "long_context"
        new["features"]["contextWindow"] = 8000
    ranked = score([old, new], task=task, **kwargs)
    assert ranked[0].model_id == OLD
    rejected = next(s for s in ranked if s.model_id == new["id"])
    assert rejected.excluded and rejected.exclusion_reason.startswith(expected)


def test_exhausted_benchmark_provider_yields_to_usable_provider():
    ranked = score([model(OLD, .8), model("challenger/future", .99)],
                   provider_health={"challenger": ProviderHealth(provider="challenger", quota="exhausted")})
    assert ranked[0].model_id == OLD


def test_explicit_modes_keep_their_own_cost_and_latency_objectives():
    ranked = score([model(OLD, .93), model(NEW, .96)], mode="fast", task="general_chat",
                   learned_stats={OLD: {"total_selected": 1000, "success_rate": 1.0}})
    assert ranked[0].model_id == OLD
    assert "benchmark_priority" not in ranked[0].score_mechanisms


def test_benchmarked_untagged_future_model_is_admitted_only_when_available_and_allowed():
    raw = {"modelPolicyAllow": ["openai/*"], "models": [
        {"key": OLD, "available": True, "tags": ["configured"]},
        {"key": NEW, "available": True},
        {"key": "blocked/future", "available": True},
        {"key": "openai/offline", "available": False},
        {"key": "openai/unmeasured", "available": True},
    ]}
    benchmarks = {"models": {key: {"reasoning": .99, "_source": "artificialanalysis"}
                              for key in (NEW, "blocked/future", "openai/offline")}}
    admitted = normalize(raw, {}, {}, benchmarks, configured_only=True)
    assert [m["id"] for m in admitted] == [OLD, NEW]
    for allow in (None, [], [OLD]):
        raw["modelPolicyAllow"] = allow
        assert [m["id"] for m in normalize(raw, {}, {}, benchmarks, configured_only=True)] == [OLD]


def test_catalog_only_rows_are_not_automatic_model_opt_ins():
    raw = {"modelPolicyAllow": ["openai/*"], "models": [{"key": NEW, "available": True}]}
    benchmarks = {"models": {NEW: {"reasoning": .99, "_source": "catalog-only"}}}
    assert normalize(raw, {}, {}, benchmarks, configured_only=True) == []


def test_tagless_or_unconfigured_catalog_cannot_bypass_explicit_allowlist():
    raw = {"modelPolicyAllow": [OLD], "models": [
        {"key": OLD, "available": True}, {"key": NEW, "available": True},
    ]}
    benchmarks = {"models": {key: {"reasoning": .99, "_source": "artificialanalysis"}
                              for key in (OLD, NEW)}}
    assert [m["id"] for m in normalize(raw, {}, {}, benchmarks, configured_only=False)] == [OLD]


def test_unknown_publisher_cannot_be_laundered_through_derived_scores():
    new = model(NEW, .99, "derived:benchmark_reasoning", "summarize")
    new["scoreSource"]["reasoning"] = "benchmark:unknown"
    ranked = score([model(OLD, .8, dimension="summarize"), new], task="summarization")
    assert ranked[0].model_id == OLD
    assert next(s for s in ranked if s.model_id == NEW).excluded


def test_refresh_adopts_a_future_benchmark_leader_without_policy_or_local_history(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_ROUTER_ENABLE_ROUTING_POLICY", "1")
    for name, value in (("load_provider_health", {}), ("load_model_stats", {}), ("load_model_metadata", {})):
        monkeypatch.setattr(router_module, name, lambda value=value: value)
    monkeypatch.setattr(router_module, "cooling_models", lambda task: set())
    registry = tmp_path / "models.json"
    policy = tmp_path / "routing.yaml"
    data = {"models": [model(OLD, .93), model(NEW, .88)]}
    registry.write_text(json.dumps(data))
    # Even forced exploration cannot overturn published evidence.
    config = copy.deepcopy(POLICY)
    config["scoring"] = {"exploration": {"enabled": True, "epsilon": 1.0, "seed": 7}}
    policy.write_text(json.dumps(config))
    router = router_module.Router(registry_path=registry, routing_path=policy, persist=False)
    classifier = ClassifierOutput(task_type="reasoning", complexity="high", confidence=.99)
    first = router.route(classifier, route_mode="auto")
    assert first.selected_model == OLD
    data["models"][1]["scores"]["reasoning"] = .97
    registry.write_text(json.dumps(data))
    stamp = registry.stat().st_mtime + 2
    os.utime(registry, (stamp, stamp))
    updated = router.route(classifier, route_mode="auto")
    assert updated.selected_model == NEW
    assert "benchmark_priority" in updated.mechanisms
    assert "bandit_explore_thompson" not in updated.mechanisms
    assert updated.explain["model_scores"][0]["benchmark"] == {
        "dimension": "reasoning", "score": .97, "source": "benchmark:artificialanalysis",
    }
    assert updated.decision_id is None
