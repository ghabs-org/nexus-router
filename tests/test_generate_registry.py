import os
import sys
from pathlib import Path

os.environ.setdefault("NEXUS_ROUTER_STATE_ROOT", str(Path(__file__).parent.parent / ".test-state"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.generate_registry as generate_registry
from src.generate_registry import normalize, resolve_scores


def _models(provider: str, count: int, configured: bool = True) -> list[dict]:
    tags = ["configured"] if configured else []
    return [{"key": f"{provider}/m{i}", "available": True, "tags": tags} for i in range(count)]


def test_normalize_configured_only_skips_unconfigured_models():
    raw = {"models": _models("openrouter", 2) + _models("openrouter-extra", 3, configured=False)}
    benchmarks = {"models": {}}

    configured = normalize(raw, {}, {}, benchmarks, configured_only=True)
    everything = normalize(raw, {}, {}, benchmarks)

    assert sorted(m["id"] for m in configured) == ["openrouter/m0", "openrouter/m1"]
    assert len(everything) == 5


def test_normalize_configured_only_skips_gemini_cli_clones(monkeypatch):
    monkeypatch.setattr(generate_registry, "_gemini_cli_enabled", lambda: True)
    raw = {"models": [{"key": "google/gemini-3.1-pro-preview", "available": True, "tags": ["configured"]}]}

    configured = normalize(raw, {}, {}, {"models": {}}, configured_only=True)
    everything = normalize(raw, {}, {}, {"models": {}})

    assert [m["id"] for m in configured] == ["google/gemini-3.1-pro-preview"]
    assert sorted(m["id"] for m in everything) == [
        "google-gemini-cli/gemini-3.1-pro-preview",
        "google/gemini-3.1-pro-preview",
    ]


def _context_score(window: int) -> float:
    scores, source = resolve_scores("p", "m", {"contextWindow": window, "supportsVision": False}, {}, {"models": {}})
    assert source["context"] == "derived:context_window"
    return scores["context"]


def test_context_score_grows_with_window_size():
    assert _context_score(200_000) == 0.80
    assert _context_score(400_000) == 0.88
    assert _context_score(262_144) < _context_score(1_000_000) < _context_score(2_000_000) == 1.0
    assert _context_score(1_000_000) - _context_score(200_000) > 0.15
    assert _context_score(8_000) == 0.5


def _entry(key: str) -> dict:
    return {"key": key, "available": True, "tags": ["configured"]}


def test_normalize_skips_configured_models_outside_the_allowlist():
    raw = {
        "models": [_entry("anthropic/claude-opus-5-5"), _entry("anthropic/claude-opus-5"), _entry("nvidia/m0")],
        "modelPolicyAllow": ["anthropic/claude-opus-5-5", "nvidia/*"],
    }

    configured = normalize(raw, {}, {}, {"models": {}}, configured_only=True)

    assert sorted(m["id"] for m in configured) == ["anthropic/claude-opus-5-5", "nvidia/m0"]


def test_free_models_score_full_cost_unless_overridden():
    raw = {"models": [_entry("nvidia/m0"), _entry("nvidia/m1"), _entry("anthropic/claude-opus-5-5")]}
    overrides = {"overrides": {"nvidia/m1": {"cost": 0.9}}}

    models = {m["id"]: m for m in normalize(raw, {}, overrides, {"models": {}})}

    assert models["nvidia/m0"]["scores"]["cost"] == 1.0
    assert models["nvidia/m0"]["scoreSource"]["cost"] == "derived:is_free"
    assert models["nvidia/m1"]["scores"]["cost"] == 0.9
    assert models["anthropic/claude-opus-5-5"]["scores"]["cost"] == 0.7


def test_override_can_retire_a_model_the_catalog_still_lists():
    raw = {"models": [_entry("nvidia/m0")]}
    overrides = {"overrides": {"nvidia/m0": {"available": False, "reasoning": 0.9}}}

    [model] = normalize(raw, {}, overrides, {"models": {}})

    assert model["availability"]["authed"] is False
    assert model["availability"]["available"] is False
    assert "available" not in model["scores"]


def test_review_and_summarize_derive_from_benchmarked_dimensions():
    benchmarks = {"models": {"p/m": {"_source": "artificialanalysis", "reasoning": 0.9, "coding": 0.8}}}

    scores, source = resolve_scores("p", "m", {"contextWindow": 0, "supportsVision": True}, {}, benchmarks)

    assert scores["summarize"] == 0.9
    assert scores["review"] == 0.85
    assert source["review"] == "derived:benchmark_coding_reasoning"
