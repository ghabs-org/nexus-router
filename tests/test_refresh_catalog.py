import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NEXUS_ROUTER_STATE_ROOT", str(Path(__file__).parent.parent / ".test-state"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.refresh_catalog as refresh_catalog
from scripts.refresh_catalog import check


def _models(provider: str, count: int, configured: bool = True) -> list[dict]:
    tags = ["configured"] if configured else []
    return [{"key": f"{provider}/m{i}", "available": True, "tags": tags} for i in range(count)]


def _catalog(*groups: list[dict], outcomes: dict[str, str] | None = None) -> dict:
    catalog = {"models": [m for group in groups for m in group]}
    if outcomes is not None:
        catalog["providerOutcomes"] = [{"provider": p, "status": s} for p, s in outcomes.items()]
    return catalog


def test_check_accepts_first_fetch():
    assert check(_catalog(_models("nvidia", 5)), None) == []


def test_check_refuses_empty_catalog():
    assert check({"models": []}, _catalog(_models("nvidia", 5))) == ["catalog has no models"]


def test_check_refuses_catalog_without_configured_models():
    problems = check(_catalog(_models("nvidia", 5, configured=False)), None)
    assert problems == ["no model is tagged 'configured'"]


def test_check_allows_provider_disabled_in_openclaw():
    # A disabled provider plugin vanishes with no discovery outcome (github-copilot, Oct 2026).
    old = _catalog(_models("github-copilot", 6), _models("nvidia", 10))
    new = _catalog(_models("nvidia", 10), outcomes={"nvidia": "ready"})
    assert check(new, old) == []


def test_check_refuses_provider_lost_to_failed_discovery():
    old = _catalog(_models("opencode", 10), _models("nvidia", 10))
    new = _catalog(_models("opencode", 2), _models("nvidia", 10), outcomes={"opencode": "auth-rejected", "nvidia": "ready"})
    assert check(new, old) == ["opencode dropped 10 -> 2 while discovery reported auth-rejected"]


def test_check_allows_failed_discovery_that_kept_its_models():
    old = _catalog(_models("opencode", 10))
    new = _catalog(_models("opencode", 12), outcomes={"opencode": "auth-rejected"})
    assert check(new, old) == []


def test_check_refuses_large_total_drop():
    old = _catalog(_models("nvidia", 10), _models("openrouter", 10))
    new = _catalog(_models("nvidia", 9))
    assert check(new, old) == ["total models dropped 20 -> 9"]



def test_fetch_allowlist_reads_openclaw_config(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='warning line\n["nvidia/*", "anthropic/claude-opus-5-5"]\n', stderr="")

    monkeypatch.setattr(refresh_catalog.subprocess, "run", fake_run)

    assert refresh_catalog.fetch_allowlist(10) == ["nvidia/*", "anthropic/claude-opus-5-5"]
    assert calls[0][1:] == ["config", "get", "agents.defaults.modelPolicy.allow", "--json"]


def test_fetch_allowlist_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(
        refresh_catalog.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Config path not found"),
    )

    assert refresh_catalog.fetch_allowlist(10) is None


def test_nvidia_key_round_trip():
    assert refresh_catalog.nvidia_api_id_to_openclaw_key("nvidia/nemotron-3-ultra-550b-a55b") == "nvidia/nemotron-3-ultra-550b-a55b"
    assert refresh_catalog.nvidia_api_id_to_openclaw_key("moonshotai/kimi-k3.1-pro") == "nvidia/moonshotai/kimi-k3.1-pro"
    assert refresh_catalog.openclaw_key_to_nvidia_api_id("nvidia/nemotron-3-ultra-550b-a55b") == "nvidia/nemotron-3-ultra-550b-a55b"
    assert refresh_catalog.openclaw_key_to_nvidia_api_id("nvidia/moonshotai/kimi-k3.1-pro") == "moonshotai/kimi-k3.1-pro"
    assert refresh_catalog.openclaw_key_to_nvidia_api_id("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free") is None


def test_is_nvidia_chat_model_filters_non_chat_rows():
    chat_ids = [
        "moonshotai/kimi-k3.1-pro",
        "mistralai/mistral-large-4-1t-a52b",
        "z-ai/glm-5.3-flash",
        "minimaxai/minimax-m3.1-flash",
        "stepfun-ai/step-5-preview",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "google/gemma-4-31b-it",
    ]
    non_chat_ids = [
        "nvidia/embed-qa-4",
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        "fastino/gliner2-pii",
        "nvidia/ai-synthetic-video-detector",
        "nvidia/riva-translate-4b-instruct-v2",
        "nvidia/nvclip",
        "bigcode/starcoder2-15b",
        "nvidia/cosmos-reason2-8b",
        "nvidia/ising-calibration-1.5-31b",
    ]
    assert all(refresh_catalog.is_nvidia_chat_model({"id": mid}) for mid in chat_ids)
    assert not any(refresh_catalog.is_nvidia_chat_model({"id": mid}) for mid in non_chat_ids)
    assert not refresh_catalog.is_nvidia_chat_model({"id": "custom/foo", "model_type": "embedding"})


def test_build_nvidia_config_patch_preserves_aliases_protects_refs_and_is_idempotent():
    live_rows = [
        {"id": "nvidia/nemotron-3-ultra-550b-a55b"},
        {
            "id": "moonshotai/kimi-k3.1-pro",
            "context_window": 1048576,
            "model_type": "vlm",
            "supports_thinking": True,
        },
        {"id": "nvidia/embed-qa-4"},
    ]
    existing_models = [
        {
            "id": "nvidia/nemotron-3-ultra-550b-a55b",
            "name": "Nemotron 3 Ultra 550B",
            "contextWindow": 1000000,
            "maxTokens": 8192,
        },
        {
            "id": "mistralai/mistral-large-3-675b-instruct-2512",
            "name": "Mistral Large 3",
            "contextWindow": 128000,
        },
        {
            "id": "google/gemma-4-31b-it",
            "name": "Gemma 4 31B IT",
            "contextWindow": 262144,
        },
    ]
    existing_defaults = {
        "nvidia/nemotron-3-ultra-550b-a55b": {"alias": "NVIDIA"},
        "nvidia/mistralai/mistral-large-3-675b-instruct-2512": {},
        "nvidia/google/gemma-4-31b-it": {},
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free": {},
    }
    patch, summary = refresh_catalog.build_nvidia_config_patch(
        live_rows=live_rows,
        existing_models=existing_models,
        existing_default_models=existing_defaults,
        protected_keys={"nvidia/google/gemma-4-31b-it"},
        featured_by_id={},
        old_catalog={
            "models": [
                {"key": "openrouter/moonshotai/kimi-k3.1-pro", "name": "MoonshotAI: Kimi K3.1 Pro", "contextWindow": 1048576, "input": "text+image"}
            ]
        },
    )
    assert summary["added"] == ["moonshotai/kimi-k3.1-pro"]
    assert summary["removed"] == ["mistralai/mistral-large-3-675b-instruct-2512"]
    assert patch is not None
    models_out = {m["id"]: m for m in patch["models"]["providers"]["nvidia"]["models"]}
    assert set(models_out) == {
        "nvidia/nemotron-3-ultra-550b-a55b",
        "moonshotai/kimi-k3.1-pro",
        "google/gemma-4-31b-it",
    }
    assert models_out["nvidia/nemotron-3-ultra-550b-a55b"]["contextWindow"] == 1000000
    assert models_out["moonshotai/kimi-k3.1-pro"]["name"] == "Kimi K3.1 Pro"
    assert models_out["moonshotai/kimi-k3.1-pro"]["reasoning"] is True
    assert models_out["moonshotai/kimi-k3.1-pro"]["input"] == ["text", "image"]
    assert models_out["moonshotai/kimi-k3.1-pro"]["contextWindow"] == 1048576

    defaults_patch = patch["agents"]["defaults"]["models"]
    assert defaults_patch == {
        "nvidia/moonshotai/kimi-k3.1-pro": {},
        "nvidia/mistralai/mistral-large-3-675b-instruct-2512": None,
    }
    assert "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free" not in defaults_patch

    # Idempotent when re-run against the updated state
    updated_defaults = {
        "nvidia/nemotron-3-ultra-550b-a55b": {"alias": "NVIDIA"},
        "nvidia/moonshotai/kimi-k3.1-pro": {},
        "nvidia/google/gemma-4-31b-it": {},
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free": {},
    }
    patch2, summary2 = refresh_catalog.build_nvidia_config_patch(
        live_rows=live_rows,
        existing_models=patch["models"]["providers"]["nvidia"]["models"],
        existing_default_models=updated_defaults,
        protected_keys={"nvidia/google/gemma-4-31b-it"},
        featured_by_id={},
        old_catalog=None,
    )
    assert patch2 is None
    assert summary2["added"] == [] and summary2["removed"] == []


def test_build_nvidia_config_patch_rejects_too_few_live_models():
    import pytest
    with pytest.raises(RuntimeError, match="Refusing NVIDIA sync"):
        refresh_catalog.build_nvidia_config_patch(
            live_rows=[{"id": "nvidia/embed-qa-4"}, {"id": "google/gemma-4-31b-it"}],
            existing_models=[],
            existing_default_models={},
            protected_keys=set(),
            featured_by_id={},
            old_catalog=None,
        )


def test_reconcile_nvidia_catalog_marks_absent_models_unavailable():
    catalog = {
        "models": [
            {"key": "nvidia/nemotron-3-ultra-550b-a55b", "available": True},
            {"key": "nvidia/moonshotai/kimi-k2.6", "available": True},
            {"key": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "available": True},
        ]
    }
    live, disabled = refresh_catalog.reconcile_nvidia_catalog(
        catalog, {"nvidia/nemotron-3-ultra-550b-a55b"}
    )
    assert (live, disabled) == (1, 1)
    assert catalog["models"][0]["available"] is True
    assert catalog["models"][1]["available"] is False
    assert catalog["models"][2]["available"] is True


def test_main_falls_back_cleanly_when_nvidia_api_fails(tmp_path, monkeypatch):
    raw_file = tmp_path / "openclaw-models.json"
    monkeypatch.setattr(refresh_catalog, "RAW_CATALOG_FILE", raw_file)
    monkeypatch.setattr(
        refresh_catalog,
        "fetch_nvidia_live_models",
        lambda: (_ for _ in ()).throw(OSError("network down")),
    )
    monkeypatch.setattr(
        refresh_catalog,
        "fetch_catalog",
        lambda _timeout: _catalog(_models("nvidia", 3)),
    )
    monkeypatch.setattr(refresh_catalog, "fetch_allowlist", lambda _timeout: ["nvidia/*"])
    monkeypatch.setattr(sys, "argv", ["refresh_catalog.py"])

    assert refresh_catalog.main() == 0
    written = json.loads(raw_file.read_text())
    assert len(written["models"]) == 3
    assert written["modelPolicyAllow"] == ["nvidia/*"]


def test_apply_nvidia_config_patch_passes_replace_path_and_dry_run(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(refresh_catalog.subprocess, "run", fake_run)
    patch = {"models": {"providers": {"nvidia": {"models": [{"id": "google/gemma-4-31b-it"}]}}}}
    refresh_catalog.apply_nvidia_config_patch(patch, dry_run=True, timeout=30)

    assert len(calls) == 1
    cmd, payload = calls[0]
    assert "--replace-path" in cmd
    assert "models.providers.nvidia.models" in cmd
    assert "--dry-run" in cmd
    assert json.loads(payload) == patch
