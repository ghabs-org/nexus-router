import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fetch_benchmarks as fb


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _http_error(url: str, code: int, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "boom", headers, io.BytesIO(b"boom"))


def test_map_public_model_name_uses_catalog_aliases():
    assert fb._map_public_model_name("Claude Sonnet 4.6 Thinking") == "github-copilot/claude-sonnet-4.6"
    assert fb._map_public_model_name("Gemini 2.5 Pro Preview 03-25") == "google-gemini-cli/gemini-2.5-pro"


def test_fetch_with_cache_falls_back_to_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)

    calls = {"count": 0}

    def succeed_once(request, timeout=30):
        calls["count"] += 1
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr(fb.urllib.request, "urlopen", succeed_once)
    first = fb._fetch_json("https://example.test/data.json")
    assert first == {"ok": True}

    def always_rate_limited(request, timeout=30):
        raise _http_error("https://example.test/data.json", 429, retry_after="0")

    monkeypatch.setattr(fb.urllib.request, "urlopen", always_rate_limited)
    monkeypatch.setattr(fb.time, "sleep", lambda *_args, **_kwargs: None)

    cached = fb._fetch_json("https://example.test/data.json")
    assert cached == {"ok": True}
    assert calls["count"] == 1


def test_fetch_with_cache_retries_on_transient_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fb.time, "sleep", lambda *_args, **_kwargs: None)

    attempts = {"count": 0}

    def flaky(request, timeout=30):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _http_error("https://example.test/retry.json", 503)
        return _FakeResponse('{"ok": true, "attempts": 3}')

    monkeypatch.setattr(fb.urllib.request, "urlopen", flaky)
    payload = fb._fetch_json("https://example.test/retry.json")

    assert payload["ok"] is True
    assert attempts["count"] == 3


def test_supported_normalizers_preserve_bounds_and_direction():
    for normalize, ceiling in (
        (fb.normalize_coding, 100.0),
        (fb.normalize_reasoning, 100.0),
        (fb.normalize_speed, 200.0),
        (fb.normalize_tools, 100.0),
        (fb.normalize_multilingual, 100.0),
    ):
        assert normalize(-1.0) == 0.0
        assert normalize(0.0) == 0.0
        assert normalize(ceiling / 2) == 0.5
        assert normalize(ceiling * 2) == 1.0
    assert fb.normalize_cost(-1.0) == 1.0
    assert fb.normalize_cost(0.0) == 1.0
    assert fb.normalize_cost(10.0) == 0.5
    assert fb.normalize_cost(40.0) == 0.0


def test_score_aa_models_reads_camelcase_fields_relative_to_the_leader():
    rows = [
        {"slug": "leader", "intelligenceIndex": 60.0, "scicode": 0.6, "terminalbenchHard": 0.4,
         "price1mInputTokens": 4, "price1mOutputTokens": 20},
        {"slug": "mid", "intelligenceIndex": 30.0, "scicode": 0.25, "tau2": 0.9},
        {"slug": "unscored", "gpqa": 0.9},
    ]

    scores = fb.score_aa_models(rows)

    # List prices do not become cost scores
    assert scores["leader"] == {"reasoning": 1.0, "coding": 1.0, "tools": 0.7222}
    # Half the leader's index lands halfway between 0.5 and 1.0
    assert scores["mid"] == {"reasoning": 0.75, "coding": 0.75, "tools": 1.0}
    assert "unscored" not in scores


def test_new_public_benchmark_identity_matches_current_catalog_without_manual_alias(monkeypatch, tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"models": [
        {"key": "openai/gpt-9.3-sol", "name": "GPT-9.3 Sol"},
        {"key": "openai/gpt-9.2-sol", "name": "GPT-9.2 Sol"},
    ]}))
    monkeypatch.setattr(fb, "MODEL_CATALOG_FILE", path)
    monkeypatch.setattr(fb, "CATALOG_ALIAS_INDEX", fb._build_catalog_alias_index())
    assert fb._map_public_model_name("GPT-9.3 Sol") == "openai/gpt-9.3-sol"
    assert fb._map_public_model_name("gpt-9-3-sol") == "openai/gpt-9.3-sol"
    assert fb._map_public_model_name("GPT-9.3 Sol (Xhigh)") == "openai/gpt-9.3-sol"
    assert fb._map_public_model_name("GPT-9.4 Sol") is None


def test_livebench_admits_new_catalog_models_without_cross_generation_aliases(monkeypatch):
    monkeypatch.setattr(fb, "CATALOG_ALIAS_INDEX", {
        "gpt-9-3-sol": ["openai/gpt-9.3-sol"],
    })
    rows = [
        {"row": {"model": "gpt-9.3-sol", "category": "reasoning", "score": 1}},
        {"row": {"model": "gpt-9.3-sol", "category": "coding", "score": 1}},
        {"row": {"model": "claude-3-7-sonnet-20250219-base", "category": "reasoning", "score": 1}},
    ]
    monkeypatch.setattr(fb, "_fetch_with_cache", lambda *args, **kwargs: {"rows": rows})
    result = fb.fetch_livebench()
    assert result == {"openai/gpt-9.3-sol": {
        "_source": "livebench", "_updated_at": fb.TODAY, "reasoning": 1.0, "coding": 1.0,
    }}


def test_nested_aa_release_schema_and_effort_variants_are_order_independent():
    rows = [
        {"release": {"slug": "future"}, "name": "Future (Max)", "intelligenceIndex": 60,
         "scores": {"terminalbench-4-0": .8, "frontiercode-2-0": .6}},
        {"release": {"slug": "future"}, "name": "Future (Medium)", "intelligenceIndex": 30,
         "scores": {"terminalbench-4-0": .9, "frontiercode-2-0": .9}},
        {"release": {"slug": "older"}, "intelligenceIndex": 40,
         "scores": {"terminalbench-hard": .5, "scicode": .5, "tau2-bench": .5}},
    ]
    scores = fb.score_aa_models(rows)
    assert scores == fb.score_aa_models(list(reversed(rows)))
    assert scores["future"] == {"reasoning": 1.0, "coding": 1.0, "tools": 1.0}
    assert scores["older"]["reasoning"] == .8333
    # Future's coherent max variant has coding .7, not the medium variant's .9.
    assert scores["older"]["coding"] == .8571


def test_aa_nested_feed_maps_a_novel_catalog_model(monkeypatch):
    monkeypatch.setattr(fb, "CATALOG_ALIAS_INDEX", {"gpt-9-3-sol": ["openai/gpt-9.3-sol"]})
    rows = [{"release": {"slug": "gpt-9-3-sol"}, "intelligenceIndex": 60,
             "scores": {"frontiercode-2-0": .6, "terminalbench-4-0": .8}}]
    raw = json.dumps(rows)
    for payload in (raw, json.dumps(raw)):
        monkeypatch.setattr(fb, "_fetch_with_cache", lambda *args, **kwargs: payload)
        result = fb.fetch_artificialanalysis()
        assert result["openai/gpt-9.3-sol"]["reasoning"] == 1.0
        assert result["openai/gpt-9.3-sol"]["_source"] == "artificialanalysis"


def test_exact_benchmark_upgrade_survives_densification(monkeypatch):
    monkeypatch.setattr(fb, "_load_catalog_models", lambda: ["openai/future", "other/future"])
    synthetic = {"reasoning": .6, "_source": "inherited:other/future"}
    actual = {"reasoning": .99, "_source": "artificialanalysis"}
    for force in (False, True):
        upgraded = fb.merge_scores(synthetic, actual, force=force)
        dense = fb.densify_benchmark_scores({
            "openai/future": upgraded, "other/future": {"reasoning": .6, "_source": "livebench"},
        })
        assert dense["openai/future"]["reasoning"] == .99
        assert dense["openai/future"]["_source"] == "artificialanalysis"


def test_extra_source_coverage_cannot_reverse_the_shared_primary_benchmark():
    first = fb.aggregate_source_entries([{"_source": "artificialanalysis", "reasoning": .9}])
    second = fb.aggregate_source_entries([
        {"_source": "artificialanalysis", "reasoning": .95},
        {"_source": "livebench", "reasoning": .8},
    ])
    assert first["reasoning"] == .9
    assert second["reasoning"] == .95
    assert second["_metadata"]["dimension_sources"]["reasoning"] == ["artificialanalysis"]


def test_dimension_publisher_metadata_survives_refresh_and_registry_generation():
    from src.generate_registry import resolve_scores
    measured = fb.aggregate_source_entries([
        {"_source": "artificialanalysis", "reasoning": .95},
        {"_source": "livebench", "coding": .8},
    ])
    merged = fb.merge_scores({}, measured, force=True)
    scores, sources = resolve_scores("p", "future", {}, {}, {"models": {"p/future": merged}})
    assert scores["reasoning"] == .95
    assert sources["reasoning"] == "benchmark:artificialanalysis"
    assert sources["coding"] == "benchmark:livebench"


def test_benchmark_metadata_round_trips_through_isolated_database(monkeypatch, tmp_path):
    from src import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "benchmarks.sqlite")
    db.ensure_schema()
    measured = fb.aggregate_source_entries([{"_source": "artificialanalysis", "reasoning": .95}])
    db.replace_benchmark_scores({"p/future": measured})
    loaded = db.load_benchmark_scores()["p/future"]
    assert loaded["_metadata"] == measured["_metadata"]
    assert loaded["reasoning"] == .95


def test_catalog_alias_index_maps_vendor_prefixed_nvidia_models(tmp_path, monkeypatch):
    catalog = tmp_path / "openclaw-models.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {"key": "nvidia/moonshotai/kimi-k3.1-pro", "name": "Kimi K3.1 Pro"},
                    {"key": "nvidia/z-ai/glm-5.3-flash", "name": "GLM 5.3 Flash"},
                ]
            }
        )
    )
    monkeypatch.setattr(fb, "MODEL_CATALOG_FILE", catalog)
    monkeypatch.setattr(fb, "CATALOG_ALIAS_INDEX", fb._build_catalog_alias_index())

    assert fb._map_public_model_name("kimi-k3-1-pro") == "nvidia/moonshotai/kimi-k3.1-pro"
    assert fb._map_public_model_name("glm-5-3-flash") == "nvidia/z-ai/glm-5.3-flash"
