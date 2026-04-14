#!/usr/bin/env python3
"""
Generate normalized model registry from:
- generated/openclaw-models.json  (openclaw models list --all --json)
- policies/families.yaml
- policies/overrides.yaml
- router.sqlite benchmark_model_scores table

Output: generated/models.json

Flags:
  --only-configured       Include only models whose provider has auth=true
  --providers A,B,C       Include only models from listed providers
  --all                   Include all 800+ catalog models (default without flags)
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent.parent

try:
    from .paths import RAW_CATALOG_FILE as RAW_CATALOG, POLICIES_ROOT, REGISTRY_FILE as OUT_FILE, ensure_parent
    from .db import load_benchmark_scores, load_outcome_counts
except ImportError:
    from paths import RAW_CATALOG_FILE as RAW_CATALOG, POLICIES_ROOT, REGISTRY_FILE as OUT_FILE, ensure_parent
    from db import load_benchmark_scores, load_outcome_counts

FAMILIES_FILE    = POLICIES_ROOT / "families.yaml"
OVERRIDES_FILE   = POLICIES_ROOT / "overrides.yaml"

COPILOT_INCLUDED_FREE_MODELS = {
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4o",
}


def resolve_is_free(provider: str, model_id: str, key: str, manual: dict | None) -> bool | None:
    if isinstance(manual, dict) and "is_free" in manual:
        value = manual.get("is_free")
        if value is None:
            return None
        return bool(value)

    normalized_key = str(key or "").strip().lower()
    normalized_model_id = str(model_id or "").strip().lower()

    if provider == "nvidia":
        return True
    if provider == "openrouter" and ":free" in normalized_key:
        return True
    if provider == "github-copilot" and normalized_model_id in COPILOT_INCLUDED_FREE_MODELS:
        return True
    return None


def load_json(path: Path) -> dict:
    raw = path.read_text()
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    obj_start = raw.find("{")
    if obj_start != -1:
        return json.loads(raw[obj_start:])
    arr_start = raw.find("[")
    if arr_start != -1:
        return json.loads(raw[arr_start:])
    raise ValueError(f"No JSON object found in {path}")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_provider(key: str) -> tuple[str, str]:
    """Return (provider, model_id) from 'provider/model-id'."""
    parts = key.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, key


def _benchmark_entry_is_usable(entry: dict) -> bool:
    if not entry:
        return False
    if not any(k in entry for k in ("coding", "review", "reasoning", "summarize", "fast", "cost", "eco", "context", "vision", "tools", "multilingual")):
        return False
    source = str(entry.get("_source") or "").strip()
    if not source or source == "catalog-only":
        return False
    return True


def _canonical_benchmark_keys(provider: str, model_id: str, bench_models: dict) -> list[str]:
    """Return candidate benchmark keys, preferring exact usable match then any same-model sibling across providers."""
    exact = f"{provider}/{model_id}"
    candidates = []

    exact_entry = bench_models.get(exact, {})
    if _benchmark_entry_is_usable(exact_entry):
        candidates.append(exact)

    # Generic provider-agnostic fallback: if another provider exposes the same
    # underlying model_id and benchmark data exists for it, reuse those capability
    # scores here. Provider-specific delivery differences are handled elsewhere.
    sibling_suffix = f"/{model_id}"
    siblings = sorted(
        key for key, entry in bench_models.items()
        if key != exact and key.endswith(sibling_suffix) and _benchmark_entry_is_usable(entry) and "inherited:" not in str(entry.get("_source") or "")
    )
    candidates.extend(siblings)
    return candidates


# Cold-start graduation threshold: once a model has at least this many recorded
# routing outcomes, its family priors are skipped entirely in favour of
# data-driven benchmark/feedback scores. Set via env var NEXUS_COLD_START_MIN_OUTCOMES.
_COLD_START_MIN_OUTCOMES = int(
    __import__("os").environ.get("NEXUS_COLD_START_MIN_OUTCOMES", "30")
)


def resolve_scores(
    provider: str,
    model_id: str,
    features: dict,
    families: dict,
    benchmarks: dict,
    outcome_counts: dict[str, int] | None = None,
) -> tuple[dict, dict]:
    """
    Resolve scoring priors for a model using:
    1. Family defaults (cold-start only — skipped once model has >= NEXUS_COLD_START_MIN_OUTCOMES outcomes)
    2. Pattern overrides within that family (cold-start only)
    3. Derived scores from catalog features (context window, vision modality) — always applied
    4. Benchmark-derived scores from SQLite — always applied, highest priority

    Returns (scores, source_map)
    """
    family_cfg = families.get("families", {}).get(provider, {})

    defaults = {
        "coding": 0.70,
        "review": 0.70,
        "reasoning": 0.70,
        "summarize": 0.70,
        "fast": 0.70,
        "cost": 0.70,
        "eco": 0.70,
        "context": 0.70,
        "vision": 0.50,
        "tools": 0.70,
        "multilingual": 0.75,
    }
    source = {k: "global:default" for k in defaults}

    # Determine whether this model has "graduated" from cold-start priors.
    # A graduated model has enough real outcomes that family priors should not influence it.
    model_key = f"{provider}/{model_id}"
    n_outcomes = (outcome_counts or {}).get(model_key, 0)
    graduated = n_outcomes >= _COLD_START_MIN_OUTCOMES

    if graduated:
        # Skip family defaults and pattern overrides entirely.
        # scoreSource will reflect "graduated:N" so the registry shows which models
        # are fully data-driven vs still on cold-start priors.
        for k in defaults:
            source[k] = f"graduated:{n_outcomes}_outcomes"
    else:
        # Apply family defaults (cold-start prior)
        for k, v in family_cfg.get("default", {}).items():
            defaults[k] = v
            source[k] = "family:default"

    # Apply context score from context window size
    ctx = features.get("contextWindow", 0)
    if ctx >= 900_000:
        defaults["context"] = 0.96
        source["context"] = "derived:context_window"
    elif ctx >= 200_000:
        defaults["context"] = 0.90
        source["context"] = "derived:context_window"
    elif ctx >= 100_000:
        defaults["context"] = 0.80
        source["context"] = "derived:context_window"
    elif ctx > 0:
        defaults["context"] = 0.65
        source["context"] = "derived:context_window"

    # Apply vision score from modalities
    if not features.get("supportsVision", False):
        defaults["vision"] = 0.0
        source["vision"] = "derived:modality"

    # Apply pattern overrides (cold-start only — skipped for graduated models)
    if not graduated:
        for pattern_entry in family_cfg.get("patterns", []):
            pattern = pattern_entry.get("match", "")
            if re.search(pattern, model_id):
                for k, v in pattern_entry.get("overrides", {}).items():
                    defaults[k] = v
                    source[k] = f"pattern:{pattern}"
                break  # first matching pattern wins

    # Apply benchmark-derived scores (highest-priority source, overrides family)
    bench_models = (benchmarks.get("models") or {})
    bench_entry = {}
    bench_key_used = None
    for candidate_key in _canonical_benchmark_keys(provider, model_id, bench_models):
        candidate = bench_models.get(candidate_key, {})
        if candidate:
            bench_entry = candidate
            bench_key_used = candidate_key
            break

    for k, v in bench_entry.items():
        if k in defaults:
            defaults[k] = v
            source_name = bench_entry.get('_source', 'unknown')
            if bench_key_used == f"{provider}/{model_id}":
                source[k] = f"benchmark:{source_name}"
            else:
                source[k] = f"benchmark-inherited:{bench_key_used}:{source_name}"

    return defaults, source


def normalize(
    raw: dict,
    families: dict,
    overrides: dict,
    benchmarks: dict,
    only_providers: set[str] | None = None,
    outcome_counts: dict[str, int] | None = None,
) -> list[dict]:
    models = []
    for entry in raw.get("models", []):
        key = entry.get("key", "")
        if not key:
            continue

        provider, model_id = extract_provider(key)

        # Provider filter
        if only_providers and provider not in only_providers:
            continue

        features = {
            "supportsVision": "image" in entry.get("input", ""),
            "supportsTools": True,  # conservative default; update when catalog exposes it
            "contextWindow": entry.get("contextWindow", 0),
            "inputModalities": [m for m in entry.get("input", "text").split("+") if m],
        }

        availability = {
            "authed": entry.get("available", False),
            "available": entry.get("available", False),
            "local": entry.get("local", False),
        }

        scores, score_source = resolve_scores(provider, model_id, features, families, benchmarks, outcome_counts=outcome_counts)

        # Apply manual overrides
        manual = (overrides.get("overrides") or {}).get(key, {})
        for k, v in manual.items():
            if k == "is_free":
                continue
            scores[k] = v
            score_source[k] = "override:manual"

        features["is_free"] = resolve_is_free(provider, model_id, key, manual)

        models.append({
            "id": key,
            "provider": provider,
            "modelId": model_id,
            "name": entry.get("name", model_id),
            "features": features,
            "availability": availability,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "scoreSource": score_source,
        })

    return models


def main():
    parser = argparse.ArgumentParser(description="Generate Nexus Router model registry")
    parser.add_argument("--only-configured", action="store_true",
                        help="Include only models whose provider has auth=true (available=true in catalog)")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated provider ids to include, e.g. openai-codex,github-copilot")
    parser.add_argument("--all", dest="include_all", action="store_true",
                        help="Include all 800+ catalog models (default without filters)")
    parser.add_argument("--purge-cold-start", action="store_true",
                        help=(
                            f"Skip family priors for models with >= {_COLD_START_MIN_OUTCOMES} outcomes "
                            "(always enabled during normal generation; this flag just makes it explicit "
                            "and prints a graduation report)."
                        ))
    args = parser.parse_args()

    print(f"Loading raw catalog from {RAW_CATALOG}...")
    raw = load_json(RAW_CATALOG)

    print(f"Loading family policies from {FAMILIES_FILE}...")
    families = load_yaml(FAMILIES_FILE)

    print(f"Loading overrides from {OVERRIDES_FILE}...")
    overrides = load_yaml(OVERRIDES_FILE)

    benchmarks = {"models": load_benchmark_scores()}
    n_bench = len((benchmarks.get("models") or {}))
    if n_bench:
        print("Loading benchmark scores from router DB...")
        print(f"  {n_bench} benchmark model entries loaded")
    else:
        print("No benchmark scores found in router DB — using family priors only")

    # Load outcome counts for cold-start graduation
    outcome_counts = load_outcome_counts()
    graduated = [m for m, n in outcome_counts.items() if n >= _COLD_START_MIN_OUTCOMES]
    cold_start = [m for m, n in outcome_counts.items() if n < _COLD_START_MIN_OUTCOMES]
    print(f"Cold-start graduation threshold: {_COLD_START_MIN_OUTCOMES} outcomes")
    print(f"  Graduated (data-driven, family priors ignored): {len(graduated)}")
    print(f"  Still on cold-start priors (< threshold): {len(cold_start)}")
    if args.purge_cold_start and graduated:
        print("  Graduated models (family priors suppressed):")
        for m in sorted(graduated):
            print(f"    {m} ({outcome_counts[m]} outcomes)")

    # Resolve provider filter
    only_providers: set[str] | None = None

    if args.providers:
        only_providers = {p.strip() for p in args.providers.split(",")}
        print(f"Provider filter: {sorted(only_providers)}")

    elif args.only_configured:
        # Collect providers that have at least one authed model
        all_models = raw.get("models", [])
        authed_providers = {
            extract_provider(m.get("key", ""))[0]
            for m in all_models
            if m.get("available", False)
        }
        only_providers = authed_providers
        print(f"Only-configured filter: {sorted(only_providers)}")

    print("Normalizing...")
    models = normalize(raw, families, overrides, benchmarks, only_providers, outcome_counts=outcome_counts)

    result = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalModels": len(models),
        "filter": {
            "onlyConfigured": args.only_configured,
            "providers": sorted(only_providers) if only_providers else None,
        },
        "models": models,
    }

    ensure_parent(OUT_FILE)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written {len(models)} models → {OUT_FILE}")

    authed = [m for m in models if m["availability"]["authed"]]
    bench_count = sum(
        1 for m in models
        if any(v.startswith("benchmark:") for v in m["scoreSource"].values())
    )
    graduated_count = sum(
        1 for m in models
        if any(v.startswith("graduated:") for v in m["scoreSource"].values())
    )
    cold_start_count = sum(
        1 for m in models
        if any(v.startswith("family:") or v.startswith("pattern:") for v in m["scoreSource"].values())
    )
    print(f"  authed/available: {len(authed)}")
    print(f"  benchmark-scored: {bench_count}")
    print(f"  graduated (data-driven, priors ignored): {graduated_count}")
    print(f"  cold-start (family priors active): {cold_start_count}")
    providers = sorted(set(m["provider"] for m in models))
    print(f"  providers ({len(providers)}): {', '.join(providers[:8])}{'...' if len(providers) > 8 else ''}")


if __name__ == "__main__":
    main()
