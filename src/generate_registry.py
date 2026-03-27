#!/usr/bin/env python3
"""
Generate normalized model registry from:
- catalog/raw/openclaw-models.json  (openclaw models list --all --json)
- policies/families.yaml
- policies/overrides.yaml

Output: catalog/normalized/models.json
"""

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

RAW_CATALOG   = ROOT / "catalog/raw/openclaw-models.json"
FAMILIES_FILE = ROOT / "policies/families.yaml"
OVERRIDES_FILE= ROOT / "policies/overrides.yaml"
OUT_FILE      = ROOT / "catalog/normalized/models.json"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_provider(key: str) -> tuple[str, str]:
    """Return (provider, model_id) from 'provider/model-id'."""
    parts = key.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, key


def resolve_scores(provider: str, model_id: str, features: dict, families: dict) -> tuple[dict, dict]:
    """
    Resolve scoring priors for a model using:
    1. Family defaults
    2. Pattern overrides within that family

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
        "context": 0.70,
        "vision": 0.50,
        "tools": 0.70,
        "multilingual": 0.75,
    }
    source = {k: "global:default" for k in defaults}

    # Apply family defaults
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

    # Apply pattern overrides
    for pattern_entry in family_cfg.get("patterns", []):
        pattern = pattern_entry.get("match", "")
        if re.search(pattern, model_id):
            for k, v in pattern_entry.get("overrides", {}).items():
                defaults[k] = v
                source[k] = f"pattern:{pattern}"
            break  # first matching pattern wins

    return defaults, source


def normalize(raw: dict, families: dict, overrides: dict) -> list[dict]:
    models = []
    for entry in raw.get("models", []):
        key = entry.get("key", "")
        if not key:
            continue

        provider, model_id = extract_provider(key)

        features = {
            "supportsVision": "image" in entry.get("input", ""),
            "supportsTools": True,  # conservative default; update when catalog exposes it
            "supportsReasoning": any(x in model_id for x in ["reasoning", "think", "r1", "o1", "o3", "o4"]),
            "contextWindow": entry.get("contextWindow", 0),
            "inputModalities": [m for m in entry.get("input", "text").split("+") if m],
        }

        availability = {
            "authed": entry.get("available", False),
            "available": entry.get("available", False),
            "local": entry.get("local", False),
        }

        scores, score_source = resolve_scores(provider, model_id, features, families)

        # Apply manual overrides
        manual = (overrides.get("overrides") or {}).get(key, {})
        for k, v in manual.items():
            scores[k] = v
            score_source[k] = "override:manual"

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
    print(f"Loading raw catalog from {RAW_CATALOG}...")
    raw = load_json(RAW_CATALOG)

    print(f"Loading family policies from {FAMILIES_FILE}...")
    families = load_yaml(FAMILIES_FILE)

    print(f"Loading overrides from {OVERRIDES_FILE}...")
    overrides = load_yaml(OVERRIDES_FILE)

    print("Normalizing...")
    models = normalize(raw, families, overrides)

    result = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalModels": len(models),
        "models": models,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written {len(models)} models → {OUT_FILE}")

    # Print a quick summary
    authed = [m for m in models if m["availability"]["authed"]]
    print(f"  authed/available: {len(authed)}")

    # Show providers present
    providers = sorted(set(m["provider"] for m in models))
    print(f"  providers ({len(providers)}): {', '.join(providers[:8])}{'...' if len(providers) > 8 else ''}")


if __name__ == "__main__":
    main()
