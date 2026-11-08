"""Model tariffs (policies/pricing.yaml) for outcome cost estimates.

Subscription providers have no per-call price: their marginal cost is 0 and the
estimate says so. Models without a tariff keep the old score-based proxy, marked
as such, so reports never present a guess as a price.
"""

from __future__ import annotations

import fnmatch
from typing import Optional

import yaml

from .paths import POLICIES_ROOT

PRICING_FILE = POLICIES_ROOT / "pricing.yaml"
_cache: dict = {"mtime": None, "table": {"models": {}, "subscription_providers": []}}


def pricing_table() -> dict:
    """The tariff table, re-read when the file changes; empty when there is no file."""
    try:
        mtime = PRICING_FILE.stat().st_mtime
    except OSError:
        return {"models": {}, "subscription_providers": [], "source": None}
    if _cache["mtime"] != mtime:
        with open(PRICING_FILE) as file:
            raw = yaml.safe_load(file) or {}
        models = {
            str(pattern): {
                "input_per_mtok": float(price.get("input_per_mtok", 0)),
                "output_per_mtok": float(price.get("output_per_mtok", 0)),
            }
            for pattern, price in (raw.get("models") or {}).items()
            if isinstance(price, dict)
        }
        _cache.update(
            mtime=mtime,
            table={"models": models, "subscription_providers": list(raw.get("subscription_providers") or [])},
        )
    return {**_cache["table"], "source": str(PRICING_FILE)}


def tariff(model_id: str) -> Optional[dict]:
    """Exact id first, then the most specific matching glob (longest pattern)."""
    models = pricing_table()["models"]
    if model_id in models:
        return models[model_id]
    matches = [pattern for pattern in models if fnmatch.fnmatchcase(model_id, pattern)]
    return models[max(matches, key=len)] if matches else None


def estimate(
    model_id: Optional[str], provider: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]
) -> tuple[Optional[float], Optional[str]]:
    """(usd, source) with source tariff|subscription, or (None, None) when no price applies."""
    if provider and provider in pricing_table()["subscription_providers"]:
        return 0.0, "subscription"
    price = tariff(model_id or "")
    if price is None or (input_tokens is None and output_tokens is None):
        return None, None
    usd = (int(input_tokens or 0) * price["input_per_mtok"] + int(output_tokens or 0) * price["output_per_mtok"]) / 1e6
    return round(usd, 8), "tariff"
