#!/usr/bin/env python3
"""
fetch_benchmarks.py — Fetch benchmark data and write benchmark scores into router SQLite.

Sources:
  - artificialanalysis.ai  → speed, cost, context, quality proxies
  - livebench.ai           → reasoning, coding
  - HELM                   → reasoning / coding proxies when available
  - Inspect AI             → instruction-following / reasoning proxies when public results are available
  - Open LLM Leaderboard   → reasoning / review proxies for open models when public rows are available
  - vellum.ai              → multi-task quality
  - BFCL                   → tool use
  - MGSM                   → multilingual

This script:
1. Fetches structured data from each source
2. Normalizes scores to 0.0–1.0 per router dimension
3. Merges per-dimension contributions across sources with a weighted average:
      final_dim = sum(weight(source) * value(source)) / sum(weight(source))
   Only sources that actually provide a value for a dimension participate in
   that dimension's denominator.
4. Writes/updates benchmark_model_scores in router.sqlite

Default source trust weights are defined in SOURCE_WEIGHTS below. Unknown
sources fall back to 1.0.

Usage:
  python src/fetch_benchmarks.py                    # fetch all configured sources
  python src/fetch_benchmarks.py --sources aa,lb    # only artificialanalysis + livebench
  python src/fetch_benchmarks.py --sources aa,helm  # include HELM if available
  python src/fetch_benchmarks.py --sources aa,inspect,openllm
  python src/fetch_benchmarks.py --dry-run          # print without writing
  python src/fetch_benchmarks.py --force            # overwrite existing scores
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("Missing: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT           = Path(__file__).parent.parent
try:
    from .paths import POLICIES_ROOT, BENCHMARK_CACHE_DIR as CACHE_DIR, RAW_CATALOG_FILE as MODEL_CATALOG_FILE, ensure_dir
    from .db import load_benchmark_scores, replace_benchmark_scores
except ImportError:
    from paths import POLICIES_ROOT, BENCHMARK_CACHE_DIR as CACHE_DIR, RAW_CATALOG_FILE as MODEL_CATALOG_FILE, ensure_dir
    from db import load_benchmark_scores, replace_benchmark_scores
SOURCES_FILE   = POLICIES_ROOT / "benchmark-sources.yaml"

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

SOURCE_WEIGHTS: dict[str, float] = {
    "artificialanalysis": 1.0,
    "livebench": 1.0,
    "helm": 0.9,
    "inspect_ai": 0.9,
    "openllm": 0.9,
    "vellum": 0.9,
    "bfcl": 1.0,
    "mgsm": 1.0,
    "hf_leaderboard": 0.8,
}

FETCH_RETRY_DELAYS = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_SECONDS = 15.0
PREFERRED_MODEL_PROVIDERS = (
    "google-gemini-cli/",
    "github-copilot/",
    "openai-codex/",
)

# ── Normalization helpers ─────────────────────────────────────────────────────

def normalize_coding(raw_score: float, max_score: float = 100.0) -> float:
    """HumanEval pass@1 or LiveCodeBench: divide by 100."""
    return round(max(0.0, min(1.0, raw_score / max_score)), 4)


def normalize_reasoning(raw_score: float, max_score: float = 100.0) -> float:
    """LiveBench reasoning composite: divide by 100."""
    return round(max(0.0, min(1.0, raw_score / max_score)), 4)


def normalize_speed(tokens_per_second: float, ceiling: float = 200.0) -> float:
    """
    Tokens/second from artificialanalysis.ai.
    200 tok/s → 1.0 (fast frontier).
    Capped at 1.0.
    """
    return round(max(0.0, min(1.0, tokens_per_second / ceiling)), 4)


def normalize_cost(avg_cost_per_million: float, ceiling: float = 20.0) -> float:
    """
    $/M tokens (blended input+output).
    $0/M → 1.0 (free), $20/M → 0.0 (expensive frontier).
    Inverted: cheaper = higher score.
    """
    return round(max(0.0, min(1.0, 1.0 - avg_cost_per_million / ceiling)), 4)


def normalize_tools(bfcl_accuracy: float, max_score: float = 100.0) -> float:
    """BFCL overall accuracy: divide by 100."""
    return round(max(0.0, min(1.0, bfcl_accuracy / max_score)), 4)


def normalize_multilingual(mgsm_score: float, max_score: float = 100.0) -> float:
    """MGSM or similar multilingual benchmark: divide by 100."""
    return round(max(0.0, min(1.0, mgsm_score / max_score)), 4)


# ── Source fetchers (stubs — implement as data access becomes available) ──────

def fetch_artificialanalysis() -> dict[str, dict]:
    """
    Fetch speed, cost, and quality data from artificialanalysis.ai.

    Uses the RSC (React Server Components) payload embedded in the page response.
    The data is a JSON array starting at a predictable offset in the payload.

    Fields extracted:
      coding_index       → coding
      intelligence_index → reasoning
      agentic_index      → proxy for tools
      gpqa               → proxy for reasoning (cross-check)
      livecodebench      → proxy for coding (cross-check)
      price_1m_blended   → cost (inverted, normalized)
      multilingual_aa.average.score → multilingual
      context_window_tokens → context (derived)

    The slug field is used for model name mapping.
    """
    import re as _re

    PAGE_URL = "https://artificialanalysis.ai/leaderboards/models"
    ARRAY_MARKER = '"critpt"'

    try:
        raw = _fetch_with_cache(
            PAGE_URL,
            cache_key="text:artificialanalysis:leaderboard",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            loader=lambda s: s,
            suffix=".txt",
        )
    except Exception as e:
        print(f"    artificialanalysis fetch error: {e}")
        return {}

    # The data is JSON-escaped inside a Next.js HTML payload
    # Unescape it first so we can parse the JSON normally
    if '\\"critpt\\"' in raw:
        # Find the outer string boundary and unescape it
        idx = raw.find('\\"critpt\\"')
        # Find the start of the enclosing string — look for [{ pattern escaped
        start = raw.rfind('[{\\"', 0, idx)
        if start != -1:
            # Find the end: look for the closing ] of the array
            # We'll extract a large slice and unescape it
            end_hint = raw.find('\\"]', idx)
            slice_raw = raw[start:end_hint + 20000] if end_hint != -1 else raw[start:start + 2_000_000]
            # Unescape: replace \" with " and \\ with \
            unescaped = slice_raw.replace('\\"', '"').replace('\\\\', '\\')
            raw = unescaped  # continue with unescaped data
            ARRAY_MARKER = '"critpt"'
        else:
            print("    artificialanalysis: could not find array start in escaped data")
            return {}

    # Find the array of model objects
    idx = raw.find(ARRAY_MARKER)
    if idx == -1:
        print("    artificialanalysis: marker not found in page")
        return {}

    # Walk back to find the opening [ of the array
    start = raw.rfind('[{', 0, idx)
    if start == -1:
        print("    artificialanalysis: could not find array start")
        return {}

    # Walk forward to find the end of the array
    partial = raw[start:]
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i, c in enumerate(partial):
        if escape:
            escape = False
            continue
        if c == '\\' and in_str:
            escape = True
            continue
        if c == '"' and not escape:
            in_str = not in_str
            continue
        if not in_str:
            if c in ('[', '{'):
                depth += 1
            elif c in (']', '}'):
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end == -1:
        print("    artificialanalysis: could not find array end")
        return {}

    try:
        models_raw = json.loads(partial[:end + 1])
    except Exception as e:
        print(f"    artificialanalysis: JSON parse error: {e}")
        return {}

    print(f"    {len(models_raw)} models found in AA page")

    # Slug → OpenClaw model ID mapping
    # AA slugs use dashes, OpenClaw uses provider/model-id format
    SLUG_MAP: dict[str, str] = {
        # openai-codex
        "gpt-5-4":           "openai-codex/gpt-5.4",
        "gpt-5-3-codex":     "openai-codex/gpt-5.3-codex",
        "gpt-5-2-codex":     "openai-codex/gpt-5.2-codex",
        "gpt-5-4-mini":      "openai-codex/gpt-5.4-mini",
        # github-copilot (claude)
        "claude-sonnet-4-6":               "github-copilot/claude-sonnet-4.6",
        "claude-opus-4-6":                 "github-copilot/claude-opus-4.6",
        "claude-opus-4-5":                 "github-copilot/claude-opus-4.5",
        "claude-opus-4-5-thinking":        "github-copilot/claude-opus-4.5",
        "claude-sonnet-4-6-adaptive":      "github-copilot/claude-sonnet-4.6",
        "claude-opus-4-6-adaptive":        "github-copilot/claude-opus-4.6",
        # github-copilot (gemini)
        "gemini-3-1-pro-preview":          "github-copilot/gemini-3.1-pro-preview",
        # google-gemini-cli
        "gemini-3-1-pro-preview":          "google-gemini-cli/gemini-3.1-pro-preview",
        "gemini-2-5-pro":                  "google-gemini-cli/gemini-2.5-pro",
        "gemini-2-5-pro-03-25":            "google-gemini-cli/gemini-2.5-pro",
        "gemini-2-5-flash-reasoning":      "google-gemini-cli/gemini-2.5-flash",
        "gemini-2-5-flash-04-2025":        "google-gemini-cli/gemini-2.5-flash",
        "gemini-2-0-flash":                "google-gemini-cli/gemini-2.0-flash",
    }

    # Normalisation helpers (local to this function to avoid name collisions)
    def _norm(val: float, ceiling: float, invert: bool = False) -> float:
        if val is None:
            return None
        v = max(0.0, min(1.0, val / ceiling))
        return round(1.0 - v if invert else v, 4)

    MAX_INTELLIGENCE = 70.0  # approximate max observed intelligence_index
    MAX_CODING       = 70.0
    MAX_AGENTIC      = 80.0
    MAX_BLENDED_COST = 15.0  # $/M — above this = cost score 0.0

    result: dict[str, dict] = {}
    for m in models_raw:
        slug = m.get("slug", "")
        oc_model = SLUG_MAP.get(slug) or _map_public_model_name(slug.replace("-", " ")) or _map_public_model_name(slug)
        if not oc_model:
            continue

        entry: dict = {"_source": "artificialanalysis", "_updated_at": TODAY}

        # Coding
        ci = m.get("coding_index")
        if ci is not None:
            entry["coding"] = _norm(ci, MAX_CODING)

        # Reasoning (intelligence_index + gpqa cross-check)
        ii = m.get("intelligence_index")
        gpqa = m.get("gpqa")
        if ii is not None:
            reasoning = _norm(ii, MAX_INTELLIGENCE)
            if gpqa is not None:
                reasoning = round((reasoning + float(gpqa)) / 2, 4)
            entry["reasoning"] = reasoning
        elif gpqa is not None:
            entry["reasoning"] = round(float(gpqa), 4)

        # Tools (agentic_index)
        ai = m.get("agentic_index")
        if ai is not None:
            entry["tools"] = _norm(ai, MAX_AGENTIC)

        # Cost (blended price, inverted)
        price = m.get("price_1m_blended_3_to_1")
        if price is not None and price > 0:
            entry["cost"] = _norm(price, MAX_BLENDED_COST, invert=True)

        # Multilingual
        multi = m.get("multilingual_aa")
        if multi and isinstance(multi, dict):
            avg = multi.get("average", {})
            ml_score = avg.get("score")
            if ml_score is not None:
                entry["multilingual"] = round(float(ml_score), 4)

        # Coding cross-check with livecodebench
        lcb = m.get("livecodebench")
        if lcb is not None and "coding" in entry:
            entry["coding"] = round((entry["coding"] + float(lcb)) / 2, 4)
        elif lcb is not None:
            entry["coding"] = round(float(lcb), 4)

        if len(entry) > 2:
            result[oc_model] = entry

    # gemini-3.1-pro-preview should be in both providers
    g31 = result.get("google-gemini-cli/gemini-3.1-pro-preview")
    if g31:
        result["github-copilot/gemini-3.1-pro-preview"] = dict(g31)

    print(f"    {len(result)} models mapped to OpenClaw IDs")
    return result


def fetch_livebench() -> dict[str, dict]:
    """
    Fetch reasoning and coding scores from LiveBench via HuggingFace datasets API.

    Data: https://huggingface.co/datasets/livebench/model_judgment
    60k+ question-level rows (score=0/1 per question). We fetch all and aggregate.

    Categories:
      reasoning  → task category "reasoning"
      coding     → task category "coding"
      language   → proxy for review / instruction following

    Returns dict of OpenClaw provider/model ID → router dimension scores.
    """
    BASE = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=livebench%2Fmodel_judgment"
        "&config=default&split=leaderboard"
    )
    PAGE = 100  # smaller pages; HF datasets-server appears to reject deeper/larger pagination intermittently
    MAX_ROWS = 5_000

    # Aggregate: {model -> {category -> (sum, count)}}
    agg: dict[str, dict[str, list]] = {}

    offset = 0
    total_fetched = 0
    try:
        while offset < MAX_ROWS:
            url = f"{BASE}&offset={offset}&length={PAGE}"
            try:
                data = _fetch_with_cache(
                    url,
                    cache_key=f"json:livebench:{offset}",
                    timeout=30,
                    headers={
                        "User-Agent": "nexus-router/1.0",
                        "Accept": "application/json",
                    },
                    loader=lambda s: json.loads(s),
                    suffix=".json",
                )
            except Exception as e:
                print(f"    LiveBench page fetch stopped at offset={offset}: {e}")
                break

            rows = data.get("rows", [])
            if not rows:
                break

            for row in rows:
                r = row.get("row", {})
                model    = r.get("model", "")
                category = r.get("category", "")
                score    = r.get("score")
                if not model or not category or score is None:
                    continue
                agg.setdefault(model, {}).setdefault(category, []).append(float(score))
                total_fetched += 1

            offset += PAGE
            if len(rows) < PAGE:
                break  # last page

    except Exception as e:
        print(f"    LiveBench fetch error: {e}")
        if not agg:
            return {}

    # Compute per-model per-category averages (×100 for %)
    raw_scores: dict[str, dict[str, float]] = {}
    for model, cats in agg.items():
        raw_scores[model] = {
            cat: (sum(v) / len(v)) * 100
            for cat, v in cats.items()
        }

    print(f"    {total_fetched} rows fetched, {len(raw_scores)} LiveBench models aggregated")

    # Map LiveBench model names → OpenClaw provider/model IDs
    MODEL_MAP = {
        "claude-3-5-sonnet-20241022":          "github-copilot/claude-sonnet-4",
        "claude-3-7-sonnet-20250219-base":     "github-copilot/claude-sonnet-4.5",
        "claude-3-opus-20240229":              "github-copilot/claude-opus-4.5",
        "gemini-2.0-flash":                    "google-gemini-cli/gemini-2.0-flash",
        "gemini-2.0-flash-001":                "google-gemini-cli/gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25":            "google-gemini-cli/gemini-2.5-pro",
        "gemini-2.5-flash":                    "google-gemini-cli/gemini-2.5-flash",
        "gpt-4o-2024-11-20":                   "github-copilot/gpt-4o",
    }

    result: dict[str, dict] = {}
    for lb_model, scores in raw_scores.items():
        oc_model = MODEL_MAP.get(lb_model)
        if not oc_model:
            continue

        entry: dict = {"_source": "livebench", "_updated_at": TODAY}

        if "reasoning" in scores:
            entry["reasoning"] = normalize_reasoning(scores["reasoning"])
        if "coding" in scores:
            entry["coding"] = normalize_coding(scores["coding"])
        if "language" in scores:
            entry["review"]    = normalize_coding(scores["language"])
            entry["summarize"] = normalize_coding(scores["language"] * 0.95)
        if "math" in scores:
            # math sub-score also proxies reasoning quality
            existing = entry.get("reasoning", 0.0)
            entry["reasoning"] = round((existing + normalize_reasoning(scores["math"])) / 2, 4)

        if len(entry) > 2:
            result[oc_model] = entry

    print(f"    {len(result)} models mapped to OpenClaw IDs")
    return result


def _cache_file_path(cache_key: str, suffix: str = ".json") -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{digest}{suffix}"


def _load_cached_payload(cache_key: str, loader, suffix: str):
    path = _cache_file_path(cache_key, suffix)
    if not path.exists():
        return None
    try:
        return loader(path.read_text())
    except Exception as e:
        print(f"    cache read failed for {cache_key}: {e}")
        return None


def _write_cached_payload(cache_key: str, raw_text: str, suffix: str):
    path = _cache_file_path(cache_key, suffix)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_text)
    except Exception as e:
        print(f"    cache write failed for {cache_key}: {e}")


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = str(value).strip()
    try:
        return max(0.0, min(MAX_RETRY_AFTER_SECONDS, float(value)))
    except ValueError:
        return None


def _fetch_with_cache(url: str, *, cache_key: Optional[str] = None, timeout: int = 30,
                      headers: Optional[dict[str, str]] = None,
                      loader=lambda s: s, suffix: str = ".txt"):
    cache_key = cache_key or url
    last_error = None
    req_headers = headers or {"User-Agent": "Mozilla/5.0"}

    for attempt, base_delay in enumerate((0.0,) + FETCH_RETRY_DELAYS, start=1):
        if attempt > 1:
            time.sleep(base_delay)
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
            payload = loader(raw_text)
            _write_cached_payload(cache_key, raw_text, suffix)
            return payload
        except urllib.error.HTTPError as e:
            last_error = e
            retry_after = _parse_retry_after(getattr(e, "headers", {}).get("Retry-After"))
            status = getattr(e, "code", None)
            if status == 429:
                cached = _load_cached_payload(cache_key, loader=loader, suffix=suffix)
                if cached is not None:
                    print(f"    using stale cache for {url} after 429 rate limit")
                    return cached
                if retry_after is not None and attempt <= len(FETCH_RETRY_DELAYS):
                    print(f"    rate limited for {url} (429); retrying after {retry_after:.1f}s")
                    time.sleep(retry_after)
                    continue
            if status not in (429, 500, 502, 503, 504) or attempt > len(FETCH_RETRY_DELAYS):
                break
            print(f"    transient HTTP {status} for {url}; retry {attempt}/{len(FETCH_RETRY_DELAYS)}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt > len(FETCH_RETRY_DELAYS):
                break
            print(f"    transient fetch error for {url}: {e} (retry {attempt}/{len(FETCH_RETRY_DELAYS)})")

    cached = _load_cached_payload(cache_key, loader=loader, suffix=suffix)
    if cached is not None:
        print(f"    using stale cache for {url} after fetch failure: {last_error}")
        return cached
    raise last_error


def _fetch_json(url: str, timeout: int = 30):
    return _fetch_with_cache(
        url,
        cache_key=f"json:{url}",
        timeout=timeout,
        headers={
            "User-Agent": "nexus-router/1.0",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        },
        loader=lambda s: json.loads(s),
        suffix=".json",
    )


def _fetch_text(url: str, timeout: int = 30) -> str:
    return _fetch_with_cache(
        url,
        cache_key=f"text:{url}",
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
        loader=lambda s: s,
        suffix=".txt",
    )


def _slugify_model_name(model_name: str) -> str:
    model_name = unescape(model_name or "")
    model_name = re.sub(r"<[^>]+>", " ", model_name)
    model_name = model_name.strip().lower()
    model_name = re.sub(r"\s+", " ", model_name)
    model_name = model_name.replace("_", "-")
    return model_name


def _normalize_model_alias(text: str) -> str:
    text = _slugify_model_name(text)
    text = text.replace(".", "-")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _model_alias_candidates(text: str) -> list[str]:
    base = _normalize_model_alias(text)
    if not base:
        return []
    candidates = [base]
    trimmed = re.sub(r"-(preview|experimental|exp|latest|instruct|turbo|base|thinking|adaptive|chat)(-|$)", "-", base)
    trimmed = re.sub(r"-(20\d{2}(?:-?\d{2}(?:-?\d{2})?)?)(-|$)", "-", trimmed)
    trimmed = re.sub(r"-(\d{2}(?:-\d{2}){1,2})(-|$)", "-", trimmed)
    trimmed = re.sub(r"-+", "-", trimmed).strip("-")
    if trimmed and trimmed not in candidates:
        candidates.append(trimmed)
    return candidates


def _build_catalog_alias_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    try:
        data = json.loads(MODEL_CATALOG_FILE.read_text())
    except Exception:
        return index

    for row in data.get("models") or []:
        model_id = str(row.get("key") or "")
        if not model_id or "/" not in model_id:
            continue
        provider, model_key = model_id.split("/", 1)
        display_name = str(row.get("name") or "")
        candidates = {
            _normalize_model_alias(model_id),
            _normalize_model_alias(model_key),
            _normalize_model_alias(display_name),
        }
        if provider in {"github-copilot", "google-gemini-cli", "openai-codex"}:
            candidates.add(_normalize_model_alias(model_key.replace(".", "-")))
        for candidate in {c for c in candidates if c}:
            index.setdefault(candidate, []).append(model_id)
    return index


CATALOG_ALIAS_INDEX = _build_catalog_alias_index()


def _pick_preferred_model_id(candidates: list[str]) -> Optional[str]:
    if not candidates:
        return None
    unique = sorted(set(candidates), key=lambda model_id: (
        next((i for i, prefix in enumerate(PREFERRED_MODEL_PROVIDERS) if model_id.startswith(prefix)), len(PREFERRED_MODEL_PROVIDERS)),
        len(model_id),
        model_id,
    ))
    return unique[0]


def _map_public_model_name(model_name: str) -> Optional[str]:
    """Map public leaderboard model names to Nexus/OpenClaw model IDs conservatively."""
    key = _slugify_model_name(model_name)
    alias_map = {
        "gpt-4o": "github-copilot/gpt-4o",
        "gpt-4 omni": "github-copilot/gpt-4o",
        "gpt-4-turbo": "github-copilot/gpt-4-turbo",
        "claude-3.5-sonnet": "github-copilot/claude-sonnet-4",
        "claude 3.5 sonnet": "github-copilot/claude-sonnet-4",
        "claude-3.7-sonnet": "github-copilot/claude-sonnet-4.5",
        "claude 3.7 sonnet": "github-copilot/claude-sonnet-4.5",
        "claude-sonnet-4": "github-copilot/claude-sonnet-4",
        "claude sonnet 4": "github-copilot/claude-sonnet-4",
        "claude-sonnet-4.5": "github-copilot/claude-sonnet-4.5",
        "claude sonnet 4.5": "github-copilot/claude-sonnet-4.5",
        "claude-sonnet-4.6": "github-copilot/claude-sonnet-4.6",
        "claude sonnet 4.6": "github-copilot/claude-sonnet-4.6",
        "claude-opus-4": "github-copilot/claude-opus-4.5",
        "claude opus 4": "github-copilot/claude-opus-4.5",
        "claude-opus-4.5": "github-copilot/claude-opus-4.5",
        "claude opus 4.5": "github-copilot/claude-opus-4.5",
        "claude-opus-4.6": "github-copilot/claude-opus-4.6",
        "claude opus 4.6": "github-copilot/claude-opus-4.6",
        "gemini-1.5-pro": "google-gemini-cli/gemini-1.5-pro",
        "gemini 1.5 pro": "google-gemini-cli/gemini-1.5-pro",
        "gemini-1.5-flash": "google-gemini-cli/gemini-1.5-flash",
        "gemini 1.5 flash": "google-gemini-cli/gemini-1.5-flash",
        "gemini-2.0-flash": "google-gemini-cli/gemini-2.0-flash",
        "gemini 2.0 flash": "google-gemini-cli/gemini-2.0-flash",
        "gemini-2.5-flash": "google-gemini-cli/gemini-2.5-flash",
        "gemini 2.5 flash": "google-gemini-cli/gemini-2.5-flash",
        "gemini-2.5-pro": "google-gemini-cli/gemini-2.5-pro",
        "gemini 2.5 pro": "google-gemini-cli/gemini-2.5-pro",
        "gemini-3.1-pro-preview": "google-gemini-cli/gemini-3.1-pro-preview",
        "gemini 3.1 pro preview": "google-gemini-cli/gemini-3.1-pro-preview",
        "gpt-5.2-codex": "openai-codex/gpt-5.2-codex",
        "gpt 5.2 codex": "openai-codex/gpt-5.2-codex",
        "gpt-5.3-codex": "openai-codex/gpt-5.3-codex",
        "gpt 5.3 codex": "openai-codex/gpt-5.3-codex",
        "gpt-5.4": "openai-codex/gpt-5.4",
        "gpt 5.4": "openai-codex/gpt-5.4",
        "gpt-5.4-mini": "openai-codex/gpt-5.4-mini",
        "gpt 5.4 mini": "openai-codex/gpt-5.4-mini",
        "claude-3-sonnet": "github-copilot/claude-sonnet-3",
        "claude 3 sonnet": "github-copilot/claude-sonnet-3",
        "devstral 2": "nvidia/mistralai/devstral-2-123b-instruct-2512",
        "devstral-2": "nvidia/mistralai/devstral-2-123b-instruct-2512",
        "mistral large 3": "nvidia/mistralai/mistral-large-3-675b-instruct-2512",
        "mistral-large-3": "nvidia/mistralai/mistral-large-3-675b-instruct-2512",
        "kimi k2 instruct": "nvidia/moonshotai/kimi-k2-instruct-0905",
        "kimi-k2-instruct": "nvidia/moonshotai/kimi-k2-instruct-0905",
        "kimi k2.5": "nvidia/moonshotai/kimi-k2.5",
        "kimi-k2.5": "nvidia/moonshotai/kimi-k2.5",
        "qwen 3.6 plus": "openrouter/qwen/qwen3.6-plus",
        "qwen3.6-plus": "openrouter/qwen/qwen3.6-plus",
        "gpt-oss 20b": "openai/gpt-oss-20b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "llama 3.3 70b instruct": "meta/llama-3.3-70b-instruct",
        "llama-3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
        "llama 3.3 70b instruct:free": "meta/llama-3.3-70b-instruct:free",
        "llama-3.3-70b-instruct:free": "meta/llama-3.3-70b-instruct:free",
    }
    if key in alias_map:
        return alias_map[key]

    key_no_prefix = re.sub(r"^(openai|anthropic|google|meta|mistral|deepseek|qwen|xai|microsoft|alibaba)[/ :-]+", "", key)
    if key_no_prefix in alias_map:
        return alias_map[key_no_prefix]

    candidates = []
    for value in (model_name, key, key_no_prefix):
        candidates.extend(_model_alias_candidates(value))
    for candidate in candidates:
        picked = _pick_preferred_model_id(CATALOG_ALIAS_INDEX.get(candidate, []))
        if picked:
            return picked
    return None


def _map_helm_model_name(model_name: str) -> Optional[str]:
    """Map HELM display names / adapter names to OpenClaw model IDs conservatively."""
    return _map_public_model_name(model_name)


def fetch_inspect_ai() -> dict[str, dict]:
    """
    Best-effort Inspect AI adapter.

    Inspect AI's public docs do not currently expose a single stable leaderboard API.
    We therefore probe a small set of likely public feeds and only emit scores when we
    can parse them confidently. If nothing stable is available, we warn and skip.
    """
    candidate_urls = [
        "https://ukgovernmentbeis.github.io/inspect_evals/search.json",
        "https://inspect.aisi.org.uk/search.json",
    ]
    for url in candidate_urls:
        try:
            payload = _fetch_json(url)
        except Exception as e:
            print(f"    inspect_ai: unable to read candidate feed {url}: {e}")
            continue

        if not isinstance(payload, list):
            print(f"    inspect_ai: unsupported payload shape at {url} — skipping")
            continue

        result: dict[str, dict] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            model_name = row.get("model") or row.get("model_name") or row.get("name")
            score = row.get("score") or row.get("accuracy")
            if not model_name or score is None:
                continue
            oc_model = _map_public_model_name(str(model_name))
            if not oc_model:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value > 1.0:
                value = value / 100.0
            if not (0.0 <= value <= 1.0):
                continue
            result.setdefault(oc_model, {"_source": "inspect_ai", "_updated_at": TODAY})["reasoning"] = round(value, 4)

        if result:
            print(f"    {len(result)} models mapped from Inspect AI public feed")
            return result

    print("    inspect_ai: no stable public model-score feed found — skipping")
    return {}


def fetch_openllm() -> dict[str, dict]:
    """
    Fetch reasoning/review proxies from the public Open LLM Leaderboard dataset.

    Public dataset:
      https://huggingface.co/datasets/open-llm-leaderboard/contents
    """
    base = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=open-llm-leaderboard%2Fcontents"
        "&config=default&split=train"
    )
    page = 100
    max_rows = 5000
    offset = 0
    result: dict[str, dict] = {}
    seen_rows = 0

    try:
        while offset < max_rows:
            data = _fetch_json(f"{base}&offset={offset}&length={page}")
            rows = data.get("rows") or []
            if not rows:
                break

            for wrapped in rows:
                row = wrapped.get("row") or {}
                seen_rows += 1
                if row.get("Flagged") or row.get("Merged"):
                    continue
                model_name = row.get("fullname") or row.get("Model") or row.get("eval_name")
                oc_model = _map_public_model_name(str(model_name or ""))
                if not oc_model:
                    continue

                entry: dict[str, object] = {"_source": "openllm", "_updated_at": TODAY}
                if row.get("IFEval") is not None:
                    entry["review"] = normalize_reasoning(float(row["IFEval"]))
                reasoning_parts = []
                for key in ("BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"):
                    value = row.get(key)
                    if value is None:
                        continue
                    reasoning_parts.append(normalize_reasoning(float(value)))
                if reasoning_parts:
                    entry["reasoning"] = round(sum(reasoning_parts) / len(reasoning_parts), 4)
                if "review" in entry:
                    entry["summarize"] = round(max(0.0, min(1.0, float(entry["review"]) * 0.97)), 4)

                if len(entry) > 2:
                    result[oc_model] = entry

            offset += page
            if len(rows) < page:
                break
    except Exception as e:
        print(f"    openllm: dataset fetch failed: {e}")
        return {}

    if result:
        print(f"    {seen_rows} Open LLM Leaderboard rows scanned, {len(result)} models mapped to OpenClaw IDs")
    else:
        print(f"    openllm: scanned {seen_rows} rows but found no confidently mappable models")
    return result


def _extract_helm_table_scores(group_name: str, dim: str) -> dict[str, dict]:
    """Read a HELM group table and extract the first score-like mean column per model."""
    base = "https://storage.googleapis.com/crfm-helm-public/benchmark_output/releases/v0.4.0"
    try:
        tables = _fetch_json(f"{base}/groups/{group_name}.json")
    except Exception as e:
        print(f"    helm: structured group fetch failed for {group_name}: {e}")
        return {}

    result: dict[str, dict] = {}
    for table in tables:
        header = table.get("header") or []
        rows = table.get("rows") or []
        if len(header) < 2 or not rows:
            continue
        header_values = [str(col.get("value", "")) for col in header]
        if not any("mean" in h.lower() for h in header_values[1:3]):
            continue

        for row in rows:
            if len(row) < 2:
                continue
            model_name = str(row[0].get("value", "")).strip()
            score = row[1].get("value")
            if not model_name or score is None:
                continue
            oc_model = _map_helm_model_name(model_name)
            if not oc_model:
                continue
            try:
                val = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                continue
            result.setdefault(oc_model, {"_source": "helm", "_updated_at": TODAY})[dim] = round(val, 4)

        if result:
            break

    return result


def fetch_helm() -> dict[str, dict]:
    """
    Fetch HELM leaderboard proxies.

    Primary path: structured JSON files published by HELM.
    Fallback path: lightweight HTML/text scrape of the public leaderboard page.

    This fetcher is intentionally conservative:
    - if HELM changes shape, warn and skip
    - only dimensions we can identify confidently are returned
    - exit path never raises to the caller unless something very unexpected happens
    """
    combined: dict[str, dict] = {}

    for group_name, dim in (("reasoning", "reasoning"), ("summarization", "review")):
        group_scores = _extract_helm_table_scores(group_name, dim)
        for model_id, payload in group_scores.items():
            combined.setdefault(model_id, {"_source": "helm", "_updated_at": TODAY}).update(payload)

    if combined:
        print(f"    {len(combined)} models mapped from HELM structured JSON")
        return combined

    try:
        html = _fetch_text("https://crfm.stanford.edu/helm/classic/latest/")
        html = unescape(html)
        for model_name, score in re.findall(r">([^<>]{2,80})</a></td>\s*<td[^>]*>(0?\.\d+)</td>", html):
            oc_model = _map_helm_model_name(model_name)
            if not oc_model:
                continue
            combined.setdefault(oc_model, {"_source": "helm", "_updated_at": TODAY})["reasoning"] = round(float(score), 4)
    except Exception as e:
        print(f"    helm: fallback leaderboard scrape failed: {e}")

    if combined:
        print(f"    {len(combined)} models mapped from HELM fallback HTML")
    else:
        print("    helm: no confidently mappable model scores found — skipping")
    return combined


def fetch_vellum() -> dict[str, dict]:
    """
    Fetch quality scores from vellum.ai leaderboard.

    Returns dict of model_id → {coding, reasoning, review, _source, _updated_at}

    Implementation note:
    - https://vellum.ai/llm-leaderboard
    - Check page source for embedded JSON or API endpoints
    """
    print("  [vellum] not yet implemented — skipping")
    return {}


def fetch_bfcl() -> dict[str, dict]:
    """
    Fetch tool use scores from Berkeley Function Calling Leaderboard.

    Returns dict of model_id → {tools, _source, _updated_at}

    Implementation note:
    - https://gorilla.cs.berkeley.edu/leaderboard.html
    - Public leaderboard data: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
    - JSON results available in the repo
    """
    print("  [bfcl] not yet implemented — skipping")
    return {}



def fetch_hf_leaderboard() -> dict[str, dict]:
    """Fetch approximate efficiency scores from a curated Hugging Face model map.

    This avoids broad catalog scans/rate limits. Eco score is a conservative
    proxy derived from model/storage size: smaller models score higher.
    """
    curated = {
        "nvidia/moonshotai/kimi-k2-instruct-0905": "moonshotai/Kimi-K2-Instruct",
        "nvidia/moonshotai/kimi-k2.5": "moonshotai/Kimi-K2.5",
        "nvidia/openai/gpt-oss-20b": "openai/gpt-oss-20b",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free": "meta-llama/Llama-3.3-70B-Instruct",
        "nvidia/meta/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "nvidia/meta/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/microsoft/phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
        "openrouter/qwen/qwen3.6-plus": "Qwen/Qwen3-32B",
        "nvidia/mistralai/mistral-large-3-675b-instruct-2512": "mistralai/Mistral-Small-24B-Instruct-2501",
        "nvidia/mistralai/devstral-2-123b-instruct-2512": "mistralai/Devstral-Small-2507",
    }

    def _eco_from_size_gb(size_gb: float) -> float:
        clamped = max(0.1, min(size_gb, 500.0))
        eco_score = (500.0 - clamped) / (500.0 - 0.1)
        return max(0.0, min(1.0, eco_score))

    result: dict[str, dict] = {}
    for model_id, hf_repo in curated.items():
        url = f'https://huggingface.co/api/models/{hf_repo}'
        try:
            payload = _fetch_json(url, timeout=20)
        except Exception as e:
            print(f"    hf_leaderboard: skip {model_id} ({hf_repo}) fetch failed: {e}")
            continue
        size_bytes = None
        st = payload.get('safetensors') or {}
        if isinstance(st, dict):
            size_bytes = st.get('total')
        if not size_bytes:
            size_bytes = payload.get('usedStorage')
        if not size_bytes:
            print(f"    hf_leaderboard: skip {model_id} ({hf_repo}) no size metadata")
            continue
        try:
            size_gb = float(size_bytes) / (1024.0 ** 3)
        except Exception:
            continue
        result[model_id] = {
            '_source': 'hf_leaderboard',
            '_updated_at': TODAY,
            'eco': round(_eco_from_size_gb(size_gb), 4),
            '_metadata': {
                'hf_model_repo': hf_repo,
                'model_size_gb': round(size_gb, 4),
                'source_kind': 'hf-model-api-curated',
            },
        }
    if result:
        print(f"    hf_leaderboard: produced {len(result)} curated eco rows")
    else:
        print('    hf_leaderboard: no curated eco rows produced')
    return result


def fetch_mgsm() -> dict[str, dict]:
    """
    Fetch multilingual scores from MGSM or similar.

    Returns dict of model_id → {multilingual, _source, _updated_at}

    Implementation note:
    - MGSM: https://github.com/google-research/url-nlp/tree/main/mgsm
    - Some leaderboards (HELM, LiveBench) include multilingual sub-scores
    """
    print("  [mgsm] not yet implemented — skipping")
    return {}


# ── Source registry ───────────────────────────────────────────────────────────

SOURCES = {
    "aa":        ("artificialanalysis", fetch_artificialanalysis),
    "lb":        ("livebench",          fetch_livebench),
    "helm":      ("helm",               fetch_helm),
    "hl":        ("helm",               fetch_helm),
    "inspect":   ("inspect_ai",         fetch_inspect_ai),
    "inspect_ai":("inspect_ai",         fetch_inspect_ai),
    "ia":        ("inspect_ai",         fetch_inspect_ai),
    "openllm":   ("openllm",            fetch_openllm),
    "open-llm":  ("openllm",            fetch_openllm),
    "ollm":      ("openllm",            fetch_openllm),
    "vl":        ("vellum",             fetch_vellum),
    "bfcl":      ("bfcl",               fetch_bfcl),
    "mgsm":      ("mgsm",               fetch_mgsm),
    "hf_leaderboard": ("hf_leaderboard", fetch_hf_leaderboard),
}

ALL_SOURCES = list(SOURCES.keys())


# ── Merge logic ───────────────────────────────────────────────────────────────

SCORE_DIMS = ["coding", "review", "reasoning", "summarize", "fast", "cost", "eco",
              "context", "vision", "tools", "multilingual"]


CONFIDENCE_COUNT_CAP = 3.0
CONFIDENCE_WEIGHT_CAP = 2.0
CONFIDENCE_STDDEV_CAP = 0.25
SINGLE_SOURCE_AGREEMENT_CONF = 0.65


def _source_weight(source_name: str) -> float:
    return float(SOURCE_WEIGHTS.get(source_name, 1.0))


def _canonical_source_name(source_name: str) -> str:
    if "+" in source_name:
        return source_name.split("+")[0]
    return source_name


def _compute_dimension_confidence(values: list[float], total_weight: float) -> float:
    """Return a moderate-to-high confidence score for one merged dimension."""
    contributors = len(values)
    if contributors == 0 or total_weight <= 0:
        return 0.0

    base_count_conf = min(1.0, contributors / CONFIDENCE_COUNT_CAP)
    base_weight_conf = min(1.0, total_weight / CONFIDENCE_WEIGHT_CAP)
    if contributors == 1:
        agreement_conf = SINGLE_SOURCE_AGREEMENT_CONF
    else:
        stddev = statistics.pstdev(values)
        agreement_conf = 1.0 - min(1.0, stddev / CONFIDENCE_STDDEV_CAP)

    final_conf = (
        0.4 * base_count_conf
        + 0.3 * base_weight_conf
        + 0.3 * agreement_conf
    )
    return round(max(0.0, min(1.0, final_conf)), 4)


def aggregate_source_entries(entries: list[dict]) -> dict:
    """Combine multiple source payloads for one model using per-dimension weights."""
    aggregated: dict = {}
    confidence: dict[str, float] = {}
    contributing_sources: set[str] = set()

    for dim in SCORE_DIMS:
        numerator = 0.0
        denominator = 0.0
        dim_sources: set[str] = set()
        dim_values: list[float] = []
        for entry in entries:
            if dim not in entry or entry[dim] is None:
                continue
            source_name = _canonical_source_name(entry.get("_source", ""))
            weight = _source_weight(source_name)
            value = float(entry[dim])
            numerator += weight * value
            denominator += weight
            dim_values.append(value)
            if source_name:
                dim_sources.add(source_name)
        if denominator > 0:
            aggregated[dim] = round(numerator / denominator, 4)
            confidence[dim] = _compute_dimension_confidence(dim_values, denominator)
            contributing_sources.update(dim_sources)

    if confidence:
        aggregated["_confidence"] = confidence
    if contributing_sources:
        aggregated["_sources"] = sorted(contributing_sources)
        aggregated["_source"] = "+".join(sorted(contributing_sources))
    elif entries:
        aggregated["_source"] = entries[-1].get("_source", "")
    aggregated["_updated_at"] = TODAY
    return aggregated


def merge_scores(existing: dict, new_scores: dict, force: bool) -> dict:
    """
    Merge aggregated benchmark scores into an existing model entry.
    If force=False, existing non-null scores are preserved.
    """
    merged = dict(existing)
    for dim in SCORE_DIMS:
        if dim in new_scores and (force or dim not in merged):
            merged[dim] = new_scores[dim]

    existing_confidence = existing.get("_confidence") or {}
    new_confidence = new_scores.get("_confidence") or {}
    merged_confidence = dict(existing_confidence)
    for dim, score in new_confidence.items():
        if force or dim not in merged_confidence:
            merged_confidence[dim] = score
    if merged_confidence:
        merged["_confidence"] = merged_confidence

    prev_sources = set(existing.get("_sources") or [])
    prev_sources |= {s for s in str(existing.get("_source", "")).split("+") if s}
    new_sources = set(new_scores.get("_sources") or [])
    new_sources |= {s for s in str(new_scores.get("_source", "")).split("+") if s}
    all_sources = sorted(prev_sources | new_sources)
    if all_sources:
        merged["_sources"] = all_sources
        merged["_source"] = "+".join(all_sources)
    merged["_updated_at"] = TODAY
    return merged


def _load_catalog_models() -> list[str]:
    """Return every OpenClaw model key from the raw catalog."""
    try:
        raw = MODEL_CATALOG_FILE.read_text()
        stripped = raw.lstrip()
        if stripped.startswith("{"):
            data = json.loads(stripped)
        else:
            obj_start = raw.find("{")
            arr_start = raw.find("[")
            start = obj_start if obj_start != -1 else arr_start
            if start == -1:
                return []
            data = json.loads(raw[start:])
    except Exception:
        return []

    result = []
    for row in data.get("models") or []:
        key = str(row.get("key") or "").strip()
        if key and "/" in key:
            result.append(key)
    return sorted(set(result))


def _sibling_model_keys(model_id: str, available_keys: set[str]) -> list[str]:
    suffix = f"/{model_id}"
    return sorted(k for k in available_keys if k.endswith(suffix))


def _has_score_dimensions(entry: dict) -> bool:
    return any(dim in entry and entry.get(dim) is not None for dim in SCORE_DIMS)


def _is_real_benchmark_source(entry: dict) -> bool:
    source = str(entry.get("_source") or "").strip()
    if not source or source == "catalog-only":
        return False
    if "inherited:" in source:
        return False
    return True


def densify_benchmark_scores(existing: dict[str, dict]) -> dict[str, dict]:
    """
    Ensure every OC catalog model has a benchmark_model_scores row.

    For models without exact benchmark data, inherit score dimensions from any
    sibling `*/same-model-id` row when available. This keeps the benchmark table
    dense over the OC catalog while preserving provider-specific runtime
    differences for other layers (health/quota/latency).
    """
    dense = {k: dict(v) for k, v in existing.items()}
    all_keys = set(existing.keys())
    catalog_models = _load_catalog_models()
    all_keys.update(catalog_models)

    for full_key in sorted(all_keys):
        if "/" not in full_key:
            continue
        provider, model_id = full_key.split("/", 1)
        row = dict(dense.get(full_key, {}))

        # Keep exact rows only when they come from a real benchmark source.
        # Synthetic rows (catalog-only / inherited) must be recomputed on each run
        # so they can upgrade from catalog-only -> inherited, refresh inherited
        # values, or fall back to catalog-only if no real donor remains.
        if not (_has_score_dimensions(row) and _is_real_benchmark_source(row)):
            siblings = [k for k in _sibling_model_keys(model_id, set(dense.keys())) if k != full_key and dense.get(k)]
            inherited = None
            for sib in siblings:
                payload = dense.get(sib) or {}
                if _has_score_dimensions(payload) and _is_real_benchmark_source(payload):
                    inherited = (sib, payload)
                    break
            if inherited:
                sib_key, payload = inherited
                row = {dim: payload[dim] for dim in SCORE_DIMS if dim in payload}
                if "_confidence" in payload:
                    row["_confidence"] = dict(payload["_confidence"])
                row["_source"] = f"inherited:{sib_key}"
                row["_updated_at"] = TODAY
            else:
                row = {"_source": "catalog-only", "_updated_at": TODAY}

        dense[full_key] = row

    return dense


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch benchmark data for Nexus Router")
    parser.add_argument("--sources", type=str, default=",".join(ALL_SOURCES),
                        help=f"Comma-separated source ids / aliases: {', '.join(ALL_SOURCES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scores (default: skip existing)")
    args = parser.parse_args()

    requested = {s.strip() for s in args.sources.split(",")}
    invalid = requested - set(SOURCES.keys())
    if invalid:
        print(f"Unknown sources: {invalid}. Valid: {set(SOURCES.keys())}", file=sys.stderr)
        sys.exit(1)

    existing: dict[str, dict] = load_benchmark_scores()
    if existing:
        print(f"Loaded {len(existing)} existing model entries from router DB")
    else:
        print("No existing benchmark entries in router DB — starting fresh")

    # Fetch and aggregate per source
    all_new: dict[str, list[dict]] = {}
    seen_labels: set[str] = set()
    for src_id in requested:
        label, fetcher = SOURCES[src_id]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        print(f"Fetching {label}...")
        try:
            results = fetcher()
            for model_id, scores in results.items():
                all_new.setdefault(model_id, []).append(scores)
            if results:
                print(f"  → {len(results)} model entries")
        except Exception as e:
            print(f"  [warning] {label} fetch failed: {e}", file=sys.stderr)

    aggregated_new = {
        model_id: aggregate_source_entries(entries)
        for model_id, entries in all_new.items()
    }

    # Merge into existing sparse benchmark set first
    merged = dict(existing)
    for model_id, new_scores in aggregated_new.items():
        merged[model_id] = merge_scores(merged.get(model_id, {}), new_scores, args.force)

    # Then densify over the full OC catalog so every model has a row.
    merged = densify_benchmark_scores(merged)

    new_count = len(aggregated_new)
    print(f"\nTotal benchmark entries: {len(merged)} ({new_count} new/updated, dense over OC catalog)")

    if args.dry_run:
        print("\n--- DRY RUN: would write ---")
        for mid, scores in list(merged.items())[:5]:
            print(f"  {mid}: {scores}")
        if len(merged) > 5:
            print(f"  ... and {len(merged) - 5} more")
        return

    replace_benchmark_scores(merged)
    print("Written benchmark scores → router DB")


if __name__ == "__main__":
    main()
