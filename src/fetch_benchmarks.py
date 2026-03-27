#!/usr/bin/env python3
"""
fetch_benchmarks.py — Fetch benchmark data and generate benchmark-scores.yaml.

Sources (configurable in policies/benchmark-sources.yaml):
  - artificialanalysis.ai  → speed, cost, context
  - livebench.ai           → reasoning, coding
  - vellum.ai              → multi-task quality
  - BFCL                   → tool use
  - MGSM                   → multilingual

This script:
1. Fetches structured data from each source
2. Normalizes scores to 0.0–1.0 per router dimension
3. Merges scores across sources (later sources don't override earlier ones
   unless --force is passed)
4. Writes/updates policies/benchmark-scores.yaml

Usage:
  python src/fetch_benchmarks.py                    # fetch all configured sources
  python src/fetch_benchmarks.py --sources aa,lb    # only artificialanalysis + livebench
  python src/fetch_benchmarks.py --dry-run          # print without writing
  python src/fetch_benchmarks.py --force            # overwrite existing scores

Note: This script currently implements the fetch framework and normalization
logic. Individual source parsers are stubs — implement them as each source
provides structured data access (API, CSV download, or scraping).
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("Missing: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT           = Path(__file__).parent.parent
BENCHMARKS_OUT = ROOT / "policies/benchmark-scores.yaml"
SOURCES_FILE   = ROOT / "policies/benchmark-sources.yaml"

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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

    Returns dict of model_id → {fast, cost, context, _source, _updated_at}

    Implementation note:
    - Check https://artificialanalysis.ai/leaderboards/models for export options
    - If no public API exists, consider scraping the JSON embedded in the page
    - Look for network requests to /api/models or similar endpoints in DevTools
    """
    print("  [artificialanalysis] not yet implemented — skipping")
    return {}


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
    PAGE = 500  # rows per request
    MAX_ROWS = 60_500

    # Aggregate: {model -> {category -> (sum, count)}}
    agg: dict[str, dict[str, list]] = {}

    offset = 0
    total_fetched = 0
    try:
        while offset < MAX_ROWS:
            url = f"{BASE}&offset={offset}&length={PAGE}"
            req = urllib.request.Request(url, headers={"User-Agent": "nexus-router/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

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
    "aa":    ("artificialanalysis", fetch_artificialanalysis),
    "lb":    ("livebench",          fetch_livebench),
    "vl":    ("vellum",             fetch_vellum),
    "bfcl":  ("bfcl",              fetch_bfcl),
    "mgsm":  ("mgsm",              fetch_mgsm),
}

ALL_SOURCES = list(SOURCES.keys())


# ── Merge logic ───────────────────────────────────────────────────────────────

SCORE_DIMS = ["coding", "review", "reasoning", "summarize", "fast", "cost",
              "context", "vision", "tools", "multilingual"]


def merge_scores(existing: dict, new_scores: dict, force: bool) -> dict:
    """
    Merge new benchmark scores into existing model entry.
    If force=False, existing non-null scores are not overwritten.
    """
    merged = dict(existing)
    for dim in SCORE_DIMS:
        if dim in new_scores:
            if force or dim not in merged:
                merged[dim] = new_scores[dim]
    # Always update source and date
    if "_source" in new_scores:
        prev_src = merged.get("_source", "")
        new_src = new_scores["_source"]
        merged["_source"] = f"{prev_src}+{new_src}".strip("+") if prev_src else new_src
    merged["_updated_at"] = TODAY
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch benchmark data for Nexus Router")
    parser.add_argument("--sources", type=str, default=",".join(ALL_SOURCES),
                        help=f"Comma-separated source ids: {', '.join(ALL_SOURCES)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scores (default: skip existing)")
    parser.add_argument("--output", type=Path, default=BENCHMARKS_OUT,
                        help=f"Output file (default: {BENCHMARKS_OUT})")
    args = parser.parse_args()

    requested = {s.strip() for s in args.sources.split(",")}
    invalid = requested - set(SOURCES.keys())
    if invalid:
        print(f"Unknown sources: {invalid}. Valid: {set(SOURCES.keys())}", file=sys.stderr)
        sys.exit(1)

    # Load existing benchmarks
    existing: dict[str, dict] = {}
    if args.output.exists():
        data = yaml.safe_load(args.output.read_text()) or {}
        existing = data.get("models") or {}
        print(f"Loaded {len(existing)} existing model entries from {args.output}")
    else:
        print(f"No existing file at {args.output} — starting fresh")

    # Fetch and merge
    all_new: dict[str, dict] = {}
    for src_id in requested:
        label, fetcher = SOURCES[src_id]
        print(f"Fetching {label}...")
        try:
            results = fetcher()
            for model_id, scores in results.items():
                all_new.setdefault(model_id, {})
                all_new[model_id].update(scores)
            if results:
                print(f"  → {len(results)} model entries")
        except Exception as e:
            print(f"  [warning] {label} fetch failed: {e}", file=sys.stderr)

    # Merge into existing
    merged = dict(existing)
    for model_id, new_scores in all_new.items():
        merged[model_id] = merge_scores(merged.get(model_id, {}), new_scores, args.force)

    new_count = len(all_new)
    print(f"\nTotal benchmark entries: {len(merged)} ({new_count} new/updated)")

    if args.dry_run:
        print("\n--- DRY RUN: would write ---")
        for mid, scores in list(merged.items())[:5]:
            print(f"  {mid}: {scores}")
        if len(merged) > 5:
            print(f"  ... and {len(merged) - 5} more")
        return

    # Write output (preserve header comments by loading and re-writing cleanly)
    _write_benchmarks(args.output, merged)
    print(f"Written → {args.output}")


def _write_benchmarks(path: Path, models: dict[str, dict]):
    # Load existing file to preserve _meta block
    header = ""
    if path.exists():
        raw = path.read_text()
        # Keep everything up to the 'models:' key
        m = re.search(r"^models:", raw, re.MULTILINE)
        if m:
            header = raw[:m.start()]
    else:
        header = "# benchmark-scores.yaml — generated by fetch_benchmarks.py\n\n"

    models_yaml = yaml.dump({"models": models}, default_flow_style=False, allow_unicode=True, sort_keys=True)
    # strip the 'models:' top-level key (already in header context)
    path.write_text(header + models_yaml)


if __name__ == "__main__":
    main()
