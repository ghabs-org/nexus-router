#!/usr/bin/env python3
"""Refresh the raw OpenClaw model catalog, refusing results that look partial.

Runs `openclaw models list --all --json` on the host (the CLI is not in the
container) and replaces RAW_CATALOG_FILE only when the result passes the guard.
It refuses (exit 2) when:
  - the output has no models, or no model tagged "configured" (the registry
    keeps only configured models, so it would come out empty)
  - the total model count dropped by more than half
  - a provider whose discovery did not report "ready" lost more than half of
    its models (a sign-in or network failure, not a config change)

A provider that disappears with no discovery outcome at all is how OpenClaw
reports a disabled provider plugin, so that is only reported, not refused.

It also stores agents.defaults.modelPolicy.allow in the catalog as
"modelPolicyAllow": some built-in models are tagged "configured" without being
allowed, and generate_registry.py needs the list to leave them out.

The previous catalog is kept next to the new one as openclaw-models.prev.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# Import paths.py directly: the src package __init__ pulls in router deps
# (pyyaml, numpy) that the host python does not have.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paths import RAW_CATALOG_FILE, ensure_parent

EXIT_REFUSED = 2
MAX_DROP_RATIO = 0.5
MIN_PROVIDER_MODELS = 3  # smaller providers can come and go without tripping the guard
MIN_NVIDIA_LIVE_CHAT_MODELS = 2

NVIDIA_MODELS_URL = os.environ.get(
    "NEXUS_ROUTER_NVIDIA_MODELS_URL",
    "https://integrate.api.nvidia.com/v1/models",
)
NVIDIA_FEATURED_MODELS_URL = os.environ.get(
    "NEXUS_ROUTER_NVIDIA_FEATURED_MODELS_URL",
    "https://assets.ngc.nvidia.com/products/api-catalog/featured-models.json",
)

_NVIDIA_NON_CHAT_MODEL_TYPES = {
    "embedding",
    "embeddings",
    "rerank",
    "reranker",
    "classifier",
    "detection",
    "asr",
    "tts",
    "safety",
    "guard",
    "translation",
}
_NVIDIA_NON_CHAT_RE = re.compile(
    r"(?:^|[-_/.:])(?:embed|embedqa|nemoretriever|retriever|rerank|nvclip|clip|"
    r"nemoguard|safety[-_]?guard|content[-_]?safety|topic[-_]?control|pii|"
    r"detector|translate|calibration|starcoder2|cosmos|whisper|parakeet|"
    r"canary|asr|tts|ocr|parse)(?:$|[-_/.:])",
    re.IGNORECASE,
)
_ACRONYM_TOKENS = {"ai", "vl", "it", "xs", "xl", "glm", "e4b", "1t", "a52b"}
_NVIDIA_ZERO_COST = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}


def openclaw_bin() -> str:
    return os.environ.get("OPENCLAW_BIN") or shutil.which("openclaw") or str(Path.home() / ".local/bin/openclaw")


def fetch_catalog(timeout: int) -> dict:
    result = subprocess.run(
        [openclaw_bin(), "models", "list", "--all", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in result.stderr.splitlines():
        if line.strip():
            print(f"  openclaw: {line.strip()}", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"openclaw exited with {result.returncode}")
    start = result.stdout.find("{")
    if start == -1:
        raise RuntimeError("openclaw printed no JSON object")
    return json.loads(result.stdout[start:])


def _parse_json_after_banner(stdout: str):
    starts = [i for i in (stdout.find("{"), stdout.find("[")) if i != -1]
    if not starts:
        return None
    return json.loads(stdout[min(starts):])


def read_openclaw_config_section(dot_path: str, timeout: int):
    """Read a JSON config section via `openclaw config get`, or None if unset."""
    try:
        result = subprocess.run(
            [openclaw_bin(), "config", "get", dot_path, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return _parse_json_after_banner(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"warning: could not read {dot_path}: {exc}", file=sys.stderr)
        return None


def fetch_allowlist(timeout: int) -> list[str] | None:
    """Return OpenClaw's model allowlist, or None when it is unset or unreadable."""
    allow = read_openclaw_config_section("agents.defaults.modelPolicy.allow", timeout)
    return allow if isinstance(allow, list) and all(isinstance(a, str) for a in allow) else None


def nvidia_api_id_to_openclaw_key(api_id: str) -> str:
    api_id = api_id.strip()
    return api_id if api_id.startswith("nvidia/") else f"nvidia/{api_id}"


def openclaw_key_to_nvidia_api_id(key: str) -> str | None:
    if not isinstance(key, str) or not key.startswith("nvidia/"):
        return None
    tail = key.split("/", 1)[1]
    if not tail:
        return None
    return tail if "/" in tail else f"nvidia/{tail}"


def is_nvidia_chat_model(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in model_id):
        return False
    model_type = str(row.get("model_type") or "").strip().lower()
    if model_type in _NVIDIA_NON_CHAT_MODEL_TYPES:
        return False
    if _NVIDIA_NON_CHAT_RE.search(model_id):
        return False
    return True


def fetch_nvidia_live_models(timeout: int = 15) -> list[dict]:
    req = urllib.request.Request(
        NVIDIA_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "nexus-router/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("NVIDIA /v1/models response missing data[]")
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"].strip()]


def fetch_nvidia_featured_models(timeout: int = 10) -> dict[str, dict]:
    try:
        req = urllib.request.Request(
            NVIDIA_FEATURED_MODELS_URL,
            headers={"Accept": "application/json", "User-Agent": "nexus-router/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    rows = payload.get("featured-models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("model") or "").strip()
        if not raw_id:
            continue
        api_id = raw_id if "/" in raw_id else f"nvidia/{raw_id}"
        out[api_id] = row
    return out


def _positive_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = int(value)
    return val if val > 0 else None


def _format_model_display_name(api_id: str) -> str:
    tail = api_id.split("/", 1)[-1]
    words = []
    for tok in tail.split("-"):
        if not tok:
            continue
        low = tok.lower()
        if low in _ACRONYM_TOKENS or re.fullmatch(r"(?:\d+[tb]|a\d+b)", low):
            words.append(tok.upper())
        else:
            words.append(tok[0].upper() + tok[1:])
    return " ".join(words) if words else api_id


def _clean_sibling_name(name: str) -> str:
    cleaned = re.sub(r"^[A-Za-z0-9._-]+:\s+", "", name.strip())
    cleaned = re.sub(r"\s+\((?:free|batch)\)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _sibling_metadata_index(old_catalog: dict | None) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in (old_catalog or {}).get("models") or []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        if not key:
            continue
        suffix = key.rsplit("/", 1)[-1].split(":", 1)[0].lower()
        if not suffix:
            continue
        candidate_rank = (
            1 if key.startswith("nvidia/") else 0,
            1 if "image" in str(row.get("input") or "").lower() else 0,
            _positive_int(row.get("contextWindow")) or 0,
        )
        prev = index.get(suffix)
        if prev is None:
            index[suffix] = row
            continue
        prev_key = str(prev.get("key") or "")
        prev_rank = (
            1 if prev_key.startswith("nvidia/") else 0,
            1 if "image" in str(prev.get("input") or "").lower() else 0,
            _positive_int(prev.get("contextWindow")) or 0,
        )
        if candidate_rank > prev_rank:
            index[suffix] = row
    return index


def build_nvidia_provider_model(
    row: dict,
    existing: dict | None,
    featured: dict | None,
    sibling: dict | None,
) -> dict:
    api_id = row["id"].strip()
    ctx = _positive_int(row.get("context_window"))
    if isinstance(existing, dict):
        entry = dict(existing)
        if ctx is not None:
            entry["contextWindow"] = ctx
        return entry

    featured_name = str((featured or {}).get("model-name") or "").strip()
    sibling_name = _clean_sibling_name(str((sibling or {}).get("name") or ""))
    name = featured_name or sibling_name or _format_model_display_name(api_id)

    model_type = str(row.get("model_type") or "").strip().lower()
    sibling_input = str((sibling or {}).get("input") or "").lower()
    has_vision = model_type == "vlm" or "-vl-" in api_id.lower() or "image" in sibling_input
    inputs = ["text", "image"] if has_vision else ["text"]

    context_window = (
        ctx
        or _positive_int((featured or {}).get("context"))
        or _positive_int((sibling or {}).get("contextWindow"))
        or 131072
    )
    max_tokens = _positive_int((featured or {}).get("max-output")) or 8192

    return {
        "id": api_id,
        "name": name,
        "api": "openai-completions",
        "reasoning": bool(row.get("supports_thinking")) or "reasoning" in api_id.lower(),
        "input": inputs,
        "cost": dict(_NVIDIA_ZERO_COST),
        "contextWindow": context_window,
        "maxTokens": max_tokens,
    }


def build_nvidia_config_patch(
    live_rows: list[dict],
    existing_models: list[dict],
    existing_default_models: dict[str, dict],
    protected_keys: set[str],
    featured_by_id: dict[str, dict],
    old_catalog: dict | None,
) -> tuple[dict | None, dict]:
    chat_rows_by_id: dict[str, dict] = {}
    for row in live_rows:
        if is_nvidia_chat_model(row):
            chat_rows_by_id[row["id"].strip()] = row

    if len(chat_rows_by_id) < MIN_NVIDIA_LIVE_CHAT_MODELS:
        raise RuntimeError(
            f"Refusing NVIDIA sync: only {len(chat_rows_by_id)} live chat models "
            f"(minimum {MIN_NVIDIA_LIVE_CHAT_MODELS})"
        )

    existing_by_id = {
        str(m.get("id") or "").strip(): m
        for m in existing_models
        if isinstance(m, dict) and str(m.get("id") or "").strip()
    }
    siblings = _sibling_metadata_index(old_catalog)

    keep_ids = set(chat_rows_by_id)
    for pkey in protected_keys:
        api_id = openclaw_key_to_nvidia_api_id(pkey)
        if api_id and api_id in existing_by_id:
            keep_ids.add(api_id)

    new_models: list[dict] = []
    for api_id in sorted(keep_ids):
        if api_id in chat_rows_by_id:
            row = chat_rows_by_id[api_id]
            suffix = api_id.rsplit("/", 1)[-1].lower()
            entry = build_nvidia_provider_model(
                row,
                existing_by_id.get(api_id),
                featured_by_id.get(api_id),
                siblings.get(suffix),
            )
        else:
            entry = dict(existing_by_id[api_id])
        new_models.append(entry)

    desired_oc_keys = {nvidia_api_id_to_openclaw_key(api_id) for api_id in keep_ids}
    desired_oc_keys.update(k for k in protected_keys if isinstance(k, str) and k.startswith("nvidia/"))

    defaults_models_patch: dict[str, dict | None] = {}
    existing_nv_defaults = {
        k: v
        for k, v in existing_default_models.items()
        if isinstance(k, str) and k.startswith("nvidia/")
    }
    for oc_key in sorted(desired_oc_keys):
        if oc_key not in existing_nv_defaults:
            defaults_models_patch[oc_key] = {}
    for oc_key in sorted(existing_nv_defaults):
        if oc_key not in desired_oc_keys:
            defaults_models_patch[oc_key] = None

    added_ids = sorted(keep_ids - set(existing_by_id))
    removed_ids = sorted(set(existing_by_id) - keep_ids)
    kept_ids = sorted(keep_ids & set(existing_by_id))
    summary = {
        "previous_total": len(existing_models),
        "total": len(new_models),
        "added": added_ids,
        "removed": removed_ids,
        "kept": kept_ids,
    }

    if new_models == existing_models and not defaults_models_patch:
        return None, summary

    patch: dict = {"models": {"providers": {"nvidia": {"models": new_models}}}}
    if defaults_models_patch:
        patch["agents"] = {"defaults": {"models": defaults_models_patch}}
    return patch, summary


def apply_nvidia_config_patch(patch: dict, *, dry_run: bool, timeout: int) -> None:
    cmd = [openclaw_bin(), "config", "patch", "--stdin"]
    if "models" in patch:
        cmd.extend(["--replace-path", "models.providers.nvidia.models"])
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(
        cmd,
        input=json.dumps(patch),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"openclaw config patch failed ({result.returncode}): {detail}")


def reconcile_nvidia_catalog(catalog: dict, live_api_ids: set[str]) -> tuple[int, int]:
    live_count = 0
    disabled_count = 0
    for model in catalog.get("models") or []:
        if not isinstance(model, dict):
            continue
        api_id = openclaw_key_to_nvidia_api_id(str(model.get("key") or ""))
        if api_id is None:
            continue
        if api_id in live_api_ids:
            live_count += 1
        else:
            model["available"] = False
            disabled_count += 1
    return live_count, disabled_count


def provider_counts(models: list[dict]) -> Counter:
    return Counter(str(m.get("key") or "").split("/", 1)[0] for m in models if m.get("key"))


def is_configured(model: dict) -> bool:
    return "configured" in (model.get("tags") or [])


def check(new: dict, old: dict | None) -> list[str]:
    """Return the reasons to refuse `new`; empty when it is safe to write."""
    models = new.get("models")
    if not isinstance(models, list) or not models:
        return ["catalog has no models"]

    problems = []
    if not any(is_configured(m) for m in models):
        problems.append("no model is tagged 'configured'")
    if old is None:
        return problems

    old_models = old.get("models") or []
    if len(models) < len(old_models) * (1 - MAX_DROP_RATIO):
        problems.append(f"total models dropped {len(old_models)} -> {len(models)}")

    outcomes = {o.get("provider"): o.get("status") for o in new.get("providerOutcomes") or []}
    new_counts = provider_counts(models)
    for provider, before in provider_counts(old_models).items():
        after = new_counts.get(provider, 0)
        status = outcomes.get(provider)
        if (
            status not in (None, "ready")
            and before >= MIN_PROVIDER_MODELS
            and after < before * (1 - MAX_DROP_RATIO)
        ):
            problems.append(f"{provider} dropped {before} -> {after} while discovery reported {status}")
    return problems


def report(new: dict, old: dict | None) -> None:
    models = new.get("models") or []
    new_counts = provider_counts(models)
    old_counts = provider_counts((old or {}).get("models") or [])
    allow = new.get("modelPolicyAllow")
    print(
        f"catalog: {sum(old_counts.values())} -> {sum(new_counts.values())} models "
        f"({sum(is_configured(m) for m in models)} configured, "
        f"{f'{len(allow)} allowlist entries' if allow else 'no allowlist'})"
    )
    for provider in sorted(set(new_counts) | set(old_counts)):
        before, after = old_counts.get(provider, 0), new_counts.get(provider, 0)
        if before != after:
            print(f"  {provider:24} {before:4} -> {after:4}")
    for outcome in new.get("providerOutcomes") or []:
        if outcome.get("status") != "ready":
            print(f"  warning: discovery for {outcome.get('provider')} reported {outcome.get('status')}")


def write_catalog(catalog: dict) -> None:
    ensure_parent(RAW_CATALOG_FILE)
    if RAW_CATALOG_FILE.exists():
        shutil.copy2(RAW_CATALOG_FILE, RAW_CATALOG_FILE.with_name(f"{RAW_CATALOG_FILE.stem}.prev.json"))
    tmp = RAW_CATALOG_FILE.with_name(f"{RAW_CATALOG_FILE.name}.tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n")
    os.replace(tmp, RAW_CATALOG_FILE)


def sync_nvidia_openclaw_config(
    *,
    old_catalog: dict | None,
    dry_run: bool,
    timeout: int,
) -> set[str]:
    live_rows = fetch_nvidia_live_models()
    featured = fetch_nvidia_featured_models()

    existing_models = read_openclaw_config_section("models.providers.nvidia.models", timeout)
    if not isinstance(existing_models, list):
        existing_models = []
    existing_defaults = read_openclaw_config_section("agents.defaults.models", timeout)
    if not isinstance(existing_defaults, dict):
        existing_defaults = {}
    default_model_cfg = read_openclaw_config_section("agents.defaults.model", timeout)
    protected_keys: set[str] = set()
    if isinstance(default_model_cfg, dict):
        candidates = [default_model_cfg.get("primary"), *(default_model_cfg.get("fallbacks") or [])]
        protected_keys = {
            k.strip() for k in candidates if isinstance(k, str) and k.strip().startswith("nvidia/")
        }

    patch, summary = build_nvidia_config_patch(
        live_rows=live_rows,
        existing_models=existing_models,
        existing_default_models=existing_defaults,
        protected_keys=protected_keys,
        featured_by_id=featured,
        old_catalog=old_catalog,
    )
    print(
        f"nvidia: {summary['previous_total']} -> {summary['total']} live chat models "
        f"(+{len(summary['added'])}, -{len(summary['removed'])})"
    )
    if patch is not None:
        apply_nvidia_config_patch(patch, dry_run=dry_run, timeout=timeout)

    return {row["id"].strip() for row in live_rows if is_nvidia_chat_model(row)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the raw OpenClaw model catalog")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and compare, but do not write")
    parser.add_argument("--force", action="store_true", help="Write even when the guard refuses")
    parser.add_argument("--skip-nvidia-sync", action="store_true", help="Skip syncing live NVIDIA chat models")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for openclaw")
    args = parser.parse_args()

    old = None
    if RAW_CATALOG_FILE.exists():
        try:
            old = json.loads(RAW_CATALOG_FILE.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: ignoring unreadable {RAW_CATALOG_FILE}: {exc}", file=sys.stderr)

    live_nvidia_ids: set[str] | None = None
    if not args.skip_nvidia_sync:
        try:
            live_nvidia_ids = sync_nvidia_openclaw_config(
                old_catalog=old,
                dry_run=args.dry_run,
                timeout=args.timeout,
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"warning: NVIDIA live sync skipped: {exc}", file=sys.stderr)

    try:
        new = fetch_catalog(args.timeout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"catalog fetch failed: {exc}", file=sys.stderr)
        return 1

    if live_nvidia_ids is not None:
        live_cnt, disabled_cnt = reconcile_nvidia_catalog(new, live_nvidia_ids)
        if disabled_cnt:
            print(f"nvidia catalog reconcile: {live_cnt} live, {disabled_cnt} marked unavailable")

    allow = fetch_allowlist(args.timeout)
    if allow is not None:
        new["modelPolicyAllow"] = allow

    report(new, old)
    problems = check(new, old)
    for problem in problems:
        print(f"  refused: {problem}", file=sys.stderr)
    if problems and not args.force:
        print(f"keeping {RAW_CATALOG_FILE} (rerun with --force to override)", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        print("dry run: catalog not written")
        return 0

    write_catalog(new)
    print(f"wrote {RAW_CATALOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
