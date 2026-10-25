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
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Import paths.py directly: the src package __init__ pulls in router deps
# (pyyaml, numpy) that the host python does not have.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paths import RAW_CATALOG_FILE, ensure_parent

EXIT_REFUSED = 2
MAX_DROP_RATIO = 0.5
MIN_PROVIDER_MODELS = 3  # smaller providers can come and go without tripping the guard


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


def fetch_allowlist(timeout: int) -> list[str] | None:
    """Return OpenClaw's model allowlist, or None when it is unset or unreadable."""
    try:
        result = subprocess.run(
            [openclaw_bin(), "config", "get", "agents.defaults.modelPolicy.allow", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        allow = json.loads(result.stdout[result.stdout.find("["):]) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"warning: could not read the model allowlist: {exc}", file=sys.stderr)
        return None
    return allow if isinstance(allow, list) and all(isinstance(a, str) for a in allow) else None


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the raw OpenClaw model catalog")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and compare, but do not write")
    parser.add_argument("--force", action="store_true", help="Write even when the guard refuses")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for openclaw")
    args = parser.parse_args()

    old = None
    if RAW_CATALOG_FILE.exists():
        try:
            old = json.loads(RAW_CATALOG_FILE.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: ignoring unreadable {RAW_CATALOG_FILE}: {exc}", file=sys.stderr)

    try:
        new = fetch_catalog(args.timeout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"catalog fetch failed: {exc}", file=sys.stderr)
        return 1

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
