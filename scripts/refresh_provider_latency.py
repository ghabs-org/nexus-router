#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.health import record_observation

PROVIDERS = [
    "openai-codex",
    "github-copilot",
    "google-gemini-cli",
]


def probe(provider: str) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        ["openclaw", "models", "status", "--probe-provider", provider, "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    if result.returncode == 0:
        record_observation(
            provider=provider,
            auth_status="ok",
            latency_ms=latency_ms,
        )
        return {"provider": provider, "ok": True, "latency_ms": latency_ms}

    combined = f"{result.stdout}\n{result.stderr}".lower()
    auth_status = "expired" if any(t in combined for t in ["401", "403", "unauthorized", "forbidden"]) else "unknown"
    error_type = "timeout" if "timeout" in combined else "unknown"
    record_observation(
        provider=provider,
        auth_status=auth_status,
        error_type=error_type,
        latency_ms=latency_ms,
    )
    return {"provider": provider, "ok": False, "latency_ms": latency_ms, "error": result.stderr.strip()}


if __name__ == "__main__":
    print(json.dumps({"results": [probe(p) for p in PROVIDERS]}, indent=2))
