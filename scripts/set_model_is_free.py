#!/usr/bin/env python3
"""Set or clear is_free metadata for a router model without restarting services.

Examples:
  python scripts/set_model_is_free.py github-copilot/gpt-5-mini true
  python scripts/set_model_is_free.py openrouter/anthropic/claude-sonnet-4.5 false
  python scripts/set_model_is_free.py openrouter/openai/gpt-4o clear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import ensure_schema, set_model_is_free  # noqa: E402


def parse_value(raw: str):
    value = str(raw or "").strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    if value in {"clear", "null", "none", "unset"}:
        return None
    raise ValueError("value must be true|false|clear")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set model_metadata.is_free for a model")
    parser.add_argument("model", help="Full model id, e.g. github-copilot/gpt-5-mini")
    parser.add_argument("value", help="true | false | clear")
    args = parser.parse_args()

    ensure_schema()
    value = parse_value(args.value)
    set_model_is_free(args.model, value)
    print(f"ok model={args.model} is_free={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
