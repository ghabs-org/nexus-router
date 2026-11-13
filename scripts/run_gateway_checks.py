#!/usr/bin/env python3
"""Run gateway checks with isolated state and optional real HTTP over socket pairs."""

import argparse
import http.server
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socketpair", action="store_true", help="Use serialized HTTP without binding TCP listeners")
    parser.add_argument("--all", action="store_true", help="Run the complete router suite")
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "tests")]
    with tempfile.TemporaryDirectory(prefix="nexus-gateway-tests-") as state:
        os.environ["NEXUS_ROUTER_STATE_ROOT"] = state
        os.environ["NEXUS_ROUTER_DB_PATH"] = str(Path(state) / "data/router.sqlite")
        registry = Path(state) / "generated/models.json"
        registry.parent.mkdir(parents=True)
        committed = root / "generated/models.json"
        if committed.exists():
            shutil.copyfile(committed, registry)
        os.environ["NEXUS_ROUTER_REGISTRY_FILE"] = str(registry)
        import pytest
        patcher = pytest.MonkeyPatch()
        try:
            if args.socketpair or os.environ.get("NEXUS_TEST_SOCKETPAIR") == "1":
                from cli_socketpair import install
                os.environ["NEXUS_TEST_SOCKETPAIR"] = "1"
                patcher.setattr(http.server, "ThreadingHTTPServer", install(patcher))
            targets = [str(root / "tests")] if args.all else [str(root / "tests" / name) for name in (
                "test_gateway.py", "test_gateway_controls.py", "test_gateway_budget.py", "test_gateway_classification.py", "test_gateway_auth.py", "test_gateway_clients.py", "test_cli_proxy.py", "test_cli_proxy_v2.py",
                "test_cli_adapters.py", "test_cli_classifier.py")]
            return pytest.main(["-q", *targets, *extra])
        finally:
            patcher.undo()


if __name__ == "__main__":
    raise SystemExit(main())
