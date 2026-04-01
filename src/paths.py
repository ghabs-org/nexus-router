from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_ROOT = Path(os.environ.get("NEXUS_ROUTER_STATE_ROOT") or (Path.home() / ".local/state/nexus-router"))
DATA_ROOT = Path(os.environ.get("NEXUS_ROUTER_DATA_ROOT") or (STATE_ROOT / "data"))
GENERATED_ROOT = Path(os.environ.get("NEXUS_ROUTER_GENERATED_ROOT") or (STATE_ROOT / "generated"))
CACHE_ROOT = Path(os.environ.get("NEXUS_ROUTER_CACHE_ROOT") or (STATE_ROOT / "cache"))
POLICIES_ROOT = Path(os.environ.get("NEXUS_ROUTER_POLICIES_ROOT") or (ROOT / "policies"))
ARTIFACTS_ROOT = Path(os.environ.get("NEXUS_ROUTER_ARTIFACTS_ROOT") or (STATE_ROOT / "artifacts"))

ROUTER_DB_PATH = Path(os.environ.get("NEXUS_ROUTER_DB_PATH") or (DATA_ROOT / "router.sqlite"))
RAW_CATALOG_FILE = Path(os.environ.get("NEXUS_ROUTER_RAW_CATALOG_FILE") or (GENERATED_ROOT / "openclaw-models.json"))
REGISTRY_FILE = Path(os.environ.get("NEXUS_ROUTER_REGISTRY_FILE") or (GENERATED_ROOT / "models.json"))
BENCHMARK_CACHE_DIR = Path(os.environ.get("NEXUS_ROUTER_BENCHMARK_CACHE_DIR") or (CACHE_ROOT / "benchmark"))
LOCAL_CLASSIFIER_DIR = Path(os.environ.get("NEXUS_ROUTER_LOCAL_CLASSIFIER_DIR") or (ARTIFACTS_ROOT / "router-classifier" / "onnx"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
