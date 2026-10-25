#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# The openclaw CLI only exists on the host and needs Homebrew's node. A guard
# refusal keeps the previous catalog; the rebuild below still runs.
PATH="/home/linuxbrew/.linuxbrew/bin:$PATH" python3 scripts/refresh_catalog.py \
  || echo "catalog refresh skipped; keeping previous catalog" >&2

docker compose -f deploy/docker-compose.yml build nexus-router
docker compose -f deploy/docker-compose.yml up -d nexus-router
sleep 4
docker exec nexus-router-1 python -m src.fetch_benchmarks --sources aa --force
docker exec nexus-router-1 python -m src.generate_registry
# The server loads the registry once at startup.
docker restart nexus-router-1

echo "done: catalog refreshed, router rebuilt, benchmarks refreshed, registry regenerated"
