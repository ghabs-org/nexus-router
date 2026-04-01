#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

docker compose -f deploy/docker-compose.yml build nexus-router
docker compose -f deploy/docker-compose.yml up -d nexus-router
sleep 4
docker exec nexus-router-1 python -m src.fetch_benchmarks --sources aa --force
docker exec nexus-router-1 python -m src.generate_registry

echo "done: router rebuilt, benchmarks refreshed, registry regenerated"
