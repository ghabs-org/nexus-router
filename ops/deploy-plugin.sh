#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="$ROOT_DIR/plugin"
EXT_DIR="$HOME/.openclaw/extensions/nexus-router"

echo "[deploy] repo: $ROOT_DIR"
echo "[deploy] building plugin..."
(
  cd "$PLUGIN_DIR"
  npm run build >/tmp/nexus-router-plugin-build.log 2>&1 || {
    echo "[deploy] plugin build failed. tail:" >&2
    tail -50 /tmp/nexus-router-plugin-build.log >&2
    exit 1
  }
)

echo "[deploy] syncing plugin to $EXT_DIR"
mkdir -p "$EXT_DIR"
cp "$PLUGIN_DIR/src/index.ts" "$EXT_DIR/index.ts"
if [[ -f "$PLUGIN_DIR/openclaw.plugin.json" ]]; then
  cp "$PLUGIN_DIR/openclaw.plugin.json" "$EXT_DIR/openclaw.plugin.json"
else
  echo "[deploy] manifest missing at $PLUGIN_DIR/openclaw.plugin.json; keeping existing installed manifest"
fi
cp "$PLUGIN_DIR/package.json" "$EXT_DIR/package.json"

echo "[deploy] restarting nexus-router container"
(
  cd "$ROOT_DIR"
  docker compose -f deploy/docker-compose.yml up -d nexus-router >/tmp/nexus-router-docker-up.log 2>&1 || {
    echo "[deploy] docker compose up failed. tail:" >&2
    tail -50 /tmp/nexus-router-docker-up.log >&2
    exit 1
  }
)

echo "[deploy] restarting openclaw gateway"
if systemctl --user restart openclaw-gateway; then
  :
else
  echo "[deploy] systemd restart returned non-zero, checking status..."
fi
sleep 2

openclaw gateway status | sed -n '1,18p'

echo "[deploy] done"
