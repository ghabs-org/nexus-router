#!/usr/bin/env bash
set -euo pipefail

OK=1

echo "== OpenClaw gateway =="
GW_STATUS="$(openclaw gateway status 2>/dev/null || true)"
echo "$GW_STATUS" | sed -n '1,18p'
if ! echo "$GW_STATUS" | grep -q 'RPC probe: ok'; then
  echo "[fail] gateway probe not ok"
  OK=0
fi

echo
echo "== Active gateway command =="
CMD_LINE="$(echo "$GW_STATUS" | awk -F': ' '/^Command:/ {print $2}')"
echo "$CMD_LINE"
if [[ "$CMD_LINE" != *"openclaw"* ]]; then
  echo "[warn] cannot determine gateway command"
fi

echo
echo "== Nexus router plugin markers =="
PLUGIN_FILE="$HOME/.openclaw/extensions/nexus-router/index.ts"
if [[ ! -f "$PLUGIN_FILE" ]]; then
  echo "[fail] plugin file not found: $PLUGIN_FILE"
  OK=0
else
  for marker in "isSlashCommand" "CLASSIFIER_SENTINEL" "STICKY_ROUTE_MODES"; do
    if grep -q "$marker" "$PLUGIN_FILE"; then
      echo "[ok] marker: $marker"
    else
      echo "[fail] missing marker: $marker"
      OK=0
    fi
  done
fi

echo
echo "== Usage footer model marker (node_modules runtime) =="
RUNTIME_FILE="$(ls $HOME/.local/lib/node_modules/openclaw/dist/agent-runner.runtime-*.js 2>/dev/null | head -1 || true)"
if [[ -z "$RUNTIME_FILE" ]]; then
  echo "[fail] runtime file not found"
  OK=0
else
  echo "$RUNTIME_FILE"
  if grep -q '· model \${modelLabel}' "$RUNTIME_FILE"; then
    echo "[ok] footer model marker present"
  else
    echo "[fail] footer model marker missing"
    OK=0
  fi
fi

echo
echo "== nexus-router container =="
if docker ps --filter name=nexus-router-1 --format '{{.Names}} {{.Status}}' | grep -q '^nexus-router-1 '; then
  docker ps --filter name=nexus-router-1 --format '{{.Names}} {{.Status}}'
else
  echo "[fail] nexus-router-1 container not running"
  OK=0
fi

echo
if [[ "$OK" -eq 1 ]]; then
  echo "VERIFY: PASS"
  exit 0
else
  echo "VERIFY: FAIL"
  exit 1
fi
