#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_APP_DIR="$PROJECT_ROOT/web_app"
VENV_PYTHON="$WEB_APP_DIR/venv/bin/python"
TARGET_PORT=8000
PORTS_TO_CLEAN=(8000 8001 5000)

if [[ ! -x "$VENV_PYTHON" ]]; then
  VENV_PYTHON="$WEB_APP_DIR/venv/bin/python3"
fi

cd "$WEB_APP_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export MODEL_TYPE="${MODEL_TYPE:-resnet}"
export PORT="${PORT:-$TARGET_PORT}"

cleanup_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping process on port $port"
    kill -9 $pids 2>/dev/null || true
  fi
}

for port in "${PORTS_TO_CLEAN[@]}"; do
  cleanup_port "$port"
done

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtual environment interpreter at $WEB_APP_DIR/venv/bin/python"
  exit 1
fi

exec "$VENV_PYTHON" app.py