#!/bin/sh
set -e
PORT="${PORT:-8000}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
echo "docker-entrypoint: binding 0.0.0.0:${PORT}" >&2
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
