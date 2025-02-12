#!/bin/bash

set -euo pipefail

uv run alembic upgrade head

# Setup the multiproc directory for all guvicon workers to expose metrics together
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-metrics
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

exec uv run gunicorn -c gunicorn.conf.py nilai_api.__main__:app
