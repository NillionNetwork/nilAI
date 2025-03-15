#!/bin/bash

set -euo pipefail

# Setup the multiproc directory for all guvicon workers to expose metrics together
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-metrics
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

exec gunicorn -c gunicorn.conf.py gpuverifier_api.__main__:app
