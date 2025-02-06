#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

uv run alembic upgrade head
uv run gunicorn -c gunicorn.conf.py nilai_api.__main__:app
