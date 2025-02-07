#!/bin/bash

set -euo pipefail

uv run alembic upgrade head
uv run gunicorn -c gunicorn.conf.py nilai_api.__main__:app
