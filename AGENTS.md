# Repository Guidelines

## Project Structure & Module Organization
The workspace uses `uv` to link several Python packages. Core web APIs live in `nilai-api/src/nilai_api`, model runtimes are under `nilai-models/src/nilai_models`, and shared utilities sit in `packages/nilai-common/src/nilai_common`. Authentication helpers are split between `nilai-auth/nilai-auth-server` and `nilai-auth/nilai-auth-client`. Integration data and infrastructure assets reside in `docker/`, `devops/`, and `scripts/`, while end-to-end, integration, and unit suites are grouped under `tests/`.

## Build, Test, and Development Commands
Run `uv sync` from the repo root to install workspace dependencies. Start the FastAPI gateway in hot-reload mode with `uv run fastapi dev nilai-api/src/nilai_api/__main__.py`, or launch a production-like instance with `uv run fastapi run ... --port 8080`. Model services follow the same pattern, e.g. `uv run fastapi dev nilai-models/src/nilai_models/models/llama_1b_cpu/__init__.py`. For full-stack experimentation, compose services with `python3 scripts/docker-composer.py --dev -o development-compose.yml` and `docker compose -f development-compose.yml up -d`.

## Coding Style & Naming Conventions
Target Python 3.12 with four-space indentation and descriptive snake_case identifiers. Before submitting changes, execute `uv run ruff check .`, `uv run black .`, and `uv run isort .` to align with the project’s linting stack; type hints should satisfy `uv run pyright`. Shared abstractions belong in `nilai_common`, API routers in `nilai_api/api`, and model adapters in dedicated folders within `nilai_models/models`.

## Testing Guidelines
The project standardizes on `pytest`; run targeted suites with commands like `uv run pytest tests/unit` or `uv run pytest tests/e2e -m smoke`. Place new tests beside the code they validate, naming files `test_<feature>.py` and parametrizing scenarios where possible. When introducing cross-service flows, add coverage in `tests/integration` or `tests/functional_tests`, and capture fixture updates in `tests/conftest.py`.

## Commit & Pull Request Guidelines
Commit messages follow Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`, etc.) as seen in `git log`. Group related changes per commit, reference issue IDs when available, and keep bodies focused on the observable behavior change. Pull requests should summarize intent, list manual or automated test results, and highlight any configuration updates (new env vars, Docker tags, or migrations). Include screenshots or logs only when they clarify the change.

## Security & Configuration Tips
Never commit real secrets; use `.env.sample` as the template and keep private keys under the packaged `private_key.key.lock` mechanism. When working locally, prefer ephemeral Hugging Face tokens and rotate them before sharing demo environments. Review Dockerfiles for pinned bases before pushing production builds.
