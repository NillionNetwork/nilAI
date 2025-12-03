# Repository Guidelines

## Project Structure & Module Organization
Source code lives under `src/nilai_py/`, with `client.py` exposing the OpenAI-compatible client, `server.py` managing delegation tokens, and `niltypes.py` centralizing typed models. Tests reside in `tests/` and mirror runtime modules (`test_server.py`, `test_nilai_openai.py`). Sample workflows sit in `examples/`, reusable prompt templates in `stored_prompts/`, and build artifacts land in `dist/`. Keep configuration in `.env` files or the `keys/` directory; do not commit secrets.

## Build, Test, and Development Commands
- `uv sync` — install runtime dependencies declared in `pyproject.toml`.
- `uv sync --group dev` — install tooling for linting and tests.
- `uv run pytest` — execute the full test suite.
- `uv run pytest tests/test_server.py -v` — target a specific module with verbose output.
- `uv run pytest --cov=nilai_py --cov-report=term-missing` — check coverage before submitting changes.
- `uv run ruff check` / `uv run ruff format` — lint and auto-format to project standards.
- `uv build` — produce distributable wheels and source archives.

## Coding Style & Naming Conventions
Stick to Python 3.12 standards with 4-space indentation, type hints, and explicit docstrings where behavior is non-trivial. Use snake_case for functions and variables, PascalCase for classes, and UPPER_CASE for module-level constants. Align imports per Ruff ordering, keep modules focused, and co-locate helper functions with their primary caller to ease review.

## Testing Guidelines
Write tests with `pytest`, placing files in `tests/` using the `test_*.py` pattern and descriptive method names. Mock external network calls so suites run offline. Maintain or improve the current ~70% coverage by running `uv run pytest --cov=nilai_py --cov-report=term-missing` and addressing gaps surfaced in `term-missing` output. When adding integrations, extend `test_nilai_openai.py`; for delegation server logic, update `test_server.py`.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes observed in the history (`feat:`, `fix:`, `docs:`, etc.) and keep subjects under 72 characters. Each pull request should: 1) summarize intent and key changes, 2) link related issues or tickets, 3) note test evidence (command output or coverage delta), and 4) include screenshots or logs when altering user-visible behavior. Request review only after lint and tests pass locally.

## Security & Configuration Tips
Load credentials via environment variables or `.env` files and treat `keys/` artifacts as local-only. When sharing examples, redact API keys and private keys. Default endpoints in the SDK point to sandbox infrastructure; document any production overrides in your PR description.
