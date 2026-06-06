# Contributing

All pull requests, large or small, from anyone are welcome!

## Table of Contents

- [Contributing](#contributing)
    - [Table of Contents](#table-of-contents)
    - [Environment Management](#environment-management)
        - [First time setup](#first-time-setup)
    - [Code Quality Tools](#code-quality-tools)
    - [Testing](#testing)
        - [Writing Tests](#writing-tests)
    - [Git Workflow](#git-workflow)
    - [Code Style and System Design](#code-style-and-system-design)

## Environment Management

[uv](https://github.com/astral-sh/uv/) is the suggested python environment management tool.

### First time setup

- Install uv, as described [in the uv installation guide](https://github.com/astral-sh/uv/#installation).
- `uv python install 3.12` -- This project requires Python 3.12+.
- `uv python pin 3.12` -- Pin the default version.
- `uv self update`
- `uv sync --all-groups`
- The `.venv/` directory will now be created with all project, development and documentation dependencies installed.
    - Be sure to point your IDE to the python binary appropriate for your OS in this directory.
- `pre-commit install` -- Set up pre-commit hooks for automatic formatting and linting on commit.

## Code Quality Tools

- [**tox**](https://tox.wiki/)
    - Creates virtual environments for CI or local pytest runs.
        - Note that the CI does not current execute calls to the production API by default.
    - Run `tox list` or see `tox.ini` for more info
- [**pre-commit**](https://pre-commit.com/)
    - Creates virtual environments for formatting and linting tools
    - Run `pre-commit run --all-files` or see `.pre-commit-config.yaml` for more info.

> Note: Many of the tools below are run by `pre-commit` automatically, but can also be run manually if desired.

- [**ruff**](https://github.com/astral-sh/ruff)
    - Provides both formatting and linting
    - Format: `ruff format .`
    - Lint: `ruff check . --fix`
    - Combined: `ruff format . && ruff check . --fix`
    - See `pyproject.toml` for the rules used.
    - See all rules (but not necessarily used in the project) [available in ruff here](https://beta.ruff.rs/docs/rules/).
- [**pyrefly**](https://pyrefly.org/)
    - Type checker: `pyrefly check .`
- [**pyright**](https://github.com/microsoft/pyright)
    - Shipped with vscode by default (via the python extension `ms-python.vscode-pylance`)
    - Suggested settings:
        - `"python.analysis.typeCheckingMode": "off"`
            - The pylance extension has certain opinionated type checking assertions which clash with other type checkers.
            - For example, overriding an optional field to be non-optional is considered by pylance to be a type error due to the field being invariant and the parent class potentially settings it to `None`. However, by convention in the SDK, this is a forbidden pattern.
        - `"python.analysis.languageServerMode": "full"`
        - `"python.testing.pytestEnabled": true`
- [**tach**](https://github.com/gauge-sh/tach)
    - Enforces internal namespace dependency constraints. This helps avoid circular dependencies and helps ensure implementations are in a logical place.

## Testing

Tests require the `AI_HORDE_TESTING=True` environment variable to be set. This prevents interference from user-specific environment variables.

```bash
# Set the required env var and run tests
export AI_HORDE_TESTING=True
pytest

# Run with coverage
pytest --cov=horde_model_reference --cov-report=html

# Run a specific test file
pytest tests/test_model_reference_manager.py
```

### Writing Tests

- **Singleton reset:** Use the `restore_manager_singleton` fixture to ensure `ModelReferenceManager` is restored to its pre-test state after your test. This is critical when testing with different manager configurations.
- **Temp directories:** Use the `primary_base` fixture for tests that need an isolated temp directory for PRIMARY mode file operations.
- **Test ordering:** Tests in `test_scripts`, `test_convert_legacy_database`, and `test_model_reference_manager` run in a fixed order (defined in `conftest.py`). Other tests run in arbitrary order.

## Git Workflow

1. Fork the repository and create a feature branch from `main`
2. Make your changes
3. Run the quality checks: `ruff format . && ruff check . --fix && pyrefly check .`
4. Run the tests: `AI_HORDE_TESTING=True pytest`
5. Commit with a clear message and open a pull request

## Code Style and System Design

- See the [python haidra style guide](docs/haidra-assets/docs/meta/python.md) for standards on code style, system design, testing, and documentation.
