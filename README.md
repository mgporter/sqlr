# sqlrunner

## Setup

With `uv` (creates and syncs `.venv` for you):

```bash
uv sync --extra dev
```

Or with a manually activated venv:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # omit [dev] to skip pytest
```

## Running the tests

```bash
uv run pytest                  # no activation needed
```

Or, with the venv activated:

```bash
pytest
```

Useful variants: `pytest tests/test_sql_analysis.py` for one file, `pytest -k lineage` to
filter by name, `pytest -q` for quiet output.

See [tests/README.md](tests/README.md) for the rule on where test fixtures live.

## Running the CLI

```bash
uv run sqlrunner --project-dir <path>       # -v for debug logging
```
