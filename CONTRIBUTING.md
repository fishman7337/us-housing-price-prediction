# Contributing

Thank you for improving this project.

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
python -m pytest
```

## Workflow

1. Create a focused branch.
2. Keep changes scoped to one concern.
3. Add or update tests for behavioral changes.
4. Run `python -m ruff check .` and `python -m pytest`.
5. Include validation evidence in the pull request.

## Data and Model Changes

- Keep raw data immutable under `data/raw/`.
- Do not commit trained binary model artifacts unless the project explicitly decides to version them.
- Record important metrics in `reports/metrics.json`.
- Update the model card when assumptions, performance, or limitations change.

## Commit Style

Use short imperative messages, for example:

```text
Add leakage-safe training pipeline
```
