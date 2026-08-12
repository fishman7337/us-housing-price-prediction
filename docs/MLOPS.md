# MLOps Guide

## Reproducibility

- Random state is centralized in `src/us_housing_price_prediction/config.py`.
- Data is loaded from a canonical CSV path.
- Preprocessing is fitted inside the model pipeline to avoid leakage.
- Metrics are written to JSON for review and comparison.

## Local Checks

```bash
python -m ruff check .
python -m pytest
python -m us_housing_price_prediction validate-data
python -m us_housing_price_prediction train --min-r2 0.45
```

When the optional quality gate fails, the command exits without persisting the
candidate model or metrics, preventing a failed candidate from replacing prior artifacts.

## CI

GitHub Actions runs linting, tests, data validation, and a training smoke test on Python 3.10, 3.11, and 3.12.

## Experiment Tracking

`dvc.yaml` defines validation and training stages. If DVC is installed, run:

```bash
dvc repro
```

Keep heavy model binaries out of Git unless the repository explicitly adopts model artifact versioning.

## Deployment Readiness

Before production use, add:

- model registry integration;
- drift monitoring;
- prediction API or batch scoring job;
- fairness and bias review;
- rollback plan;
- data provenance and privacy review.
