# Quality Assurance

## Current Quality Controls

- Source code is packaged under `src/us_housing_price_prediction`.
- Reusable logic is tested with pytest.
- CI runs linting, tests, data validation, and a training smoke test.
- The model pipeline prevents target and identifier leakage.
- Feature engineering runs inside the pipeline.
- Data validation enforces schema and domain constraints.
- Model metrics are written to `reports/metrics.json`.
- The original notebook is preserved, while modular notebooks support lighter review.

## Local Validation Commands

```bash
python -m ruff check .
python -m pytest
python -m us_housing_price_prediction validate-data
python -m us_housing_price_prediction train --min-r2 0.45
```

## Documentation Standard

Documentation should explain both what was done and why it was done. For this project, that means keeping:

- academic context and assignment details;
- data dictionary and assumptions;
- feature engineering rationale;
- statistical testing explanation;
- baseline and final model comparison;
- deployment limitations and ethical considerations;
- reproducible commands.

## Residual Risks

- The dataset is small, so real-world generalization is limited.
- The city categories are limited to the available data.
- The model does not include current market conditions, interest rates, neighborhood-level features, or time effects.
- The full original notebook is large because it preserves historical output and narrative.
