# US Housing Price Prediction MLOps

Production-ready machine learning project for predicting US housing prices from structured property features. The original notebook analysis is preserved, and the reusable workflow now lives in a tested Python package with validation, feature engineering, statistical tests, model training, metrics, and CI.

## What Changed

- Normalized the mislabeled data file from `housing_price_data.xls` to `data/raw/housing_price_data.csv`.
- Moved original coursework artifacts into `notebooks/` and `reports/presentations/`.
- Added a reusable `src/us_housing_price_prediction` package.
- Added leakage-safe preprocessing through a scikit-learn `Pipeline`.
- Added feature engineering that drops redundant source fields after deriving richer features.
- Added p-value diagnostics for raw features and a paired t-test against the baseline model.
- Added pytest tests, packaging metadata, CI, DVC metadata, docs, and contribution guidelines.

## Project Structure

```text
.
|-- .github/workflows/ci.yml
|-- configs/
|-- data/raw/housing_price_data.csv
|-- docs/
|-- models/
|-- notebooks/original_regression_analysis.ipynb
|-- reports/
|-- src/us_housing_price_prediction/
|-- tests/
|-- dvc.yaml
|-- params.yaml
|-- pyproject.toml
`-- README.md
```

## Quick Start

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
python -m pytest
python -m us_housing_price_prediction validate-data
python -m us_housing_price_prediction train
```

After training, the model is written to `models/housing-price-regressor.joblib` and metrics are written to `reports/metrics.json`.

## Prediction Example

```bash
python -m us_housing_price_prediction predict \
  --city Chicago \
  --house-area-sqm 742 \
  --bedrooms 4 \
  --toilets 2 \
  --stories 3 \
  --renovation-status furnished
```

## Modeling Approach

The pipeline uses:

- canonical data validation and schema checks;
- derived housing features including total rooms, area per bedroom, toilet-bedroom ratio, and story-area interaction;
- redundant feature removal after engineering for cleaner model diagnostics;
- median imputation, scaling, and one-hot encoding fitted only on training folds;
- a voting regressor built from Ridge, Random Forest, and Gradient Boosting models;
- a median `DummyRegressor` baseline;
- cross-validation and test-set evaluation.

## Metrics

The training command reports:

- R2
- MAE
- MSE
- RMSE
- MAPE
- explained variance
- cross-validation summary
- paired t-test of model absolute error versus baseline absolute error
- p-value tests for feature-target relationships

## Documentation

- [MLOps Guide](docs/MLOPS.md)
- [Data Card](docs/DATA_CARD.md)
- [Model Card](docs/MODEL_CARD.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
