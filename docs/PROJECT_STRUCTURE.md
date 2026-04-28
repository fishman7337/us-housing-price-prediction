# Project Structure

```text
data/raw/                       Immutable source data
docs/                           Academic, project, data, model, and MLOps documentation
models/                         Local trained model artifacts, ignored by Git
notebooks/                      Numbered workflow notebooks
reports/                        Metrics, figures, and presentation artifacts
src/us_housing_price_prediction Reusable Python package
tests/                          Unit and integration tests
```

The production workflow should use the package in `src/`. Notebooks remain useful for narrative analysis, but reusable logic should not live only inside notebooks.
