# Project Structure

```text
data/raw/                       Immutable source data
docs/                           Academic, project, data, model, and MLOps documentation
models/                         Local trained model artifacts, ignored by Git
notebooks/                      Original academic notebook plus numbered workflow notebooks
reports/                        Metrics, figures, and presentation artifacts
src/us_housing_price_prediction Reusable Python package
tests/                          Unit and integration tests
```

The production workflow should use the package in `src/`. The original notebook preserves the CA1 narrative, while the numbered notebooks provide smaller entry points for rerunning the workflow. Reusable logic should not live only inside notebooks.
