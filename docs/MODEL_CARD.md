# Model Card

## Model

Voting regressor combining Ridge, Random Forest, and Gradient Boosting regressors.

## Project Context

This model was developed for Singapore Polytechnic's AI & Machine Learning (`ST1511`) CA1 Part B under the Diploma in Applied AI & Analytics.

## Intended Use

Estimate housing prices from structured property attributes for experimentation, coursework, and prototype analytics.

## Not Intended For

- final lending decisions;
- tax assessment;
- insurance underwriting;
- legal valuation;
- automated decisions without human review.

## Features

Raw features are validated, engineered, and transformed inside a scikit-learn pipeline. Identifier and target columns are never passed into the model.

Engineered features include:

- area per bedroom;
- toilet-to-bedroom ratio;
- total rooms;
- story-area interaction.

Bedroom, toilet, and story source columns are removed after engineering by default to reduce redundant signals.

## Evaluation

Use:

```bash
python -m us_housing_price_prediction train
```

The command writes test-set metrics, cross-validation metrics, baseline comparison, p-value tests, and paired error t-test results to `reports/metrics.json`.

## Risks

The dataset is small, so estimates may not generalize to current housing markets. City categories are limited, and the model does not include interest rates, neighborhood-level factors, school districts, crime, seasonality, or broader economic indicators.
