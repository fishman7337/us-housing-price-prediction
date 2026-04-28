# Experiments

## Baseline

The baseline is a median `DummyRegressor`. It gives a minimum acceptable comparison point for the trained model.

## Candidate Model

The current candidate is a voting regressor:

- Ridge for a stable linear component;
- Random Forest for nonlinear interactions;
- Gradient Boosting for residual pattern learning.

## Statistical Tests

The training command records:

- Pearson correlation p-values for numeric raw features;
- one-way ANOVA p-values for categorical raw features;
- paired t-test comparing candidate absolute errors with baseline absolute errors.

These tests support interpretation. They do not prove causation.
