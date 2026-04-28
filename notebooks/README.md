# Notebook Guide

The original all-in-one analysis notebook has been split into smaller notebooks so each stage is easier to open, review, and rerun.

## Order

1. `00_project_overview.ipynb` - coursework context and workflow map.
2. `01_data_validation_and_eda.ipynb` - schema validation, data quality checks, EDA, and feature significance tests.
3. `02_feature_engineering.ipynb` - engineered features and redundant source feature removal.
4. `03_model_training_and_evaluation.ipynb` - model training, baseline comparison, cross-validation, p-values, and saved metrics.
5. `04_prediction_demo.ipynb` - loading the saved model and running a single prediction.

## Design

Reusable code belongs in `src/us_housing_price_prediction`. Notebooks should stay narrative and lightweight.
