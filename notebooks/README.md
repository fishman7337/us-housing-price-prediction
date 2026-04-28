# Notebook Guide

The original all-in-one analysis notebook is maintained for academic traceability, and the workflow has also been split into smaller notebooks so each stage is easier to open, review, and rerun. The modular notebooks intentionally use the same academic markdown style as the original notebook, with section headings, rationale, interpretation, and conclusion notes around the code.

## Original Academic Notebook

`original_regression_analysis.ipynb` is the full CA1 Part B narrative notebook. Keep it when preserving the submission record, lecturer-facing explanation, and original step-by-step reasoning.

## Modular Workflow Order

1. `00_project_overview.ipynb` - coursework context and workflow map.
2. `01_data_validation_and_eda.ipynb` - schema validation, data quality checks, EDA, and feature significance tests.
3. `02_feature_engineering.ipynb` - engineered features and redundant source feature removal.
4. `03_model_training_and_evaluation.ipynb` - model training, baseline comparison, cross-validation, p-values, and saved metrics.
5. `04_prediction_demo.ipynb` - loading the saved model and running a single prediction.

## Design

Reusable code belongs in `src/us_housing_price_prediction`. The modular notebooks should keep the original report-style explanation while using reusable package code, and the original notebook remains the complete historical academic write-up.
