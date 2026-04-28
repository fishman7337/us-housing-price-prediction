# CA1 Report Summary

## Assignment Context

This project was completed for Singapore Polytechnic, School of Computing, Diploma in Applied AI & Analytics. It belongs to AI & Machine Learning (`ST1511`), CA1 Part B, AY24/25 Year 1 Semester 2.

- Student: Goh Kun Ming
- Lecturer: Adjunct Lecturer Tai Hock Lin (Andy)

## Objective

The objective is to create a supervised regression model that predicts US housing prices from structured property information. The project demonstrates data analysis, cleaning, feature engineering, model training, evaluation, interpretation, and deployment considerations.

## Dataset

The canonical dataset is stored at `data/raw/housing_price_data.csv`. It contains 545 rows and the following fields:

- house identifier;
- city;
- house area in square meters;
- number of bedrooms;
- number of toilets;
- number of stories;
- renovation status;
- price in USD.

The original file extension was `.xls`, but the file contents were CSV-formatted text. It has been normalized to `.csv` for correctness.

## Data Understanding

The dataset includes numerical and categorical variables. `house_id` is an identifier and is excluded from training because it does not describe a property characteristic. `price_usd` is the target variable. City and renovation status are categorical predictors and are encoded inside the model pipeline.

## Data Cleaning

The production workflow validates:

- required columns;
- missing values;
- duplicate rows and duplicate IDs;
- positive numerical values;
- allowed renovation status categories.

Cleaning is handled in `src/us_housing_price_prediction/data.py` so notebook analysis and CLI workflows use the same rules.

## Feature Engineering

The project creates additional property-level features:

- `area_per_bedroom`;
- `toilet_to_bedroom_ratio`;
- `total_rooms`;
- `stories_house_area_interaction`.

After these engineered features are created, redundant source count fields are removed by default. This addresses the feature engineering issue where old source fields can remain after stronger derived features are introduced, increasing repeated signal and making diagnostics harder to interpret.

## Statistical Testing

The workflow includes p-value based diagnostics:

- Pearson correlation tests for numerical features against price;
- one-way ANOVA tests for categorical features against price;
- paired t-test comparing model absolute errors against the baseline absolute errors.

These tests support interpretation and model comparison. They should not be treated as proof of causation.

## Modeling

The production model is a scikit-learn pipeline containing:

- feature engineering;
- median imputation for numerical fields;
- most-frequent imputation for categorical fields;
- scaling for numerical fields;
- one-hot encoding for categorical fields;
- a voting regressor combining Ridge, Random Forest, and Gradient Boosting regressors.

The baseline is a median `DummyRegressor`.

## Evaluation

The model is evaluated with:

- R2;
- MAE;
- MSE;
- RMSE;
- MAPE;
- explained variance;
- cross-validation;
- baseline comparison.

The latest metrics are stored in `reports/metrics.json`.

## Deployment and Practical Implications

The model is suitable for coursework, learning, and prototype analytics. It is not suitable as the sole basis for lending, taxation, insurance, or legal valuation. Real deployment would require stronger data provenance, monitoring, human review, privacy checks, and fairness review.

## Notebook Strategy

Two notebook formats are intentionally maintained:

- `notebooks/original_regression_analysis.ipynb` keeps the full academic write-up and original reasoning style.
- `notebooks/00` to `04` provide smaller executable notebooks that reuse production code while keeping original-style academic markdown explanations.

This allows the project to satisfy academic documentation expectations while also meeting modern maintainability expectations.
