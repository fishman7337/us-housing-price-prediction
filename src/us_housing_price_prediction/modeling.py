"""Model construction, training, evaluation, and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from us_housing_price_prediction.config import (
    CV_SPLITS,
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from us_housing_price_prediction.data import load_housing_data, make_train_test_split
from us_housing_price_prediction.features import HousingFeatureEngineer
from us_housing_price_prediction.statistics import (
    paired_error_t_test,
    run_feature_significance_tests,
)


@dataclass(frozen=True)
class TrainingResult:
    """Serializable output from a training run."""

    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    cross_validation: dict[str, float]
    residual_test: dict[str, object]
    feature_significance: list[dict[str, object]]
    rows: int
    target: str = TARGET_COLUMN

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the training result."""
        return asdict(self)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing that is fitted only inside the training pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_regressor(random_state: int = RANDOM_STATE) -> VotingRegressor:
    """Build the ensemble regressor used for the production pipeline."""
    return VotingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=10.0)),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(
                    n_estimators=160,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state,
                ),
            ),
        ],
        n_jobs=-1,
    )


def build_pipeline(
    *,
    regressor: Any | None = None,
    drop_source_features: bool = True,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a leakage-safe model pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineering", HousingFeatureEngineer(drop_source_features)),
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                regressor if regressor is not None else build_regressor(random_state),
            ),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Build a median baseline with the same preprocessing contract."""
    return build_pipeline(regressor=DummyRegressor(strategy="median"))


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return common regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Run reproducible cross-validation and summarize performance."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
            "neg_rmse": "neg_root_mean_squared_error",
        },
        n_jobs=-1,
    )

    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_neg_mae"].mean()),
        "rmse_mean": float(-scores["test_neg_rmse"].mean()),
    }


def train_and_evaluate(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, TrainingResult]:
    """Train the model and return the fitted pipeline plus evaluation artifacts."""
    df = load_housing_data(data_path)
    X_train, X_test, y_train, y_test = make_train_test_split(
        df,
        random_state=random_state,
    )

    pipeline = build_pipeline(random_state=random_state)
    baseline_pipeline = build_baseline_pipeline()

    pipeline.fit(X_train, y_train)
    baseline_pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    baseline_predictions = baseline_pipeline.predict(X_test)

    metrics = evaluate_regression(y_test, predictions)
    baseline_metrics = evaluate_regression(y_test, baseline_predictions)
    cv_summary = cross_validate_pipeline(
        build_pipeline(random_state=random_state),
        pd.concat([X_train, X_test], axis=0),
        pd.concat([y_train, y_test], axis=0),
        random_state=random_state,
    )

    significance = run_feature_significance_tests(df)
    result = TrainingResult(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        cross_validation=cv_summary,
        residual_test=paired_error_t_test(
            y_test,
            predictions,
            baseline_predictions,
        ),
        feature_significance=significance.to_dict(orient="records"),
        rows=len(df),
    )
    return pipeline, result


def save_model(model: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a fitted model pipeline."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a persisted model pipeline."""
    return joblib.load(path)


def save_metrics(result: TrainingResult, path: str | Path = DEFAULT_METRICS_PATH) -> Path:
    """Write training metrics and diagnostics to JSON."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return metrics_path
