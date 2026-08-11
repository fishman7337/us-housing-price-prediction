"""Statistical tests used to interpret model and feature behavior."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from us_housing_price_prediction.config import (
    CATEGORICAL_SOURCE_COLUMNS,
    FEATURE_SOURCE_COLUMNS,
    TARGET_COLUMN,
)
from us_housing_price_prediction.data import clean_housing_data


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def run_feature_significance_tests(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run p-value tests between each raw feature and the target.

    Numeric features use Pearson correlation. Categorical features use one-way
    ANOVA across categories.
    """
    clean_df = clean_housing_data(df)
    rows: list[dict[str, object]] = []

    for feature in FEATURE_SOURCE_COLUMNS:
        if feature in CATEGORICAL_SOURCE_COLUMNS:
            groups = [
                group[target].to_numpy()
                for _, group in clean_df.groupby(feature, observed=True)
                if len(group) >= 2
            ]
            if len(groups) < 2:
                continue

            statistic, p_value = stats.f_oneway(*groups)
            test_name = "one_way_anova"
        else:
            statistic, p_value = stats.pearsonr(clean_df[feature], clean_df[target])
            test_name = "pearson_correlation"

        rows.append(
            {
                "feature": feature,
                "test": test_name,
                "statistic": _safe_float(statistic),
                "p_value": _safe_float(p_value),
                "significant_at_alpha": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)


def paired_error_t_test(
    y_true: pd.Series | np.ndarray,
    candidate_predictions: pd.Series | np.ndarray,
    baseline_predictions: pd.Series | np.ndarray,
) -> dict[str, object]:
    """Compare absolute prediction errors with a paired t-test."""
    y_true_array = np.asarray(y_true, dtype=float)
    candidate_error = np.abs(y_true_array - np.asarray(candidate_predictions, dtype=float))
    baseline_error = np.abs(y_true_array - np.asarray(baseline_predictions, dtype=float))

    statistic, p_value = stats.ttest_rel(
        candidate_error,
        baseline_error,
        nan_policy="omit",
    )

    return {
        "test": "paired_t_test_absolute_error",
        "statistic": _safe_float(statistic),
        "p_value": _safe_float(p_value),
        "mean_error_delta": _safe_float(candidate_error.mean() - baseline_error.mean()),
        "interpretation": (
            "Negative mean_error_delta means the candidate model has lower "
            "absolute error than the baseline on the same rows."
        ),
    }
