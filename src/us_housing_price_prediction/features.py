"""Feature engineering transformers for the modeling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from us_housing_price_prediction.config import (
    DROP_AFTER_ENGINEERING_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    NUMERIC_SOURCE_COLUMNS,
)
from us_housing_price_prediction.data import ensure_columns


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived housing predictors and optionally drop redundant sources."""

    def __init__(self, drop_source_features: bool = True) -> None:
        """Configure whether redundant source predictors are removed.

        Args:
            drop_source_features: Remove source columns superseded by engineered features.
        """
        self.drop_source_features = drop_source_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HousingFeatureEngineer:
        """Validate the source feature schema without learning state.

        Args:
            X: Input housing features.
            y: Optional target values accepted for scikit-learn compatibility.

        Returns:
            This fitted transformer instance.
        """
        ensure_columns(X, NUMERIC_SOURCE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio, room-count, and interaction features.

        Args:
            X: Housing features containing the required numeric source columns.

        Returns:
            A copied frame containing engineered predictors.
        """
        ensure_columns(X, NUMERIC_SOURCE_COLUMNS)
        transformed = X.copy()

        bedrooms = pd.to_numeric(transformed["bedrooms"], errors="coerce")
        toilets = pd.to_numeric(transformed["toilets"], errors="coerce")
        stories = pd.to_numeric(transformed["stories"], errors="coerce")
        area = pd.to_numeric(transformed["house_area_sqm"], errors="coerce")

        safe_bedrooms = bedrooms.replace(0, np.nan)
        transformed["area_per_bedroom"] = area / safe_bedrooms
        transformed["toilet_to_bedroom_ratio"] = toilets / safe_bedrooms
        transformed["total_rooms"] = bedrooms + toilets
        transformed["stories_house_area_interaction"] = stories * area

        transformed[ENGINEERED_NUMERIC_COLUMNS] = transformed[ENGINEERED_NUMERIC_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        )

        if self.drop_source_features:
            transformed = transformed.drop(
                columns=DROP_AFTER_ENGINEERING_COLUMNS,
                errors="ignore",
            )

        return transformed

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        """Return output feature names after engineering and optional dropping.

        Args:
            input_features: Feature names supplied by the enclosing scikit-learn pipeline.

        Returns:
            Output feature names in transformer order.
        """
        if input_features is None:
            return np.array([], dtype=object)

        features = list(input_features)
        for engineered_column in ENGINEERED_NUMERIC_COLUMNS:
            if engineered_column not in features:
                features.append(engineered_column)

        if self.drop_source_features:
            features = [
                feature for feature in features if feature not in DROP_AFTER_ENGINEERING_COLUMNS
            ]

        return np.asarray(features, dtype=object)
