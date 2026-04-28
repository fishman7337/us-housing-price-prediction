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
        self.drop_source_features = drop_source_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HousingFeatureEngineer:
        ensure_columns(X, NUMERIC_SOURCE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        transformed[ENGINEERED_NUMERIC_COLUMNS] = transformed[
            ENGINEERED_NUMERIC_COLUMNS
        ].replace([np.inf, -np.inf], np.nan)

        if self.drop_source_features:
            transformed = transformed.drop(
                columns=DROP_AFTER_ENGINEERING_COLUMNS,
                errors="ignore",
            )

        return transformed

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)

        features = list(input_features)
        for engineered_column in ENGINEERED_NUMERIC_COLUMNS:
            if engineered_column not in features:
                features.append(engineered_column)

        if self.drop_source_features:
            features = [
                feature
                for feature in features
                if feature not in DROP_AFTER_ENGINEERING_COLUMNS
            ]

        return np.asarray(features, dtype=object)
