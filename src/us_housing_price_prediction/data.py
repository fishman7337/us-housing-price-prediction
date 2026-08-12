"""Data loading and validation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from us_housing_price_prediction.config import (
    DEFAULT_DATA_PATH,
    FEATURE_SOURCE_COLUMNS,
    ID_COLUMN,
    RANDOM_STATE,
    RAW_TO_CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DataValidationError(ValueError):
    """Raised when the housing dataset does not meet the expected contract."""


def load_housing_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the housing data from CSV and return a validated canonical dataframe."""
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = pd.read_csv(data_path)
    return clean_housing_data(raw)


def clean_housing_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, data types, categories, and duplicate rows."""
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.rename(columns=RAW_TO_CANONICAL_COLUMNS)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise DataValidationError("Missing required columns: " + ", ".join(missing_columns))

    df = df[REQUIRED_COLUMNS].copy()

    numeric_columns = [
        ID_COLUMN,
        "house_area_sqm",
        "bedrooms",
        "toilets",
        "stories",
        TARGET_COLUMN,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df[ID_COLUMN] = df[ID_COLUMN].astype("int64")
    df["bedrooms"] = df["bedrooms"].astype("int64")
    df["toilets"] = df["toilets"].astype("int64")
    df["stories"] = df["stories"].astype("int64")

    df["city"] = df["city"].astype(str).str.strip()
    df["renovation_status"] = df["renovation_status"].astype(str).str.strip().str.lower()

    df = df.drop_duplicates().reset_index(drop=True)
    validate_housing_data(df)
    return df


def validate_housing_data(df: pd.DataFrame) -> None:
    """Validate schema and domain rules for a canonical housing dataframe."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise DataValidationError("Missing required columns: " + ", ".join(missing_columns))

    errors: list[str] = []
    positive_columns = {
        "house_area_sqm": "house area must be positive",
        "bedrooms": "bedroom count must be positive",
        "toilets": "toilet count must be positive",
        "stories": "story count must be positive",
        TARGET_COLUMN: "price must be positive",
    }
    for column, message in positive_columns.items():
        if (df[column] <= 0).any():
            errors.append(message)

    if df[ID_COLUMN].duplicated().any():
        errors.append("house_id values must be unique")

    allowed_renovation_statuses = {"furnished", "semi-furnished", "unfurnished"}
    unknown_statuses = set(df["renovation_status"]) - allowed_renovation_statuses
    if unknown_statuses:
        errors.append("unknown renovation_status values: " + ", ".join(sorted(unknown_statuses)))

    if df.isna().any().any():
        null_columns = df.columns[df.isna().any()].tolist()
        errors.append("null values found in: " + ", ".join(null_columns))

    if errors:
        raise DataValidationError("; ".join(errors))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target without identifier or target leakage."""
    clean_df = clean_housing_data(df)
    X = clean_df[FEATURE_SOURCE_COLUMNS].copy()
    y = clean_df[TARGET_COLUMN].copy()
    return X, y


def make_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split from a canonical dataframe."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if a dataframe is missing expected columns."""
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise DataValidationError("Missing columns: " + ", ".join(missing))
