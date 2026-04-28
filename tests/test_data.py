import pandas as pd
import pytest

from us_housing_price_prediction.config import ID_COLUMN, TARGET_COLUMN
from us_housing_price_prediction.data import (
    DataValidationError,
    load_housing_data,
    split_features_target,
)


def test_load_housing_data_normalizes_schema() -> None:
    df = load_housing_data()

    assert len(df) == 545
    assert ID_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].min() > 0
    assert set(df["renovation_status"]) == {
        "furnished",
        "semi-furnished",
        "unfurnished",
    }


def test_split_features_target_prevents_identifier_and_target_leakage() -> None:
    df = load_housing_data()
    X, y = split_features_target(df)

    assert TARGET_COLUMN not in X.columns
    assert ID_COLUMN not in X.columns
    assert len(X) == len(y) == 545


def test_invalid_values_raise_clear_validation_error() -> None:
    df = load_housing_data()
    df.loc[0, "house_area_sqm"] = -1

    with pytest.raises(DataValidationError, match="house area must be positive"):
        split_features_target(df)


def test_original_column_names_are_supported() -> None:
    raw = pd.DataFrame(
        [
            {
                "House ID": 1,
                "City": "Chicago",
                "House Area (sqm)": 100.0,
                "No. of Bedrooms": 2,
                "No. of Toilets": 1,
                "Stories": 1,
                "Renovation Status": "Furnished",
                "Price ($)": 250000,
            }
        ]
    )

    X, y = split_features_target(raw)

    assert X.loc[0, "renovation_status"] == "furnished"
    assert y.iloc[0] == 250000
