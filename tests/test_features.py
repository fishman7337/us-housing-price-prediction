import pandas as pd

from us_housing_price_prediction.features import HousingFeatureEngineer


def test_feature_engineering_adds_features_and_drops_redundant_sources() -> None:
    X = pd.DataFrame(
        [
            {
                "city": "Chicago",
                "house_area_sqm": 120.0,
                "bedrooms": 3,
                "toilets": 2,
                "stories": 2,
                "renovation_status": "furnished",
            }
        ]
    )

    transformed = HousingFeatureEngineer(drop_source_features=True).fit_transform(X)

    assert transformed.loc[0, "area_per_bedroom"] == 40.0
    assert transformed.loc[0, "toilet_to_bedroom_ratio"] == 2 / 3
    assert transformed.loc[0, "total_rooms"] == 5
    assert transformed.loc[0, "stories_house_area_interaction"] == 240.0
    assert "bedrooms" not in transformed.columns
    assert "toilets" not in transformed.columns
    assert "stories" not in transformed.columns
    assert "city" in transformed.columns


def test_feature_engineering_can_keep_sources_for_experiments() -> None:
    X = pd.DataFrame(
        [
            {
                "city": "Denver",
                "house_area_sqm": 90.0,
                "bedrooms": 3,
                "toilets": 1,
                "stories": 1,
                "renovation_status": "semi-furnished",
            }
        ]
    )

    transformed = HousingFeatureEngineer(drop_source_features=False).fit_transform(X)

    assert {"bedrooms", "toilets", "stories"}.issubset(transformed.columns)
