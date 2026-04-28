"""Shared configuration for project paths, schema, and reproducibility."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing_price_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "housing-price-regressor.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

ID_COLUMN = "house_id"
TARGET_COLUMN = "price_usd"

RAW_TO_CANONICAL_COLUMNS = {
    "House ID": ID_COLUMN,
    "City": "city",
    "House Area (sqm)": "house_area_sqm",
    "No. of Bedrooms": "bedrooms",
    "No. of Toilets": "toilets",
    "Stories": "stories",
    "Renovation Status": "renovation_status",
    "Price ($)": TARGET_COLUMN,
}

FEATURE_SOURCE_COLUMNS = [
    "city",
    "house_area_sqm",
    "bedrooms",
    "toilets",
    "stories",
    "renovation_status",
]

NUMERIC_SOURCE_COLUMNS = [
    "house_area_sqm",
    "bedrooms",
    "toilets",
    "stories",
]

CATEGORICAL_SOURCE_COLUMNS = [
    "city",
    "renovation_status",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "area_per_bedroom",
    "toilet_to_bedroom_ratio",
    "total_rooms",
    "stories_house_area_interaction",
]

# These fields are intentionally removed after deriving richer versions. This
# reduces redundant signal and makes linear diagnostics less collinear.
DROP_AFTER_ENGINEERING_COLUMNS = [
    "bedrooms",
    "toilets",
    "stories",
]

REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_SOURCE_COLUMNS, TARGET_COLUMN]
