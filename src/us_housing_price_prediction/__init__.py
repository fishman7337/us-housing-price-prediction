"""Reusable package for US housing price prediction workflows."""

from us_housing_price_prediction.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from us_housing_price_prediction.modeling import build_pipeline

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_MODEL_PATH",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "build_pipeline",
]

__version__ = "0.1.0"
