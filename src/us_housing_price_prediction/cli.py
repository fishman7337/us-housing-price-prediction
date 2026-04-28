"""Command-line entrypoints for the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from us_housing_price_prediction.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
)
from us_housing_price_prediction.data import load_housing_data
from us_housing_price_prediction.modeling import (
    load_model,
    save_metrics,
    save_model,
    train_and_evaluate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="us-housing-price-prediction",
        description="Train and use the US housing price prediction pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data", help="Validate raw data.")
    validate_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
    train_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    train_parser.add_argument(
        "--min-r2",
        type=float,
        default=None,
        help="Optional quality gate for test-set R2.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict one house price.")
    predict_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--city", required=True)
    predict_parser.add_argument("--house-area-sqm", type=float, required=True)
    predict_parser.add_argument("--bedrooms", type=int, required=True)
    predict_parser.add_argument("--toilets", type=int, required=True)
    predict_parser.add_argument("--stories", type=int, required=True)
    predict_parser.add_argument(
        "--renovation-status",
        required=True,
        choices=["furnished", "semi-furnished", "unfurnished"],
    )

    return parser


def _validate_data(args: argparse.Namespace) -> int:
    df = load_housing_data(args.data_path)
    print(json.dumps({"status": "ok", "rows": len(df)}, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    model, result = train_and_evaluate(data_path=args.data_path)
    save_model(model, args.model_path)
    save_metrics(result, args.metrics_path)

    output = {
        "model_path": str(args.model_path),
        "metrics_path": str(args.metrics_path),
        "metrics": result.metrics,
        "baseline_metrics": result.baseline_metrics,
        "cross_validation": result.cross_validation,
        "residual_test": result.residual_test,
    }
    print(json.dumps(output, indent=2))

    if args.min_r2 is not None and result.metrics["r2"] < args.min_r2:
        print(
            f"R2 quality gate failed: {result.metrics['r2']:.4f} < {args.min_r2:.4f}"
        )
        return 2

    return 0


def _predict(args: argparse.Namespace) -> int:
    model = load_model(args.model_path)
    row = pd.DataFrame(
        [
            {
                "city": args.city,
                "house_area_sqm": args.house_area_sqm,
                "bedrooms": args.bedrooms,
                "toilets": args.toilets,
                "stories": args.stories,
                "renovation_status": args.renovation_status,
            }
        ]
    )
    prediction = float(model.predict(row)[0])
    print(json.dumps({"predicted_price_usd": prediction}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-data":
        return _validate_data(args)
    if args.command == "train":
        return _train(args)
    if args.command == "predict":
        return _predict(args)

    parser.error(f"Unknown command: {args.command}")
    return 2
