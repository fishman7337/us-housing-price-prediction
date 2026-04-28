import numpy as np
from sklearn.linear_model import Ridge

from us_housing_price_prediction.data import load_housing_data, make_train_test_split
from us_housing_price_prediction.modeling import build_pipeline, evaluate_regression


def test_pipeline_fits_and_predicts_without_target_leakage() -> None:
    df = load_housing_data()
    X_train, X_test, y_train, y_test = make_train_test_split(df)
    pipeline = build_pipeline(regressor=Ridge(alpha=1.0))

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test.head(10))

    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()
    assert set(X_train.columns) == {
        "city",
        "house_area_sqm",
        "bedrooms",
        "toilets",
        "stories",
        "renovation_status",
    }
    assert "price_usd" not in X_train.columns

    metrics = evaluate_regression(y_test.head(10), predictions)
    assert set(metrics) == {
        "r2",
        "mae",
        "mse",
        "rmse",
        "mape",
        "explained_variance",
    }
