import json

from sklearn.linear_model import Ridge

from us_housing_price_prediction.cli import main
from us_housing_price_prediction.data import load_housing_data, split_features_target
from us_housing_price_prediction.modeling import build_pipeline, load_model, save_model


def test_validate_data_cli_outputs_row_count(capsys) -> None:
    exit_code = main(["validate-data"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output == {"status": "ok", "rows": 545}


def test_predict_cli_uses_saved_pipeline(tmp_path, capsys) -> None:
    df = load_housing_data()
    X, y = split_features_target(df)
    model = build_pipeline(regressor=Ridge(alpha=1.0))
    model.fit(X, y)

    model_path = tmp_path / "housing-price-regressor.joblib"
    save_model(model, model_path)

    exit_code = main(
        [
            "predict",
            "--model-path",
            str(model_path),
            "--city",
            "Chicago",
            "--house-area-sqm",
            "742",
            "--bedrooms",
            "4",
            "--toilets",
            "2",
            "--stories",
            "3",
            "--renovation-status",
            "furnished",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    reloaded_model = load_model(model_path)

    assert exit_code == 0
    assert output["predicted_price_usd"] > 0
    assert reloaded_model.predict(X.head(1)).shape == (1,)
