import json
from types import SimpleNamespace

import pytest
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


def test_failed_training_quality_gate_does_not_persist_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    result = SimpleNamespace(
        metrics={"r2": 0.1},
        baseline_metrics={},
        cross_validation={},
        residual_test={},
    )
    monkeypatch.setattr(
        "us_housing_price_prediction.cli.train_and_evaluate",
        lambda **_kwargs: (object(), result),
    )
    model_path = tmp_path / "failed-model.joblib"
    metrics_path = tmp_path / "failed-metrics.json"

    exit_code = main(
        [
            "train",
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
            "--min-r2",
            "0.45",
        ]
    )

    assert exit_code == 2
    assert "quality gate failed" in capsys.readouterr().out
    assert not model_path.exists()
    assert not metrics_path.exists()


@pytest.mark.parametrize("min_r2", ["nan", "inf", "-inf"])
def test_training_rejects_non_finite_min_r2(min_r2, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["train", f"--min-r2={min_r2}"])

    assert error.value.code == 2
    assert "must be a finite number" in capsys.readouterr().err


@pytest.mark.parametrize("measured_r2", [float("nan"), float("inf"), float("-inf")])
def test_training_rejects_non_finite_measured_r2_before_persistence(
    tmp_path,
    monkeypatch,
    capsys,
    measured_r2,
) -> None:
    result = SimpleNamespace(
        metrics={"r2": measured_r2},
        baseline_metrics={},
        cross_validation={},
        residual_test={},
    )
    monkeypatch.setattr(
        "us_housing_price_prediction.cli.train_and_evaluate",
        lambda **_kwargs: (object(), result),
    )
    model_path = tmp_path / "invalid-model.joblib"
    metrics_path = tmp_path / "invalid-metrics.json"

    exit_code = main(
        [
            "train",
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
        ]
    )

    assert exit_code == 2
    assert "measured R2 must be finite" in capsys.readouterr().out
    assert not model_path.exists()
    assert not metrics_path.exists()
